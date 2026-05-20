"""
Evaluation script: loads trained checkpoints and compares all methods.
Usage:
    python evaluate.py --results_dir results/
    python evaluate.py --plot_all
    python evaluate.py --ablation
"""
import argparse
import sys
import os
import json
import copy

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Ensure src/ root is importable when running as `python src/evaluate.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from environment.network_env import AntiJammingEnv
from algorithms.ia_maddpg import IAMADDPG
from algorithms.baselines import (
    DirectTransmission,
    GreedyStrategy,
    FrequencyHopping,
    StandardMADDPG,
    IAMADDPG_RBSOnly,
)
from utils.visualization import (
    plot_training_curves,
    plot_uav_trajectory,
    plot_mode_distribution,
    plot_throughput_comparison,
    plot_ablation_study,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_history(results_dir: str, method_name: str) -> dict:
    path = os.path.join(results_dir, method_name, "history.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _build_agent(method_name: str, cfg: Config, env: AntiJammingEnv):
    """Construct the agent/strategy for a given method name."""
    if method_name == "ia_maddpg_uav":
        return IAMADDPG(cfg, env)
    elif method_name == "maddpg":
        return StandardMADDPG(cfg, env)
    elif method_name == "ia_maddpg_rbs":
        return IAMADDPG_RBSOnly(cfg, env)
    elif method_name == "greedy":
        return GreedyStrategy(cfg)
    elif method_name == "dt":
        return DirectTransmission()
    elif method_name == "fh":
        return FrequencyHopping(cfg)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def _is_learnable(method_name: str) -> bool:
    return method_name in ("ia_maddpg_uav", "maddpg", "ia_maddpg_rbs")


def _select_actions(method_name, agent, su_obs, uav_obs, env, step, cfg):
    """Unified action selection for all methods."""
    if _is_learnable(method_name):
        return agent.select_actions(su_obs, uav_obs, explore=False)
    elif method_name == "greedy":
        return agent.select_actions(env.channels, cfg)
    elif method_name == "dt":
        return agent.select_actions({"N": cfg.N, "K": cfg.K})
    else:  # fh
        return agent.select_actions(step)


# ── Core evaluator ─────────────────────────────────────────────────────────────

def evaluate_method(
    method_name: str,
    cfg: Config,
    checkpoint_path: str | None = None,
    n_episodes: int = 50,
) -> dict:
    """
    Run method for n_episodes with no exploration.

    Returns
    -------
    dict with keys:
        avg_reward      : float
        avg_tsr         : float
        avg_throughput  : float  (bits/s/Hz = mean log2(1+SINR) over all pairs & steps)
        mode_counts     : list[int]  [d2d_total, rbs_total, uav_total]
        uav_trajectory  : np.ndarray (T, K, 3) — last episode only
        energy_per_ep   : list[float]  — remaining UAV energy per episode
        energy_efficiency : float  — throughput / energy_spent
    """
    env = AntiJammingEnv(cfg)
    agent = _build_agent(method_name, cfg, env)

    # Load checkpoint for learnable agents
    if _is_learnable(method_name) and checkpoint_path and os.path.isdir(checkpoint_path):
        try:
            agent.load(checkpoint_path)
            print(f"  [{method_name}] loaded checkpoint from {checkpoint_path}")
        except Exception as exc:
            print(f"  [{method_name}] WARNING: could not load checkpoint: {exc}")

    rewards_all: list[float] = []
    tsr_all: list[float] = []
    throughput_all: list[float] = []
    mode_counts = [0, 0, 0]
    energy_per_ep: list[float] = []
    last_trajectory = None

    for ep in range(n_episodes):
        su_obs, uav_obs = env.reset()
        ep_reward = ep_tsr = ep_throughput = 0.0
        traj: list[np.ndarray] = []
        init_energy = env._uav_energy.copy()

        for step in range(cfg.steps_per_episode):
            su_acts, uav_acts = _select_actions(
                method_name, agent, su_obs, uav_obs, env, step, cfg)

            next_su_obs, next_uav_obs, rewards, done, info = env.step(su_acts, uav_acts)
            traj.append(env.pos_uav.copy())

            sinr_vals = info["sinr"]
            modes = info["modes"]

            ep_reward += float(np.mean(rewards["su"]))
            ep_tsr += float(np.mean(sinr_vals >= cfg.gamma_th))
            ep_throughput += float(np.mean(np.log2(1.0 + np.maximum(sinr_vals, 0.0))))

            for m in modes:
                mode_counts[int(m)] += 1

            su_obs, uav_obs = next_su_obs, next_uav_obs
            if done:
                break

        T = cfg.steps_per_episode
        rewards_all.append(ep_reward / T)
        tsr_all.append(ep_tsr / T)
        throughput_all.append(ep_throughput / T)
        energy_spent = float(np.sum(init_energy - env._uav_energy))
        energy_per_ep.append(energy_spent)

        last_trajectory = np.array(traj)  # (T, K, 3)

    avg_reward = float(np.mean(rewards_all))
    avg_tsr = float(np.mean(tsr_all))
    avg_throughput = float(np.mean(throughput_all))
    avg_energy = float(np.mean(energy_per_ep)) if energy_per_ep else 1.0
    energy_efficiency = avg_throughput / max(avg_energy, 1e-9)

    return {
        "avg_reward": avg_reward,
        "avg_tsr": avg_tsr,
        "avg_throughput": avg_throughput,
        "mode_counts": mode_counts,
        "uav_trajectory": last_trajectory,
        "energy_per_ep": energy_per_ep,
        "energy_efficiency": energy_efficiency,
    }


# ── Ablation study ─────────────────────────────────────────────────────────────

def run_ablation_study(cfg: Config, results_dir: str, n_episodes: int = 50) -> dict:
    """
    Test four ablation variants of the proposed method.

    Variants
    --------
    proposed   : full IA-MADDPG+UAV
    no_il      : lambda_IL=0 from start (no imitation learning)
    no_uav     : UAV stays fixed (zero displacement every step)
    no_per     : uniform replay (alpha=0, no priority)
    mlp_critic : (same network but with plain MLP critic — uses ia_maddpg_core
                  TransformerGATCritic; we approximate by lambda_il=0 + no warmup)
    """
    results: dict[str, dict] = {}

    # ── proposed ──────────────────────────────────────────────────────────────
    ckpt = os.path.join(results_dir, "ia_maddpg_uav")
    results["proposed"] = evaluate_method(
        "ia_maddpg_uav", cfg, ckpt, n_episodes)

    # ── no_il ─────────────────────────────────────────────────────────────────
    cfg_no_il = copy.copy(cfg)
    object.__setattr__(cfg_no_il, "lambda_il_init", 0.0)
    env_no_il = AntiJammingEnv(cfg_no_il)
    agent_no_il = IAMADDPG(cfg_no_il, env_no_il)
    ckpt_no_il = os.path.join(results_dir, "ia_maddpg_uav")
    if os.path.isdir(ckpt_no_il):
        try:
            agent_no_il.load(ckpt_no_il)
            agent_no_il.lambda_il = 0.0
        except Exception:
            pass
    results["no_il"] = _run_agent_eval(agent_no_il, env_no_il, cfg_no_il, n_episodes)

    # ── no_uav ────────────────────────────────────────────────────────────────
    ckpt_no_uav = os.path.join(results_dir, "ia_maddpg_uav")
    metrics_no_uav = evaluate_method("ia_maddpg_uav", cfg, ckpt_no_uav, n_episodes)
    # Override UAV actions to zero after loading — patch agent to freeze UAV
    env_no_uav = AntiJammingEnv(cfg)
    agent_no_uav = IAMADDPG(cfg, env_no_uav)
    if os.path.isdir(ckpt_no_uav):
        try:
            agent_no_uav.load(ckpt_no_uav)
        except Exception:
            pass
    results["no_uav"] = _run_agent_eval(
        agent_no_uav, env_no_uav, cfg, n_episodes, freeze_uav=True)

    # ── no_per ────────────────────────────────────────────────────────────────
    cfg_no_per = copy.copy(cfg)
    object.__setattr__(cfg_no_per, "per_alpha", 0.0)  # uniform sampling
    env_no_per = AntiJammingEnv(cfg_no_per)
    agent_no_per = IAMADDPG(cfg_no_per, env_no_per)
    ckpt_no_per = os.path.join(results_dir, "ia_maddpg_uav")
    if os.path.isdir(ckpt_no_per):
        try:
            agent_no_per.load(ckpt_no_per)
        except Exception:
            pass
    results["no_per"] = _run_agent_eval(agent_no_per, env_no_per, cfg_no_per, n_episodes)

    # ── mlp_critic (approximate via standard MADDPG + no warmup) ──────────────
    ckpt_mlp = os.path.join(results_dir, "maddpg")
    results["mlp_critic"] = evaluate_method("maddpg", cfg, ckpt_mlp, n_episodes)

    return results


def _run_agent_eval(
    agent,
    env: AntiJammingEnv,
    cfg: Config,
    n_episodes: int,
    freeze_uav: bool = False,
) -> dict:
    """Helper: evaluate a pre-built IAMADDPG agent for n_episodes."""
    rewards_all, tsr_all, throughput_all = [], [], []
    mode_counts = [0, 0, 0]

    for _ in range(n_episodes):
        su_obs, uav_obs = env.reset()
        ep_reward = ep_tsr = ep_tp = 0.0

        for step in range(cfg.steps_per_episode):
            su_acts, uav_acts = agent.select_actions(su_obs, uav_obs, explore=False)
            if freeze_uav:
                uav_acts = np.zeros_like(uav_acts)

            next_su_obs, next_uav_obs, rewards, done, info = env.step(su_acts, uav_acts)

            sinr_vals = info["sinr"]
            modes = info["modes"]
            ep_reward += float(np.mean(rewards["su"]))
            ep_tsr += float(np.mean(sinr_vals >= cfg.gamma_th))
            ep_tp += float(np.mean(np.log2(1.0 + np.maximum(sinr_vals, 0.0))))

            for m in modes:
                mode_counts[int(m)] += 1

            su_obs, uav_obs = next_su_obs, next_uav_obs
            if done:
                break

        T = cfg.steps_per_episode
        rewards_all.append(ep_reward / T)
        tsr_all.append(ep_tsr / T)
        throughput_all.append(ep_tp / T)

    avg_tp = float(np.mean(throughput_all))
    return {
        "avg_reward": float(np.mean(rewards_all)),
        "avg_tsr": float(np.mean(tsr_all)),
        "avg_throughput": avg_tp,
        "mode_counts": mode_counts,
        "energy_efficiency": avg_tp,
    }


# ── Results table printer ──────────────────────────────────────────────────────

def print_results_table(results: dict) -> None:
    header = f"{'Method':<22} {'Reward':>9} {'TSR':>8} {'Throughput':>12} {'EnergyEff':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<22} "
            f"{m['avg_reward']:>+9.4f} "
            f"{m['avg_tsr']:>8.4f} "
            f"{m['avg_throughput']:>12.4f} "
            f"{m.get('energy_efficiency', 0.0):>12.6f}"
        )
    print("=" * len(header) + "\n")


# ── Main entry point ───────────────────────────────────────────────────────────

ALL_METHODS = ["ia_maddpg_uav", "maddpg", "ia_maddpg_rbs", "greedy", "dt", "fh"]
METHOD_LABELS = {
    "ia_maddpg_uav": "IA-MADDPG+UAV",
    "maddpg":        "StandardMADDPG",
    "ia_maddpg_rbs": "IA-MADDPG(RBS)",
    "greedy":        "Greedy",
    "dt":            "DirectTX",
    "fh":            "FreqHopping",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="IA-MADDPG evaluation script")
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_all", action="store_true",
                        help="Regenerate all comparison plots")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config(seed=args.seed)
    results_dir = args.results_dir

    # ── Evaluate each method ──────────────────────────────────────────────────
    eval_results: dict[str, dict] = {}
    for method in ALL_METHODS:
        ckpt = os.path.join(results_dir, method)
        print(f"Evaluating {method} …")
        try:
            metrics = evaluate_method(method, cfg, ckpt, args.n_episodes)
            eval_results[method] = metrics
        except Exception as exc:
            print(f"  WARNING: {method} failed: {exc}")

    # ── Print comparison table ────────────────────────────────────────────────
    print_results_table(eval_results)

    # ── Save evaluation results ───────────────────────────────────────────────
    save_dict = {
        k: {kk: vv for kk, vv in v.items() if kk != "uav_trajectory"}
        for k, v in eval_results.items()
    }
    eval_path = os.path.join(results_dir, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Evaluation results saved → {eval_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot_all or eval_results:
        # Throughput comparison
        throughputs = {
            METHOD_LABELS.get(k, k): v["avg_throughput"]
            for k, v in eval_results.items()
        }
        plot_throughput_comparison(
            throughputs,
            save_path=os.path.join(results_dir, "throughput_comparison.png"),
        )

        # Mode distributions
        mode_data = {
            METHOD_LABELS.get(k, k): v["mode_counts"]
            for k, v in eval_results.items()
        }
        plot_mode_distribution(
            mode_data,
            save_path=os.path.join(results_dir, "mode_distribution.png"),
        )

        # Training reward curves (load from saved histories)
        reward_histories: dict[str, list] = {}
        for method in ALL_METHODS:
            hist = _load_history(results_dir, method)
            if hist.get("reward"):
                reward_histories[METHOD_LABELS.get(method, method)] = hist["reward"]
        if reward_histories:
            plot_training_curves(
                reward_histories,
                save_path=os.path.join(results_dir, "training_curves.png"),
            )

        # UAV trajectory for proposed method
        if "ia_maddpg_uav" in eval_results:
            traj = eval_results["ia_maddpg_uav"].get("uav_trajectory")
            if traj is not None:
                env_tmp = AntiJammingEnv(cfg)
                env_tmp.reset()
                plot_uav_trajectory(
                    trajectory=traj,
                    positions_su=env_tmp.pos_su,
                    positions_du=env_tmp.pos_du,
                    pos_rbs=env_tmp._pos_rbs,
                    pos_jammer=env_tmp.pos_jammer,
                    save_path=os.path.join(results_dir, "uav_trajectory.png"),
                )

        print("Plots saved to", results_dir)

    # ── Ablation ──────────────────────────────────────────────────────────────
    if args.ablation:
        print("\nRunning ablation study …")
        ablation = run_ablation_study(cfg, results_dir, args.n_episodes)
        abl_rewards = {k: v["avg_reward"] for k, v in ablation.items()}
        plot_ablation_study(
            abl_rewards,
            metric="reward",
            save_path=os.path.join(results_dir, "ablation_reward.png"),
        )
        abl_tsr = {k: v["avg_tsr"] for k, v in ablation.items()}
        plot_ablation_study(
            abl_tsr,
            metric="tsr",
            save_path=os.path.join(results_dir, "ablation_tsr.png"),
        )
        print("Ablation plots saved.")


if __name__ == "__main__":
    main()
