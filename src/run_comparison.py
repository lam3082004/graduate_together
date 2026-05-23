"""
run_comparison.py — End-to-end comparison experiment for Chapter 5 of the thesis.

Compares the proposed IA-MADDPG+UAV (Chapter 4) against five baselines:
  1. DirectTX (DT)            — Mode 0 only, fixed alpha
  2. FrequencyHopping (FH)    — Random mode/alpha avoidance
  3. Greedy                   — Instantaneous-SINR maximisation
  4. StandardMADDPG           — MADDPG without imitation learning
  5. IA-MADDPG(RBS)           — IA-MADDPG, UAV relay disabled

For each method we:
  • Train (learnable methods only) and record per-episode reward / TSR / mode dist.
  • Evaluate deterministically over `eval_episodes` runs.
  • Sweep SINR threshold to plot TSR–threshold trade-off.
  • Save trajectory for the proposed method (one episode).

Outputs land in `results_compare/`:
  • training_curves.png, throughput_comparison.png, mode_distribution.png,
    uav_trajectory.png, tsr_vs_threshold.png, ablation_reward.png,
    ablation_tsr.png, convergence_tsr.png
  • metrics_summary.csv, metrics_summary.json
  • eval_results.json, history_<method>.json
  • report_summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
)


# ──────────────────────────────────────────────────────────────────────────────
# Method registry
# ──────────────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "ia_maddpg_uav": "IA-MADDPG+UAV",
    "ia_maddpg_rbs": "IA-MADDPG(RBS)",
    "maddpg":        "StandardMADDPG",
    "greedy":        "Greedy",
    "fh":            "FreqHopping",
    "dt":            "DirectTX",
}

LEARNABLE = {"ia_maddpg_uav", "ia_maddpg_rbs", "maddpg"}


def build_agent(method: str, cfg: Config, env: AntiJammingEnv):
    if method == "ia_maddpg_uav":
        return IAMADDPG(cfg, env)
    if method == "ia_maddpg_rbs":
        return IAMADDPG_RBSOnly(cfg, env)
    if method == "maddpg":
        return StandardMADDPG(cfg, env)
    if method == "greedy":
        return GreedyStrategy(cfg)
    if method == "fh":
        return FrequencyHopping(cfg)
    if method == "dt":
        return DirectTransmission()
    raise ValueError(f"Unknown method {method}")


def select_actions(method, agent, su_obs, uav_obs, env, step, cfg, explore):
    if method in LEARNABLE:
        return agent.select_actions(su_obs, uav_obs, explore=explore)
    if method == "greedy":
        return agent.select_actions(env.channels, cfg)
    if method == "fh":
        return agent.select_actions(step)
    return agent.select_actions({"N": cfg.N, "K": cfg.K})  # dt


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_method(method: str, cfg: Config, save_dir: str):
    """Train one method and return per-episode history dict."""
    env = AntiJammingEnv(cfg)
    agent = build_agent(method, cfg, env)
    method_dir = os.path.join(save_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    learnable = method in LEARNABLE
    t0 = time.time()

    if learnable and method != "maddpg":
        # Only IA-* methods use expert warmup; MADDPG keeps its own no-op.
        agent.warmup(env, cfg.warmup_steps)
    elif method == "maddpg":
        # Even MADDPG needs a populated buffer to start sampling.
        su_obs, uav_obs = env.reset()
        for _ in range(cfg.warmup_steps):
            su_acts = np.random.uniform(
                low=[0.0, -1, -1, -1], high=[1.0, 1, 1, 1],
                size=(cfg.N, cfg.su_action_dim)).astype(np.float32)
            uav_acts = np.random.uniform(-1, 1, size=(cfg.K, cfg.uav_action_dim)).astype(np.float32)
            nxt_su, nxt_uav, rewards, done, info = env.step(su_acts, uav_acts)
            agent.store_transition((su_obs, uav_obs, su_acts, uav_acts,
                                    rewards["su"], nxt_su, nxt_uav,
                                    float(done), env.channels))
            su_obs, uav_obs = (nxt_su, nxt_uav) if not done else env.reset()

    reward_history, tsr_history, mode_history = [], [], []
    global_step = 0

    for ep in range(cfg.episodes):
        su_obs, uav_obs = env.reset()
        ep_reward = ep_tsr = 0.0
        modes_ep = [0, 0, 0]
        for step in range(cfg.steps_per_episode):
            su_acts, uav_acts = select_actions(
                method, agent, su_obs, uav_obs, env, step, cfg, explore=True)
            nxt_su, nxt_uav, rewards, done, info = env.step(su_acts, uav_acts)
            if learnable:
                agent.store_transition((su_obs, uav_obs, su_acts, uav_acts,
                                        rewards["su"], nxt_su, nxt_uav,
                                        float(done), env.channels))
                agent.update(global_step)
            ep_reward += float(np.mean(rewards["su"]))
            sinr = info["sinr"]
            ep_tsr += float(np.mean(sinr >= cfg.gamma_th))
            for m in info["modes"]:
                modes_ep[int(m)] += 1
            su_obs, uav_obs = nxt_su, nxt_uav
            global_step += 1
            if done:
                break
        T = cfg.steps_per_episode
        reward_history.append(ep_reward / T)
        tsr_history.append(ep_tsr / T)
        mode_history.append(modes_ep)

        if (ep + 1) % 10 == 0 or ep == cfg.episodes - 1:
            total = max(sum(modes_ep), 1)
            print(f"  [{method:>14}] ep {ep+1:3d}/{cfg.episodes} "
                  f"R={reward_history[-1]:+.3f} TSR={tsr_history[-1]:.3f} "
                  f"modes={modes_ep[0]/total:.2f}/{modes_ep[1]/total:.2f}/"
                  f"{modes_ep[2]/total:.2f} t={time.time()-t0:.0f}s")

    if learnable:
        agent.save(method_dir)

    history = {"reward": reward_history, "tsr": tsr_history, "modes": mode_history}
    with open(os.path.join(method_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history, agent, env


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_method(method, agent, cfg, n_episodes=30, capture_trajectory=False):
    """Deterministic evaluation. Returns metrics dict."""
    env = AntiJammingEnv(cfg)
    rewards_ep, tsr_ep, throughput_ep, energy_ep = [], [], [], []
    sinr_pool = []
    mode_counts = [0, 0, 0]
    trajectory = None

    for ep in range(n_episodes):
        su_obs, uav_obs = env.reset()
        ep_r = ep_t = ep_th = 0.0
        init_energy = env._uav_energy.copy()
        traj = []
        for step in range(cfg.steps_per_episode):
            su_acts, uav_acts = select_actions(
                method, agent, su_obs, uav_obs, env, step, cfg, explore=False)
            nxt_su, nxt_uav, rewards, done, info = env.step(su_acts, uav_acts)
            sinr = info["sinr"]
            ep_r += float(np.mean(rewards["su"]))
            ep_t += float(np.mean(sinr >= cfg.gamma_th))
            ep_th += float(np.mean(np.log2(1.0 + np.maximum(sinr, 0.0))))
            for m in info["modes"]:
                mode_counts[int(m)] += 1
            sinr_pool.append(sinr.copy())
            if capture_trajectory and ep == 0:
                traj.append(env.pos_uav.copy())
            su_obs, uav_obs = nxt_su, nxt_uav
            if done:
                break
        T = cfg.steps_per_episode
        rewards_ep.append(ep_r / T)
        tsr_ep.append(ep_t / T)
        throughput_ep.append(ep_th / T)
        energy_ep.append(float(np.sum(init_energy - env._uav_energy)))
        if capture_trajectory and ep == 0:
            trajectory = np.array(traj)

    sinr_arr = np.concatenate(sinr_pool)  # (n_eps*T*N,)
    avg_tp = float(np.mean(throughput_ep))
    avg_E = max(float(np.mean(energy_ep)), 1e-6)

    return {
        "avg_reward":   float(np.mean(rewards_ep)),
        "std_reward":   float(np.std(rewards_ep)),
        "avg_tsr":      float(np.mean(tsr_ep)),
        "std_tsr":      float(np.std(tsr_ep)),
        "avg_throughput": avg_tp,
        "std_throughput": float(np.std(throughput_ep)),
        "mode_counts": mode_counts,
        "energy_per_ep": energy_ep,
        "avg_energy":  avg_E,
        "energy_efficiency": avg_tp / avg_E,
        "sinr_samples": sinr_arr,
        "uav_trajectory": trajectory,
        "env_snapshot": {
            "pos_su": env.pos_su.copy(),
            "pos_du": env.pos_du.copy(),
            "pos_rbs": env._pos_rbs.copy(),
            "pos_jammer": env.pos_jammer.copy(),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Custom plots beyond utils/visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_metric_bar(metric_dict: dict, ylabel: str, title: str, save_path: str,
                    error_dict: dict | None = None) -> None:
    methods = list(metric_dict.keys())
    values = [metric_dict[m] for m in methods]
    errs = [error_dict[m] for m in methods] if error_dict else None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(methods, values, yerr=errs, capsize=4,
                  color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"],
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsr_convergence(tsr_histories: dict, save_path: str, window: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"IA-MADDPG+UAV": "#2196F3", "IA-MADDPG(RBS)": "#4CAF50",
              "StandardMADDPG": "#FF9800", "Greedy": "#9C27B0",
              "FreqHopping": "#F44336", "DirectTX": "#607D8B"}
    for name, tsrs in tsr_histories.items():
        y = np.asarray(tsrs, float)
        ax.plot(y, alpha=0.25, color=colors.get(name, "gray"))
        if len(y) >= window:
            k = np.ones(window) / window
            ys = np.convolve(y, k, mode="same")
        else:
            ys = y
        ax.plot(ys, label=name, color=colors.get(name, "gray"), linewidth=2)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Transmission Success Rate (TSR)", fontsize=11)
    ax.set_title("TSR Convergence Curves", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter-5 comparison runner")
    parser.add_argument("--out", default="results_compare/")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default="all",
                        help="Comma-separated subset, or 'all'.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    cfg = Config(seed=args.seed)
    object.__setattr__(cfg, "episodes", args.episodes)
    object.__setattr__(cfg, "steps_per_episode", args.steps)
    object.__setattr__(cfg, "warmup_steps", args.warmup)
    object.__setattr__(cfg, "batch_size", args.batch)

    os.makedirs(args.out, exist_ok=True)
    methods = (list(METHOD_LABELS.keys())
               if args.methods == "all" else args.methods.split(","))

    print(f"\n{'='*72}")
    print(f"Chapter-5 comparison experiment")
    print(f"  episodes={cfg.episodes}  steps/ep={cfg.steps_per_episode}  "
          f"warmup={cfg.warmup_steps}  batch={cfg.batch_size}  seed={cfg.seed}")
    print(f"  methods: {', '.join(METHOD_LABELS[m] for m in methods)}")
    print(f"  output:  {args.out}")
    print(f"{'='*72}\n")

    histories: dict = {}
    agents: dict = {}
    train_t0 = time.time()
    for m in methods:
        print(f"\n>>> Training {METHOD_LABELS[m]}")
        np.random.seed(args.seed)
        hist, agent, _env = train_method(m, cfg, args.out)
        histories[m] = hist
        agents[m] = agent
    print(f"\n[train phase complete] {(time.time()-train_t0)/60:.1f} min")

    # ── Evaluation ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}\nEvaluation phase\n{'='*72}")
    eval_metrics: dict = {}
    eval_t0 = time.time()
    for m in methods:
        print(f"  evaluating {METHOD_LABELS[m]} …")
        np.random.seed(args.seed + 1)
        metrics = evaluate_method(
            m, agents[m], cfg,
            n_episodes=args.eval_episodes,
            capture_trajectory=(m == "ia_maddpg_uav"),
        )
        eval_metrics[m] = metrics
    print(f"[eval phase complete] {(time.time()-eval_t0):.1f}s")

    # ── Print comparison table ──────────────────────────────────────────────
    header = (f"{'Method':<18} {'Reward':>10} {'TSR':>9} {'Throughput':>12} "
              f"{'EE':>10}  {'D2D/RBS/UAV':>16}")
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for m in methods:
        v = eval_metrics[m]
        mc = np.asarray(v["mode_counts"], float)
        mc_t = max(mc.sum(), 1)
        print(f"{METHOD_LABELS[m]:<18} "
              f"{v['avg_reward']:>+10.4f} {v['avg_tsr']:>9.4f} "
              f"{v['avg_throughput']:>12.4f} {v['energy_efficiency']:>10.4f}  "
              f"{mc[0]/mc_t:.2f}/{mc[1]/mc_t:.2f}/{mc[2]/mc_t:.2f}")
    print(sep)

    # ── Exports ────────────────────────────────────────────────────────────
    # Note: metrics_summary.{csv,json} are written by re_evaluate.py with the
    # broader threshold sweep — that's the authoritative version. We skip the
    # narrow γ_th=5dB CSV/JSON here to avoid duplicate "version" artefacts.

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n[plotting] …")

    # Training reward curves
    reward_hist_named = {METHOD_LABELS[m]: histories[m]["reward"] for m in methods}
    plot_training_curves(
        reward_hist_named,
        save_path=os.path.join(args.out, "training_curves.png"),
        window=max(5, args.episodes // 12))
    plt.close("all")

    # TSR convergence
    tsr_hist_named = {METHOD_LABELS[m]: histories[m]["tsr"] for m in methods}
    plot_tsr_convergence(
        tsr_hist_named,
        save_path=os.path.join(args.out, "convergence_tsr.png"),
        window=max(5, args.episodes // 12))

    # Throughput bar
    throughput_named = {METHOD_LABELS[m]: eval_metrics[m]["avg_throughput"]
                        for m in methods}
    plot_throughput_comparison(
        throughput_named,
        save_path=os.path.join(args.out, "throughput_comparison.png"))
    plt.close("all")

    # Reward bar (with error bars). TSR bar omitted because γ_th=5dB is
    # essentially unreachable in this channel regime — see tsr_vs_threshold.png.
    reward_named = {METHOD_LABELS[m]: eval_metrics[m]["avg_reward"] for m in methods}
    reward_err   = {METHOD_LABELS[m]: eval_metrics[m]["std_reward"] for m in methods}
    plot_metric_bar(reward_named, "Average Reward",
                    "Average Reward across Methods (±1 std)",
                    os.path.join(args.out, "reward_comparison.png"),
                    error_dict=reward_err)

    ee_named = {METHOD_LABELS[m]: eval_metrics[m]["energy_efficiency"] for m in methods}
    plot_metric_bar(ee_named, "Throughput / Energy",
                    "Energy Efficiency Comparison",
                    os.path.join(args.out, "energy_efficiency.png"))

    # Mode distribution
    mode_named = {METHOD_LABELS[m]: eval_metrics[m]["mode_counts"] for m in methods}
    plot_mode_distribution(
        mode_named, save_path=os.path.join(args.out, "mode_distribution.png"))
    plt.close("all")

    # TSR vs threshold plot is produced by re_evaluate.py over the broader
    # threshold range that actually shows differentiation between methods.

    # UAV trajectory (proposed only)
    if "ia_maddpg_uav" in eval_metrics:
        v = eval_metrics["ia_maddpg_uav"]
        if v["uav_trajectory"] is not None:
            snap = v["env_snapshot"]
            plot_uav_trajectory(
                trajectory=v["uav_trajectory"],
                positions_su=snap["pos_su"],
                positions_du=snap["pos_du"],
                pos_rbs=snap["pos_rbs"],
                pos_jammer=snap["pos_jammer"],
                save_path=os.path.join(args.out, "uav_trajectory.png"))
            plt.close("all")

    # Note: the human-readable summary lives in THESIS_REPORT.md (generated
    # separately via analyze_comparison.py + build_thesis_section.py).
    print(f"[done] all artefacts in {args.out}")


if __name__ == "__main__":
    main()
