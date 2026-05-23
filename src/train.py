"""
train.py — Main training script (numpy-only, torch-free).

Usage:
    python train.py                    # train proposed method
    python train.py --method maddpg    # train a specific baseline
    python train.py --episodes 600 --seed 42
    python train.py --all              # train all methods sequentially
"""
import argparse
import sys
import os
import time
import json

import numpy as np

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


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)


# ── Per-method training ────────────────────────────────────────────────────────

def train_one_method(method_name: str, cfg: Config, save_dir: str):
    """
    Train a single method for cfg.episodes episodes.

    Returns
    -------
    reward_history : list[float]  — per-episode mean reward
    tsr_history    : list[float]  — per-episode transmission success rate
    mode_history   : list[list]   — per-episode [d2d_count, rbs_count, uav_count]
    """
    env = AntiJammingEnv(cfg)
    method_dir = os.path.join(save_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)

    # ── Build agent/strategy ──────────────────────────────────────────────────
    learnable = method_name in ("ia_maddpg_uav", "maddpg", "ia_maddpg_rbs")
    agent = strategy = None

    if method_name == "ia_maddpg_uav":
        agent = IAMADDPG(cfg, env)
    elif method_name == "maddpg":
        agent = StandardMADDPG(cfg, env)
    elif method_name == "ia_maddpg_rbs":
        agent = IAMADDPG_RBSOnly(cfg, env)
    elif method_name == "greedy":
        strategy = GreedyStrategy(cfg)
    elif method_name == "dt":
        strategy = DirectTransmission()
    elif method_name == "fh":
        strategy = FrequencyHopping(cfg)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    # Warm-up for learnable agents
    if learnable:
        print(f"  [warmup] {cfg.warmup_steps} steps …")
        agent.warmup(env, cfg.warmup_steps)
        print(f"  [warmup] done — buffer size: {len(agent.buffer)}")

    reward_history: list = []
    tsr_history: list = []
    mode_history: list = []
    global_step = 0
    t0 = time.time()

    for ep in range(cfg.episodes):
        su_obs, uav_obs = env.reset()
        ep_reward = 0.0
        ep_tsr = 0.0
        mode_counts = [0, 0, 0]

        for step in range(cfg.steps_per_episode):
            # ── Select actions ────────────────────────────────────────────────
            if learnable:
                su_acts, uav_acts = agent.select_actions(su_obs, uav_obs, explore=True)
            elif method_name == "greedy":
                su_acts, uav_acts = strategy.select_actions(env.channels, cfg)
            elif method_name == "dt":
                su_acts, uav_acts = strategy.select_actions({"N": cfg.N, "K": cfg.K})
            else:  # fh
                su_acts, uav_acts = strategy.select_actions(step)

            # ── Environment step ──────────────────────────────────────────────
            next_su_obs, next_uav_obs, rewards, done, info = env.step(su_acts, uav_acts)

            # ── Store + update (learnable agents) ─────────────────────────────
            if learnable:
                agent.store_transition((
                    su_obs, uav_obs,
                    su_acts, uav_acts,
                    rewards["su"],
                    next_su_obs, next_uav_obs,
                    float(done),
                    env.channels,
                ))
                agent.update(global_step)

            # ── Accumulate metrics ────────────────────────────────────────────
            ep_reward += float(np.mean(rewards["su"]))
            sinr_vals = info["sinr"]
            ep_tsr += float(np.mean(sinr_vals >= cfg.gamma_th))
            for m in info["modes"]:
                mode_counts[int(m)] += 1

            su_obs, uav_obs = next_su_obs, next_uav_obs
            global_step += 1
            if done:
                break

        ep_mean_reward = ep_reward / cfg.steps_per_episode
        ep_mean_tsr    = ep_tsr    / cfg.steps_per_episode
        reward_history.append(ep_mean_reward)
        tsr_history.append(ep_mean_tsr)
        mode_history.append(mode_counts)

        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t0
            mode_total = max(sum(mode_counts), 1)
            print(
                f"  [{method_name}] ep {ep+1:4d}/{cfg.episodes} | "
                f"reward={ep_mean_reward:+.4f} | tsr={ep_mean_tsr:.3f} | "
                f"modes=({mode_counts[0]/mode_total:.2f}/"
                f"{mode_counts[1]/mode_total:.2f}/"
                f"{mode_counts[2]/mode_total:.2f}) | t={elapsed:.0f}s"
            )

        if learnable and (ep + 1) % 100 == 0:
            ckpt_dir = os.path.join(method_dir, f"ep{ep+1}")
            agent.save(ckpt_dir)
            print(f"  [ckpt] saved → {ckpt_dir}")

    if learnable:
        agent.save(method_dir)

    history = {"reward": reward_history, "tsr": tsr_history, "modes": mode_history}
    with open(os.path.join(method_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"  [{method_name}] training complete — history saved.")
    return reward_history, tsr_history, mode_history


# ── Main ───────────────────────────────────────────────────────────────────────

ALL_METHODS = ["ia_maddpg_uav", "maddpg", "ia_maddpg_rbs", "greedy", "dt", "fh"]


def main() -> None:
    parser = argparse.ArgumentParser(description="IA-MADDPG training script")
    parser.add_argument("--method", default="ia_maddpg_uav",
                        choices=ALL_METHODS + ["all"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", default="results/")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = Config(seed=args.seed)
    if args.episodes is not None:
        object.__setattr__(cfg, "episodes", args.episodes)
    os.makedirs(args.save_dir, exist_ok=True)

    methods = ALL_METHODS if args.method == "all" else [args.method]
    all_rewards: dict = {}

    for method in methods:
        print(f"\n{'='*60}\n Training: {method}  (episodes={cfg.episodes})\n{'='*60}")
        rewards, tsrs, modes = train_one_method(method, cfg, args.save_dir)
        all_rewards[method] = rewards

    summary_path = os.path.join(args.save_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({m: v[-1] for m, v in all_rewards.items()}, f, indent=2)
    print(f"\nFinal rewards summary saved → {summary_path}")

    if len(all_rewards) > 1:
        import matplotlib
        matplotlib.use("Agg")
        from utils.visualization import plot_training_curves
        plot_path = os.path.join(args.save_dir, "training_curves.png")
        plot_training_curves(all_rewards, save_path=plot_path)
        print(f"Training curves saved → {plot_path}")


if __name__ == "__main__":
    main()
