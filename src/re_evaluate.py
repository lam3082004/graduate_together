"""
re_evaluate.py — Re-evaluate trained checkpoints with a SINR threshold range
that is meaningful for the actual channel/jammer model.

The default training threshold (5 dB) is far above what any method can attain
in this regime (typical SINR ranges -40 dB to +5 dB). Re-evaluating with a
broader sweep produces interpretable TSR curves for Chapter 5.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from environment.network_env import AntiJammingEnv
from algorithms.ia_maddpg import IAMADDPG
from algorithms.baselines import (
    DirectTransmission, GreedyStrategy, FrequencyHopping,
    StandardMADDPG, IAMADDPG_RBSOnly,
)

METHOD_LABELS = {
    "ia_maddpg_uav": "IA-MADDPG+UAV",
    "ia_maddpg_rbs": "IA-MADDPG(RBS)",
    "maddpg":        "StandardMADDPG",
    "greedy":        "Greedy",
    "fh":            "FreqHopping",
    "dt":            "DirectTX",
}
LEARNABLE = {"ia_maddpg_uav", "ia_maddpg_rbs", "maddpg"}
COLORS = {"IA-MADDPG+UAV": "#2196F3", "IA-MADDPG(RBS)": "#4CAF50",
          "StandardMADDPG": "#FF9800", "Greedy": "#9C27B0",
          "FreqHopping": "#F44336", "DirectTX": "#607D8B"}


def build_agent(method, cfg, env):
    if method == "ia_maddpg_uav": return IAMADDPG(cfg, env)
    if method == "ia_maddpg_rbs": return IAMADDPG_RBSOnly(cfg, env)
    if method == "maddpg":        return StandardMADDPG(cfg, env)
    if method == "greedy":        return GreedyStrategy(cfg)
    if method == "fh":            return FrequencyHopping(cfg)
    if method == "dt":            return DirectTransmission()
    raise ValueError(method)


def pick_action(method, agent, su_obs, uav_obs, env, step, cfg):
    if method in LEARNABLE:
        return agent.select_actions(su_obs, uav_obs, explore=False)
    if method == "greedy": return agent.select_actions(env.channels, cfg)
    if method == "fh":     return agent.select_actions(step)
    return agent.select_actions({"N": cfg.N, "K": cfg.K})


def evaluate(method, cfg, ckpt_dir, n_eps=40, capture_traj=False):
    np.random.seed(cfg.seed + 7)
    env = AntiJammingEnv(cfg)
    agent = build_agent(method, cfg, env)
    if method in LEARNABLE:
        try:
            agent.load(ckpt_dir)
        except Exception as e:
            print(f"  [{method}] load failed: {e}")

    sinr_pool, throughput_ep, reward_ep, energy_ep = [], [], [], []
    mode_counts = [0, 0, 0]
    trajectory = None
    for ep in range(n_eps):
        su_obs, uav_obs = env.reset()
        ep_th = ep_r = 0.0
        e0 = env._uav_energy.copy()
        traj = []
        for s in range(cfg.steps_per_episode):
            su_a, uav_a = pick_action(method, agent, su_obs, uav_obs, env, s, cfg)
            su_obs, uav_obs, rew, done, info = env.step(su_a, uav_a)
            sinr = info["sinr"]
            sinr_pool.append(sinr.copy())
            ep_th += float(np.mean(np.log2(1.0 + np.maximum(sinr, 0.0))))
            ep_r += float(np.mean(rew["su"]))
            for m in info["modes"]:
                mode_counts[int(m)] += 1
            if capture_traj and ep == 0:
                traj.append(env.pos_uav.copy())
            if done: break
        T = cfg.steps_per_episode
        throughput_ep.append(ep_th / T)
        reward_ep.append(ep_r / T)
        energy_ep.append(float(np.sum(e0 - env._uav_energy)))
        if capture_traj and ep == 0:
            trajectory = np.array(traj)

    sinr_arr = np.concatenate(sinr_pool)
    return dict(
        sinr_db = 10 * np.log10(np.maximum(sinr_arr, 1e-12)),
        sinr_lin = sinr_arr,
        avg_throughput = float(np.mean(throughput_ep)),
        std_throughput = float(np.std(throughput_ep)),
        avg_reward = float(np.mean(reward_ep)),
        std_reward = float(np.std(reward_ep)),
        avg_energy = float(np.mean(energy_ep)),
        energy_efficiency = float(np.mean(throughput_ep))
                          / max(float(np.mean(energy_ep)), 1e-6),
        mode_counts = mode_counts,
        trajectory = trajectory,
        env_snapshot = dict(
            pos_su=env.pos_su.copy(),
            pos_du=env.pos_du.copy(),
            pos_rbs=env._pos_rbs.copy(),
            pos_jammer=env.pos_jammer.copy(),
        ),
    )


def plot_sinr_cdf(sinr_dict: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, db in sinr_dict.items():
        x = np.sort(db)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, label=name, color=COLORS.get(name, "gray"), linewidth=2)
    ax.set_xlabel("SINR (dB)", fontsize=11)
    ax.set_ylabel("Empirical CDF", fontsize=11)
    ax.set_title("Phân bố SINR (CDF) — đầu ra eval policy", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-60, 10)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tsr_curve(sinr_dict: dict, thresholds_db, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, db in sinr_dict.items():
        tsr = [float(np.mean(db >= th)) for th in thresholds_db]
        ax.plot(thresholds_db, tsr, marker="o", markersize=4,
                label=name, color=COLORS.get(name, "gray"), linewidth=2)
    ax.set_xlabel("Ngưỡng SINR (dB)", fontsize=11)
    ax.set_ylabel("Transmission Success Rate (TSR)", fontsize=11)
    ax.set_title("TSR theo ngưỡng SINR (toàn dải)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_bar(metrics: dict, save_path: str) -> None:
    methods = list(metrics.keys())
    vals = [metrics[m]["avg_throughput"] for m in methods]
    errs = [metrics[m]["std_throughput"] for m in methods]
    cols = [COLORS.get(m, "gray") for m in methods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(methods, vals, yerr=errs, capsize=4, color=cols,
                  alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylabel("Throughput trung bình (bits/s/Hz)", fontsize=11)
    ax.set_title("So sánh thông lượng (eval deterministic)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mode_distribution(mode_dict, save_path):
    methods = list(mode_dict.keys())
    n = len(methods)
    x = np.arange(3)
    w = 0.8 / max(n, 1)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i, m in enumerate(methods):
        counts = np.asarray(mode_dict[m], float)
        fracs = counts / max(counts.sum(), 1.0)
        ax.bar(x + i * w - 0.4 + w / 2, fracs, width=w * 0.9,
               color=COLORS.get(m, "gray"), label=m, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["D2D (mode 0)", "RBS (mode 1)", "UAV (mode 2)"])
    ax.set_ylabel("Tỷ lệ chọn", fontsize=11)
    ax.set_title("Phân bố chế độ truyền (eval)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results_compare/")
    parser.add_argument("--n_eps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    object.__setattr__(cfg, "steps_per_episode", args.steps)

    methods = list(METHOD_LABELS.keys())
    metrics = {}
    sinr_db = {}
    print(f"\nRe-evaluating in {args.results_dir} with {args.n_eps} eps × "
          f"{args.steps} steps …")
    for m in methods:
        print(f"  → {METHOD_LABELS[m]}")
        ckpt = os.path.join(args.results_dir, m)
        v = evaluate(m, cfg, ckpt, n_eps=args.n_eps,
                     capture_traj=(m == "ia_maddpg_uav"))
        metrics[METHOD_LABELS[m]] = v
        sinr_db[METHOD_LABELS[m]] = v["sinr_db"]

    # Threshold sweep (broader: -30 → +5 dB)
    thresholds = np.arange(-30, 6, 2.5)

    # Tabular summary at three thresholds
    print(f"\n{'Method':<18}  {'Throughput':>10}  {'Reward':>9}  "
          f"{'TSR@-15dB':>10}  {'TSR@-10dB':>10}  {'TSR@-5dB':>9}")
    print("─" * 78)
    summary_rows = []
    for name, v in metrics.items():
        s = v["sinr_db"]
        t15 = float(np.mean(s >= -15))
        t10 = float(np.mean(s >= -10))
        t5  = float(np.mean(s >= -5))
        print(f"{name:<18}  {v['avg_throughput']:>10.4f}  "
              f"{v['avg_reward']:>+9.4f}  "
              f"{t15:>10.4f}  {t10:>10.4f}  {t5:>9.4f}")
        mc = np.asarray(v["mode_counts"], float)
        mc_t = max(mc.sum(), 1)
        summary_rows.append({
            "method": name,
            "throughput": v["avg_throughput"],
            "throughput_std": v["std_throughput"],
            "reward": v["avg_reward"],
            "reward_std": v["std_reward"],
            "tsr@-15dB": t15,
            "tsr@-10dB": t10,
            "tsr@-5dB":  t5,
            "energy": v["avg_energy"],
            "energy_efficiency": v["energy_efficiency"],
            "mode_d2d": mc[0] / mc_t,
            "mode_rbs": mc[1] / mc_t,
            "mode_uav": mc[2] / mc_t,
        })
    print("─" * 78)

    # ── Exports ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.results_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            r = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
            w.writerow(r)

    json_path = os.path.join(args.results_dir, "metrics_summary.json")
    serialisable = []
    for r in summary_rows:
        serialisable.append({k: float(v) if isinstance(v, np.floating)
                             else v for k, v in r.items()})
    tsr_curves = {name: [float(np.mean(s >= th)) for th in thresholds]
                  for name, s in sinr_db.items()}
    with open(json_path, "w") as f:
        json.dump({
            "summary": serialisable,
            "thresholds_db": [float(t) for t in thresholds],
            "tsr_vs_threshold": tsr_curves,
        }, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_sinr_cdf(sinr_db, os.path.join(args.results_dir, "sinr_cdf.png"))
    plot_tsr_curve(sinr_db, thresholds,
                   os.path.join(args.results_dir, "tsr_vs_threshold.png"))
    plot_throughput_bar(metrics,
                        os.path.join(args.results_dir, "throughput_comparison.png"))
    mode_dict = {n: v["mode_counts"] for n, v in metrics.items()}
    plot_mode_distribution(mode_dict,
                           os.path.join(args.results_dir, "mode_distribution.png"))

    # Save the refreshed UAV trajectory too
    if "IA-MADDPG+UAV" in metrics and metrics["IA-MADDPG+UAV"]["trajectory"] is not None:
        from utils.visualization import plot_uav_trajectory
        v = metrics["IA-MADDPG+UAV"]
        snap = v["env_snapshot"]
        plot_uav_trajectory(
            trajectory=v["trajectory"],
            positions_su=snap["pos_su"], positions_du=snap["pos_du"],
            pos_rbs=snap["pos_rbs"], pos_jammer=snap["pos_jammer"],
            save_path=os.path.join(args.results_dir, "uav_trajectory.png"))
        plt.close("all")

    print(f"\n[done] csv  → {csv_path}")
    print(f"[done] json → {json_path}")
    print(f"[done] plots → {args.results_dir}")


if __name__ == "__main__":
    main()
