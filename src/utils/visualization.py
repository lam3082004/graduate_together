"""
visualization.py — Plotting utilities for IA-MADDPG training analysis
and result comparison across baseline methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (activates 3D projection)


# ── Shared style ───────────────────────────────────────────────────────────────

_COLORS = {
    "IA-MADDPG+UAV": "#2196F3",
    "IA-MADDPG(RBS)": "#4CAF50",
    "StandardMADDPG": "#FF9800",
    "Greedy":         "#9C27B0",
    "FreqHopping":    "#F44336",
    "DirectTX":       "#607D8B",
}
_DEFAULT_COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def _method_color(name: str, idx: int = 0) -> str:
    return _COLORS.get(name, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])


def _smooth(y: np.ndarray, window: int = 20) -> np.ndarray:
    """Uniform moving-average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def _savefig(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


# ── 1. Training reward curves ─────────────────────────────────────────────────

def plot_training_curves(reward_histories: dict, save_path=None,
                          window: int = 20) -> None:
    """
    Plot smoothed reward curves for all methods.

    Parameters
    ----------
    reward_histories : dict[str, list[float]] — per-episode rewards
    save_path        : optional file path (.png / .pdf)
    window           : smoothing window size
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (name, rewards) in enumerate(reward_histories.items()):
        r = np.asarray(rewards, dtype=float)
        raw = ax.plot(r, alpha=0.2, color=_method_color(name, idx))
        ax.plot(_smooth(r, window), label=name,
                color=_method_color(name, idx), linewidth=2)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Average Network Reward", fontsize=12)
    ax.set_title("Training Reward Curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    _savefig(fig, save_path)


# ── 2. UAV trajectory ─────────────────────────────────────────────────────────

def plot_uav_trajectory(trajectory: np.ndarray,
                         positions_su: np.ndarray,
                         positions_du: np.ndarray,
                         pos_rbs: np.ndarray,
                         pos_jammer: np.ndarray,
                         save_path=None) -> None:
    """
    3-D trajectory plot of UAV(s) over one episode.

    Parameters
    ----------
    trajectory    : (T, K, 3) — UAV positions over time
    positions_su  : (N, 2)    — SU ground positions
    positions_du  : (N, 2)    — DU ground positions
    pos_rbs       : (2,)      — RBS position
    pos_jammer    : (2,)      — Jammer position
    """
    T, K, _ = trajectory.shape
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("tab10")
    for k in range(K):
        traj_k = trajectory[:, k, :]
        ax.plot(traj_k[:, 0], traj_k[:, 1], traj_k[:, 2],
                color=cmap(k), linewidth=1.5, label=f"UAV {k+1}")
        ax.scatter(*traj_k[0], color=cmap(k), marker="o", s=60, zorder=5)
        ax.scatter(*traj_k[-1], color=cmap(k), marker="^", s=80, zorder=5)

    z0 = 0
    ax.scatter(positions_su[:, 0], positions_su[:, 1],
               z0, marker="s", c="blue", s=50, label="SU")
    ax.scatter(positions_du[:, 0], positions_du[:, 1],
               z0, marker="D", c="green", s=50, label="DU")
    ax.scatter(*pos_rbs, z0, marker="P", c="purple", s=80, label="RBS")
    ax.scatter(*pos_jammer, z0, marker="X", c="red", s=100, label="Jammer")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("UAV 3-D Trajectory", fontsize=13)
    ax.legend(fontsize=9)
    _savefig(fig, save_path)


# ── 3. Mode distribution ──────────────────────────────────────────────────────

def plot_mode_distribution(mode_counts: dict, save_path=None) -> None:
    """
    Grouped bar chart of D2D / RBS / UAV mode selection proportions.

    Parameters
    ----------
    mode_counts : dict[str, list[int|float]] — {method: [d2d, rbs, uav]} counts
    """
    methods = list(mode_counts.keys())
    n = len(methods)
    modes = ["Direct D2D", "RBS Relay", "UAV Relay"]
    x = np.arange(len(modes))
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, method in enumerate(methods):
        counts = np.asarray(mode_counts[method], dtype=float)
        total = counts.sum()
        fracs = counts / max(total, 1.0)
        ax.bar(x + idx * width - 0.4 + width / 2,
               fracs, width=width * 0.9,
               color=_method_color(method, idx), label=method, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Selection Proportion", fontsize=12)
    ax.set_title("Transmission Mode Distribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _savefig(fig, save_path)


# ── 4. Throughput comparison ──────────────────────────────────────────────────

def plot_throughput_comparison(results: dict, save_path=None) -> None:
    """
    Horizontal bar chart comparing average throughput across methods.

    Parameters
    ----------
    results : dict[str, float] — average throughput in bits/s/Hz
    """
    methods = list(results.keys())
    values = [results[m] for m in methods]
    colors = [_method_color(m, i) for i, m in enumerate(methods)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(methods, values, color=colors, alpha=0.85, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ax.set_xlabel("Average Throughput (bits/s/Hz)", fontsize=12)
    ax.set_title("Throughput Comparison", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.invert_yaxis()
    _savefig(fig, save_path)


# ── 5. Ablation study ─────────────────────────────────────────────────────────

def plot_ablation_study(ablation_results: dict,
                         metric: str = "reward",
                         save_path=None) -> None:
    """
    Bar chart for ablation study results.

    Parameters
    ----------
    ablation_results : dict[str, float] — {variant_name: metric_value}
    metric           : label for y-axis
    """
    variants = list(ablation_results.keys())
    values = [ablation_results[v] for v in variants]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
              for i in range(len(variants))]
    bars = ax.bar(variants, values, color=colors, alpha=0.85, width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f"Ablation Study — {metric.capitalize()}", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    _savefig(fig, save_path)


# ── 6. TSR vs SINR threshold ──────────────────────────────────────────────────

def plot_tsr_vs_threshold(tsr_results: dict,
                           thresholds: list,
                           save_path=None) -> None:
    """
    Line plot: Transmission Success Rate vs SINR threshold.

    Parameters
    ----------
    tsr_results : dict[str, list[float]] — {method: [tsr @ each threshold]}
    thresholds  : list[float] — SINR threshold values (dB)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (name, tsrs) in enumerate(tsr_results.items()):
        ax.plot(thresholds, tsrs, marker="o", markersize=5,
                color=_method_color(name, idx), label=name, linewidth=2)

    ax.set_xlabel("SINR Threshold (dB)", fontsize=12)
    ax.set_ylabel("Transmission Success Rate (TSR)", fontsize=12)
    ax.set_title("TSR vs. SINR Threshold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    _savefig(fig, save_path)
