"""
baselines.py — 5 baseline comparison methods for IA-MADDPG evaluation (torch-free).

Methods:
  1. DirectTransmission   — Mode 0 only, alpha=0.2 fixed.
  2. GreedyStrategy       — Instantaneous-SINR maximization, no UAV relay.
  3. FrequencyHopping     — Random mode/alpha avoidance.
  4. StandardMADDPG       — MADDPG without imitation learning (lambda_IL=0).
  5. IAMADDPG_RBSOnly     — IA-MADDPG with UAV relay disabled.
"""

import copy
import numpy as np

from algorithms.ia_maddpg import IAMADDPG


# ── SINR helpers (mode 0 / 1 only, for GreedyStrategy) ───────────────────────

def _sinr_mode0(i: int, alpha: float, ch: dict, cfg) -> float:
    num = cfg.G * cfg.P_J * ch["g_JS"][i] * ch["g_SD"][i] * alpha ** 2
    den = cfg.P_J * ch["g_JD"][i] + cfg.N0
    return num / max(den, 1e-30)


def _sinr_mode1(i: int, alpha: float, ch: dict, cfg) -> float:
    den_r = cfg.P_J * ch["g_JR"] + cfg.N0
    den_d = cfg.P_J * ch["g_JD"][i] + cfg.N0
    hop1 = cfg.G * cfg.P_J * ch["g_JS"][i] * ch["g_SR"][i] * alpha ** 2 / max(den_r, 1e-30)
    hop2 = cfg.G * cfg.P_J * ch["g_JR"] * ch["g_RD"][i] * alpha ** 2 / max(den_d, 1e-30)
    return min(hop1, hop2)


# ── 1. Direct Transmission ────────────────────────────────────────────────────

class DirectTransmission:
    """Mode 0 only, alpha=0.2 fixed. No relay, no learning."""

    ALPHA = 0.2

    def select_actions(self, env_state: dict):
        """
        env_state: dict with at least 'N' and 'K'.
        Returns su_actions (N, 4), uav_actions (K, 3).
        """
        N = env_state.get("N", 5)
        K = env_state.get("K", 2)
        su_actions = np.zeros((N, 4), dtype=np.float32)
        su_actions[:, 0] = self.ALPHA
        su_actions[:, 1] = 10.0    # strongly favour mode 0
        su_actions[:, 2] = -10.0
        su_actions[:, 3] = -10.0
        uav_actions = np.zeros((K, 3), dtype=np.float32)
        return su_actions, uav_actions


# ── 2. Greedy Strategy ────────────────────────────────────────────────────────

class GreedyStrategy:
    """
    Selects mode from {0=D2D, 1=RBS} to maximize instantaneous SINR.
    UAV relay (mode 2) is never used. UAV stays in place (zero displacement).
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def select_actions(self, channels: dict, cfg=None):
        """
        channels: dict from channel_model.compute_all_channels.
        Returns su_actions (N, 4), uav_actions (K, 3).
        """
        if cfg is None:
            cfg = self.cfg
        N, K = cfg.N, cfg.K
        su_actions = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            best_sinr, best_mode, best_alpha = -np.inf, 0, 0.2
            for alpha in np.linspace(0.1, 1.0, 10):
                sinr0 = _sinr_mode0(i, alpha, channels, cfg)
                sinr1 = _sinr_mode1(i, alpha, channels, cfg)
                for mode, sinr in enumerate([sinr0, sinr1]):
                    if sinr > best_sinr:
                        best_sinr, best_mode, best_alpha = sinr, mode, alpha
            logits = np.full(3, -10.0, dtype=np.float32)
            logits[best_mode] = 10.0
            su_actions[i] = [best_alpha, *logits]
        uav_actions = np.zeros((K, 3), dtype=np.float32)
        return su_actions, uav_actions


# ── 3. Frequency Hopping ──────────────────────────────────────────────────────

class FrequencyHopping:
    """
    Stochastic channel avoidance: random alpha ∈ [0.1, 0.5] and
    random mode from {0, 1} (no UAV relay).
    """

    def __init__(self, cfg, n_channels: int = 8) -> None:
        self.cfg = cfg
        self.n_channels = n_channels
        self.rng = np.random.default_rng(cfg.seed)

    def select_actions(self, step: int):
        """Returns su_actions (N, 4), uav_actions (K, 3)."""
        N, K = self.cfg.N, self.cfg.K
        su_actions = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            alpha = float(self.rng.uniform(0.1, 0.5))
            mode = int(self.rng.integers(0, 2))     # modes 0 or 1 only
            logits = np.full(3, -10.0, dtype=np.float32)
            logits[mode] = 10.0
            su_actions[i] = [alpha, *logits]
        uav_actions = np.zeros((K, 3), dtype=np.float32)
        return su_actions, uav_actions


# ── 4. Standard MADDPG (no imitation learning) ────────────────────────────────

class StandardMADDPG(IAMADDPG):
    """MADDPG without imitation learning: lambda_IL is always 0."""

    def __init__(self, cfg, env) -> None:
        cfg_copy = _clone_cfg(cfg, lambda_il_init=0.0, lambda_il_decay=1.0)
        super().__init__(cfg_copy, env)
        self.lambda_il = 0.0

    def warmup(self, env, n_steps: int) -> None:
        """No expert warmup for standard MADDPG."""
        pass

    def update(self, step: int) -> dict:
        result = super().update(step)
        self.lambda_il = 0.0    # keep IL at zero
        return result


# ── 5. IA-MADDPG (RBS only, UAV relay disabled) ───────────────────────────────

class IAMADDPG_RBSOnly(IAMADDPG):
    """
    IA-MADDPG with imitation learning enabled, but UAV relay mode 2 is
    disabled. UAV stays in place (zero displacement every step).
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env, allow_uav_mode=False)

    def select_actions(self, su_obs: np.ndarray, uav_obs: np.ndarray,
                       explore: bool = True):
        """Force mode 2 logit to -10 and freeze UAV."""
        su_acts, _ = super().select_actions(su_obs, uav_obs, explore)
        su_acts[:, 3] = -10.0
        uav_acts = np.zeros((self.cfg.K, 3), dtype=np.float32)
        return su_acts, uav_acts


# ── Helper ─────────────────────────────────────────────────────────────────────

def _clone_cfg(cfg, **overrides):
    """Return a shallow copy of cfg dataclass with overrides applied."""
    c = copy.copy(cfg)
    for k, v in overrides.items():
        object.__setattr__(c, k, v)
    return c
