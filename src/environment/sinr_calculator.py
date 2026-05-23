"""
sinr_calculator.py — SINR computation for three transmission modes
and reward calculation for the anti-jamming network.

Mode 0: Direct SU→DU link (no relay, no UAV)
Mode 1: SU→RBS→DU two-hop relay
Mode 2: SU→UAV (ambient backscatter) → DU via UAV
"""

import numpy as np
from typing import Dict


# ─── Mode SINR formulas ────────────────────────────────────────────────────────

def sinr_mode0(g_JS: float, g_SD: float, g_JD: float,
               alpha: float, cfg) -> float:
    """
    Direct SU→DU SINR (Mode 0).

    SU uses ambient backscatter of jammer signal with reflection coefficient α.
    Signal power at DU  : G · P_J · g_JS · g_SD · α²
    Interference at DU  : P_J · g_JD
    Noise               : N0

    Returns
    -------
    float — SINR (linear), clipped to [0, 1e9]
    """
    signal = cfg.G * cfg.P_J * g_JS * g_SD * (alpha ** 2)
    interference = cfg.P_J * g_JD + cfg.N0
    return float(np.clip(signal / max(interference, 1e-30), 0.0, 1e9))


def sinr_mode1(g_JS: float, g_SR: float, g_JR: float,
               g_RD: float, g_JD: float,
               alpha: float, cfg) -> float:
    """
    Two-hop relay SU→RBS→DU SINR (Mode 1).

    Hop 1 (SU → RBS):
        signal    : G · P_J · g_JS · g_SR · α²
        interf+n  : P_J · g_JR + N0

    Hop 2 (RBS → DU):  RBS re-transmits with same reflected power
        signal    : G · P_J · g_JS · g_SR · α² (amplify & forward analogy)
        interf+n  : P_J · g_JD + N0

    End-to-end SINR = min(SINR_hop1, SINR_hop2)  (bottleneck)

    Returns
    -------
    float — SINR (linear)
    """
    signal1 = cfg.G * cfg.P_J * g_JS * g_SR * (alpha ** 2)
    denom1 = cfg.P_J * g_JR + cfg.N0
    sinr1 = signal1 / max(denom1, 1e-30)

    signal2 = cfg.G * cfg.P_J * g_JS * g_RD * (alpha ** 2)
    denom2 = cfg.P_J * g_JD + cfg.N0
    sinr2 = signal2 / max(denom2, 1e-30)

    return float(np.clip(min(sinr1, sinr2), 0.0, 1e9))


def sinr_mode2(g_JS: float, g_bar_SU: float, g_bar_JU: float,
               g_bar_UD: float, g_JD: float,
               alpha: float, cfg) -> float:
    """
    UAV-assisted ambient backscatter SINR (Mode 2).

    Hop 1 (SU → UAV via backscatter):
        signal    : G · P_J · g_JS · ḡ_SU · α²
        interf+n  : P_J · ḡ_JU + N0

    Hop 2 (UAV → DU):
        signal    : G · P_J · ḡ_JU · ḡ_UD · α²
        interf+n  : P_J · g_JD + N0

    End-to-end SINR = min(SINR_hop1, SINR_hop2)

    Returns
    -------
    float — SINR (linear)
    """
    signal1 = cfg.G * cfg.P_J * g_JS * g_bar_SU * (alpha ** 2)
    denom1 = cfg.P_J * g_bar_JU + cfg.N0
    sinr1 = signal1 / max(denom1, 1e-30)

    signal2 = cfg.G * cfg.P_J * g_bar_JU * g_bar_UD * (alpha ** 2)
    denom2 = cfg.P_J * g_JD + cfg.N0
    sinr2 = signal2 / max(denom2, 1e-30)

    return float(np.clip(min(sinr1, sinr2), 0.0, 1e9))


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def compute_sinr(i: int, mode: int, alpha: float,
                 channels: Dict[str, np.ndarray],
                 uav_idx: np.ndarray, cfg) -> float:
    """
    Compute SINR for SU-DU pair i given a transmission mode.

    Parameters
    ----------
    i         : SU index (0 … N-1)
    mode      : 0 (direct), 1 (relay), 2 (UAV backscatter)
    alpha     : backscatter reflection coefficient ∈ [0, 1]
    channels  : dict returned by compute_all_channels()
    uav_idx   : (N,) array — UAV index assigned to each SU
    cfg       : Config dataclass

    Returns
    -------
    float — SINR (linear)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    k = int(uav_idx[i])

    if mode == 0:
        return sinr_mode0(
            g_JS=channels["g_JS"][i],
            g_SD=channels["g_SD"][i],
            g_JD=channels["g_JD"][i],
            alpha=alpha,
            cfg=cfg,
        )
    elif mode == 1:
        return sinr_mode1(
            g_JS=channels["g_JS"][i],
            g_SR=channels["g_SR"][i],
            g_JR=float(channels["g_JR"]),
            g_RD=channels["g_RD"][i],
            g_JD=channels["g_JD"][i],
            alpha=alpha,
            cfg=cfg,
        )
    elif mode == 2:
        return sinr_mode2(
            g_JS=channels["g_JS"][i],
            g_bar_SU=channels["g_bar_SU"][i, k],
            g_bar_JU=channels["g_bar_JU"][k],
            g_bar_UD=channels["g_bar_UD"][i, k],
            g_JD=channels["g_JD"][i],
            alpha=alpha,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown mode {mode}. Must be 0, 1, or 2.")


# ─── Reward ────────────────────────────────────────────────────────────────────

def compute_reward(gamma: float, alpha: float, mode: int,
                   delta_p_uav_norm: float, cfg) -> float:
    """
    Reward signal for a single SU-DU pair.

    r = w1 · log2(1 + γ)
      + w2 · tanh(γ/γ_th − 1)
      − w3 · α²
      − w4 · Δp_UAV_norm    (only if mode == 2)

    Parameters
    ----------
    gamma            : achieved SINR (linear)
    alpha            : reflection coefficient ∈ [0, 1]
    mode             : transmission mode
    delta_p_uav_norm : normalised UAV position change magnitude ∈ [0, 1]
    cfg              : Config dataclass

    Returns
    -------
    float — scalar reward
    """
    gamma = max(gamma, 0.0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    delta_p_uav_norm = float(np.clip(delta_p_uav_norm, 0.0, 1.0))

    r = (cfg.w1 * np.log2(1.0 + gamma)
         + cfg.w2 * np.tanh(gamma / max(cfg.gamma_th, 1e-30) - 1.0)
         - cfg.w3 * alpha ** 2)

    if mode == 2:
        r -= cfg.w4 * delta_p_uav_norm

    return float(r)
