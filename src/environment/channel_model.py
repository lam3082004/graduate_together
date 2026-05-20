"""
channel_model.py — G2G (Rayleigh) and A2G (probabilistic LoS) channel models
for the UAV-assisted Ambient Backscatter Anti-Jamming network.
"""

import numpy as np
from typing import Dict

# Speed of light (m/s)
_C = 3e8


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _dist2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """2-D Euclidean distance; both arrays are (2,) or broadcastable."""
    diff = np.asarray(p1[:2], dtype=float) - np.asarray(p2[:2], dtype=float)
    return float(np.sqrt(np.dot(diff, diff)))


def _dist3d(p_uav: np.ndarray, p_ground_2d: np.ndarray) -> float:
    """3-D distance from UAV (x,y,z) to ground node (x,y)."""
    dx = float(p_uav[0]) - float(p_ground_2d[0])
    dy = float(p_uav[1]) - float(p_ground_2d[1])
    dz = float(p_uav[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


# ─── G2G channel (Rayleigh fading) ────────────────────────────────────────────

def channel_gain_g2g(pos1: np.ndarray, pos2: np.ndarray,
                     eta: float, rng: np.random.Generator) -> float:
    """Sample G2G |h|^2 ~ Exp(d^{-eta}). pos1/pos2: (2,) ground coords."""
    d = max(_dist2d(pos1, pos2), 1e-3)          # avoid d=0
    mean_gain = d ** (-eta)
    # Rayleigh envelope → |h|^2 ~ Exponential(1/mean_gain)
    return float(rng.exponential(mean_gain))


# ─── A2G channel (probabilistic LoS) ──────────────────────────────────────────

def prob_los(height: float, horiz_dist: float,
             a: float, b: float) -> float:
    """ITU-R LoS probability: 1/(1+a·exp(-b·(theta-a))), theta=elevation°."""
    horiz_dist = max(horiz_dist, 1e-3)
    theta_deg = np.degrees(np.arctan2(height, horiz_dist))
    exponent = -b * (theta_deg - a)
    # Clip exponent to prevent overflow
    exponent = np.clip(exponent, -500, 500)
    return float(1.0 / (1.0 + a * np.exp(exponent)))


def mean_channel_gain_a2g(pos_uav: np.ndarray, pos_ground: np.ndarray,
                          config) -> float:
    """Mean A2G gain: ḡ = P_LoS/L_LoS + (1-P_LoS)/L_NLoS.
    pos_uav (3,), pos_ground (2,)."""
    d3d = max(_dist3d(pos_uav, pos_ground), 1e-3)
    horiz_dist = max(_dist2d(pos_uav[:2], pos_ground[:2]), 1e-3)
    height = float(pos_uav[2])

    # Free-space path loss factor
    fspl_factor = (4 * np.pi * config.fc * d3d / _C) ** 2
    fspl_factor = max(fspl_factor, 1e-30)

    l_los = config.xi_los * fspl_factor
    l_nlos = config.xi_nlos * fspl_factor

    p_los = prob_los(height, horiz_dist, config.a_los, config.b_los)
    p_nlos = 1.0 - p_los

    return float(p_los / l_los + p_nlos / l_nlos)


# ─── Full channel computation ──────────────────────────────────────────────────

def compute_all_channels(
    positions_su: np.ndarray,
    positions_du: np.ndarray,
    pos_rbs: np.ndarray,
    positions_uav: np.ndarray,
    pos_jammer: np.ndarray,
    config,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Return dict of all channel gains for SINR calculation.
    Keys: g_SD(N), g_JS(N), g_JD(N), g_SR(N), g_JR, g_RD(N),
          g_bar_SU(N,K), g_bar_UD(N,K), g_bar_JU(K)."""
    N, K = config.N, config.K
    eta = config.eta

    # G2G: SU → DU
    g_sd = np.array([channel_gain_g2g(positions_su[i], positions_du[i], eta, rng)
                     for i in range(N)])

    # G2G: Jammer → SU
    g_js = np.array([channel_gain_g2g(pos_jammer, positions_su[i], eta, rng)
                     for i in range(N)])

    # G2G: Jammer → DU
    g_jd = np.array([channel_gain_g2g(pos_jammer, positions_du[i], eta, rng)
                     for i in range(N)])

    # G2G: SU → RBS
    g_sr = np.array([channel_gain_g2g(positions_su[i], pos_rbs, eta, rng)
                     for i in range(N)])

    # G2G: Jammer → RBS
    g_jr = channel_gain_g2g(pos_jammer, pos_rbs, eta, rng)

    # G2G: RBS → DU
    g_rd = np.array([channel_gain_g2g(pos_rbs, positions_du[i], eta, rng)
                     for i in range(N)])

    # A2G (mean): SU_i → UAV_k
    g_bar_su = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            g_bar_su[i, k] = mean_channel_gain_a2g(
                positions_uav[k], positions_su[i], config)

    # A2G (mean): UAV_k → DU_i  (downlink, same mean by reciprocity)
    g_bar_ud = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            g_bar_ud[i, k] = mean_channel_gain_a2g(
                positions_uav[k], positions_du[i], config)

    # A2G (mean): Jammer → UAV_k
    g_bar_ju = np.array([mean_channel_gain_a2g(positions_uav[k], pos_jammer, config)
                         for k in range(K)])

    return {
        "g_SD": g_sd,
        "g_JS": g_js,
        "g_JD": g_jd,
        "g_SR": g_sr,
        "g_JR": g_jr,
        "g_RD": g_rd,
        "g_bar_SU": g_bar_su,
        "g_bar_UD": g_bar_ud,
        "g_bar_JU": g_bar_ju,
    }
