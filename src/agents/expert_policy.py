"""
expert_policy.py — Analytical expert (imitation) policies for IA-MADDPG warm-up.

SUExpertPolicy : greedy search over (alpha, mode) → max instantaneous SINR.
UAVExpertPolicy: heuristic 9-candidate move → max sum-SINR for served SUs.

Both policies are stateless and numpy-only (no torch dependency).
"""

import numpy as np


# ─── SINR helpers ─────────────────────────────────────────────────────────────

def _sinr_mode0(alpha, g_js, g_sd, g_jd, cfg) -> float:
    num = cfg.G * cfg.P_J * g_js * g_sd * alpha ** 2
    return num / (cfg.P_J * g_jd + cfg.N0 + 1e-12)


def _sinr_mode1(alpha, g_js, g_sr, g_jr, g_rd, g_jd, cfg) -> float:
    hop1 = cfg.G * cfg.P_J * g_js * g_sr * alpha ** 2 / (cfg.P_J * g_jr + cfg.N0 + 1e-12)
    hop2 = cfg.G * cfg.P_J * g_jr * g_rd * alpha ** 2 / (cfg.P_J * g_jd + cfg.N0 + 1e-12)
    return min(hop1, hop2)


def _sinr_mode2(alpha, g_js, g_su_uav, g_j_uav, g_uav_d, g_jd, cfg) -> float:
    hop1 = cfg.G * cfg.P_J * g_js * g_su_uav * alpha ** 2 / (cfg.P_J * g_j_uav + cfg.N0 + 1e-12)
    hop2 = cfg.G * cfg.P_J * g_uav_d * alpha ** 2 / (cfg.P_J * g_jd + cfg.N0 + 1e-12)
    return min(hop1, hop2)


def _plos(H: float, d_horiz: float, a: float, b: float) -> float:
    """Probability of LoS for UAV-ground link."""
    if d_horiz < 1e-3:
        return 1.0
    theta = (180.0 / np.pi) * np.arctan(H / d_horiz)
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def _uav_channel(pos_uav: np.ndarray, pos_node: np.ndarray, cfg) -> float:
    """Average UAV→ground channel gain (LoS/NLoS mixture)."""
    d3 = np.linalg.norm(pos_uav - pos_node) + 1e-3
    H  = abs(pos_uav[2])
    d_h = np.sqrt(max((pos_uav[0] - pos_node[0]) ** 2 + (pos_uav[1] - pos_node[1]) ** 2, 1e-6))
    p   = _plos(H, d_h, cfg.a_los, cfg.b_los)
    fc  = cfg.fc
    lam = 3e8 / fc
    L0  = (4 * np.pi * d3 / lam) ** 2
    return p / (cfg.xi_los * L0) + (1 - p) / (cfg.xi_nlos * L0)


# ─── SU Expert ────────────────────────────────────────────────────────────────

_ALPHA_CANDS = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])


class SUExpertPolicy:
    """
    Greedy expert: exhaustive search over alpha ∈ {0.1,…,1.0} × mode ∈ {0,1,2}
    to maximise instantaneous SINR for SU i.

    channels dict keys expected (all floats):
        g_js, g_sd, g_jd           — required for mode 0
        g_sr, g_jr, g_rd           — required for mode 1
    channel_gains_uav              — (K,) UAV→SU channel gains
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def select_action(
        self,
        i: int,
        channels: dict,
        positions_uav: np.ndarray,          # (K, 3)
        uav_assignment: np.ndarray,         # (K,) index of assigned SU (-1 = unassigned)
    ) -> tuple[float, int]:
        """
        Returns:
            alpha (float ∈ [0,1]), mode (int ∈ {0,1,2})
        """
        cfg    = self.cfg
        best_sinr, best_alpha, best_mode = -np.inf, 0.5, 0

        # Find which UAV (if any) serves SU i
        uav_idx = -1
        for k, assigned in enumerate(uav_assignment):
            if int(assigned) == i:
                uav_idx = k
                break

        for alpha in _ALPHA_CANDS:
            # Mode 0
            s0 = _sinr_mode0(
                alpha,
                channels.get("g_js", 1e-5),
                channels.get("g_sd", 1e-5),
                channels.get("g_jd", 1e-5),
                cfg,
            )
            if s0 > best_sinr:
                best_sinr, best_alpha, best_mode = s0, alpha, 0

            # Mode 1
            s1 = _sinr_mode1(
                alpha,
                channels.get("g_js", 1e-5),
                channels.get("g_sr", 1e-5),
                channels.get("g_jr", 1e-5),
                channels.get("g_rd", 1e-5),
                channels.get("g_jd", 1e-5),
                cfg,
            )
            if s1 > best_sinr:
                best_sinr, best_alpha, best_mode = s1, alpha, 1

            # Mode 2 (only if a UAV serves this SU)
            if uav_idx >= 0:
                g_su_uav = channels.get(f"g_su_uav_{uav_idx}", 1e-5)
                g_j_uav  = channels.get(f"g_j_uav_{uav_idx}", 1e-5)
                s2 = _sinr_mode2(
                    alpha,
                    channels.get("g_js", 1e-5),
                    g_su_uav,
                    g_j_uav,
                    channels.get("g_uav_d", 1e-5),
                    channels.get("g_jd", 1e-5),
                    cfg,
                )
                if s2 > best_sinr:
                    best_sinr, best_alpha, best_mode = s2, alpha, 2

        return float(best_alpha), int(best_mode)


# ─── UAV Expert ───────────────────────────────────────────────────────────────

# 8 cardinal directions in the xy-plane + hover
_DIRECTION_VECS = np.array([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [0, 0, 0],  # hover
], dtype=float)


class UAVExpertPolicy:
    """
    Heuristic: evaluate 9 candidate moves (8 directions + stay), pick the
    one that maximises sum-SINR across SUs in the cluster.

    alpha_prev: last-known alpha values for each SU (length N array).
    su_cluster: indices of SUs served by UAV k.
    su_positions: (N, 3) ground node positions.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def select_action(
        self,
        k: int,
        positions_uav: np.ndarray,   # (K, 3)
        su_cluster: list[int],        # SU indices served by UAV k
        channels: dict,               # g_js, g_jd per SU index i
        alpha_prev: np.ndarray,       # (N,) last alpha values
        su_positions: np.ndarray,     # (N, 3)
    ) -> np.ndarray:
        """
        Returns:
            delta_p (3,) normalised to [-1, 1]  (multiply by v_max*dt outside)
        """
        cfg     = self.cfg
        p_uav   = positions_uav[k].copy()
        step    = cfg.v_max * cfg.dt
        best_val, best_dp = -np.inf, np.zeros(3)

        for raw_dir in _DIRECTION_VECS:
            norm = np.linalg.norm(raw_dir)
            d    = raw_dir / norm if norm > 0 else raw_dir
            candidate = p_uav + d * step
            # Clip altitude
            candidate[2] = np.clip(candidate[2], cfg.H_min, cfg.H_max)

            total_sinr = 0.0
            for i in su_cluster:
                g_su_uav = _uav_channel(candidate, su_positions[i], cfg)
                g_jd     = channels.get(f"g_jd_{i}", 1e-5)
                g_js     = channels.get(f"g_js_{i}", 1e-5)
                g_j_uav  = _uav_channel(candidate, su_positions[i], cfg)  # approx
                g_uav_d  = _uav_channel(candidate, su_positions[i], cfg)
                alpha    = float(alpha_prev[i]) if i < len(alpha_prev) else 0.5
                total_sinr += _sinr_mode2(alpha, g_js, g_su_uav, g_j_uav, g_uav_d, g_jd, cfg)

            if total_sinr > best_val:
                best_val = total_sinr
                best_dp  = d  # already in [-1, 1] range (unit vector)

        return best_dp.astype(np.float32)
