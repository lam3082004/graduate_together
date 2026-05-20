"""
network_env.py — Gym-like multi-agent environment for the UAV-assisted
Ambient Backscatter Anti-Jamming network.
N SU agents: choose (alpha, mode logits). K UAV agents: choose (dx,dy,dz).
"""

import numpy as np
from typing import Dict, Tuple

from config import Config
from environment.channel_model import compute_all_channels
from environment.sinr_calculator import compute_sinr, compute_reward


class AntiJammingEnv:
    """Multi-agent anti-jamming environment.

    SU obs (4,) : [gamma_norm, E_norm, mode_norm, P_J_hat_norm]
    UAV obs (3+2N+2,): [px/A, py/A, pz/Hmax, g_SU×N, g_UD×N, P_J_hat, E/Emax]
    SU action (4,) : [alpha, logit_m0, logit_m1, logit_m2]
    UAV action (3,): [dx,dy,dz] ∈ [-1,1]³ scaled to v_max·dt
    """

    _SINR_NORM = 100.0  # normalisation constant for SINR in observations

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._step_count = 0
        self._pos_rbs = np.array([cfg.area_size / 2, cfg.area_size / 2])
        # Allocated in reset()
        self.pos_su = np.zeros((cfg.N, 2))
        self.pos_du = np.zeros((cfg.N, 2))
        self.pos_jammer = np.zeros(2)
        self.pos_uav = np.zeros((cfg.K, 3))
        self.channels: Dict[str, np.ndarray] = {}
        self.uav_assignment = np.zeros(cfg.N, dtype=int)
        self._gamma_prev = np.zeros(cfg.N)
        self._mode_prev = np.zeros(cfg.N, dtype=int)
        self._uav_energy = np.zeros(cfg.K)
        self._pj_hat: float = cfg.P_J

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset to new episode. Returns (su_obs (N,4), uav_obs (K,obs_dim))."""
        cfg = self.cfg
        self._step_count = 0
        self.pos_su = self.rng.uniform(0, cfg.area_size, size=(cfg.N, 2))
        self.pos_du = self.rng.uniform(0, cfg.area_size, size=(cfg.N, 2))
        self.pos_jammer = self.rng.uniform(0, cfg.area_size, size=(2,))
        xy_uav = self.rng.uniform(0, cfg.area_size, size=(cfg.K, 2))
        self.pos_uav = np.column_stack([xy_uav, np.full(cfg.K, cfg.H_init)])
        self._uav_energy = np.full(cfg.K, cfg.E_max)
        self._gamma_prev = np.zeros(cfg.N)
        self._mode_prev = np.zeros(cfg.N, dtype=int)
        self._pj_hat = max(float(cfg.P_J + self.rng.normal(0, 0.05 * cfg.P_J)), 0.0)
        self.channels = compute_all_channels(
            self.pos_su, self.pos_du, self._pos_rbs,
            self.pos_uav, self.pos_jammer, cfg, self.rng)
        self.uav_assignment = self._assign_uav_to_su()
        return self._build_observations()

    def step(
        self,
        su_actions: np.ndarray,
        uav_actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], bool, dict]:
        """Advance one timestep.

        su_actions  (N,4): [alpha, logit_m0, logit_m1, logit_m2]
        uav_actions (K,3): normalised displacements ∈ [-1,1]³
        Returns: su_obs, uav_obs, rewards{'su','uav'}, done, info
        """
        cfg = self.cfg
        self._step_count += 1

        prev_pos_uav = self.pos_uav.copy()
        self._update_uav_positions(uav_actions)

        su_actions = np.asarray(su_actions, dtype=float)
        alphas = np.clip(su_actions[:, 0], 0.0, 1.0)
        modes = np.argmax(su_actions[:, 1:4], axis=1).astype(int)

        self.channels = compute_all_channels(
            self.pos_su, self.pos_du, self._pos_rbs,
            self.pos_uav, self.pos_jammer, cfg, self.rng)
        self.uav_assignment = self._assign_uav_to_su()

        sinr_vals = np.zeros(cfg.N)
        su_rewards = np.zeros(cfg.N)
        uav_delta_norm = self._uav_delta_norms(prev_pos_uav)

        for i in range(cfg.N):
            sinr_vals[i] = compute_sinr(
                i, modes[i], alphas[i], self.channels, self.uav_assignment, cfg)
            su_rewards[i] = compute_reward(
                sinr_vals[i], alphas[i], modes[i],
                uav_delta_norm[self.uav_assignment[i]], cfg)

        uav_rewards = np.zeros(cfg.K)
        for k in range(cfg.K):
            assigned = np.where(self.uav_assignment == k)[0]
            if len(assigned):
                uav_rewards[k] = float(np.mean(np.log2(1.0 + sinr_vals[assigned])))

        self._gamma_prev = sinr_vals.copy()
        self._mode_prev = modes.copy()
        self._pj_hat = max(float(cfg.P_J + self.rng.normal(0, 0.05 * cfg.P_J)), 0.0)

        done = self._step_count >= cfg.steps_per_episode
        info = {"sinr": sinr_vals, "modes": modes,
                "alphas": alphas, "uav_energy": self._uav_energy.copy()}
        su_obs, uav_obs = self._build_observations()
        return su_obs, uav_obs, {"su": su_rewards, "uav": uav_rewards}, done, info

    def get_state_dim(self) -> int:
        """Total dimension of the flattened global state vector."""
        cfg = self.cfg
        return cfg.N * cfg.su_obs_dim + cfg.K * cfg.uav_obs_dim

    def get_global_state(self) -> np.ndarray:
        """Concatenated observations → shape (get_state_dim(),)."""
        su_obs, uav_obs = self._build_observations()
        return np.concatenate([su_obs.flatten(), uav_obs.flatten()])

    # ── Internal helpers ─────────────────────────────────────────────────────────

    def _update_uav_positions(self, uav_actions: np.ndarray) -> None:
        """Move UAVs: clip arena bounds, altitude [H_min, H_max], deduct energy."""
        cfg = self.cfg
        delta = np.clip(np.asarray(uav_actions, dtype=float), -1.0, 1.0) * cfg.v_max * cfg.dt
        self.pos_uav += delta
        self.pos_uav[:, 0] = np.clip(self.pos_uav[:, 0], 0.0, cfg.area_size)
        self.pos_uav[:, 1] = np.clip(self.pos_uav[:, 1], 0.0, cfg.area_size)
        self.pos_uav[:, 2] = np.clip(self.pos_uav[:, 2], cfg.H_min, cfg.H_max)
        self._uav_energy = np.maximum(
            self._uav_energy - np.linalg.norm(delta, axis=1) * 10.0, 0.0)

    def _assign_uav_to_su(self) -> np.ndarray:
        """Return (N,) array: index of nearest UAV for each SU (3-D dist, SU at z=0)."""
        assignment = np.zeros(self.cfg.N, dtype=int)
        for i in range(self.cfg.N):
            su_3d = np.append(self.pos_su[i], 0.0)
            assignment[i] = int(np.argmin(np.linalg.norm(self.pos_uav - su_3d, axis=1)))
        return assignment

    def _build_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build (su_obs (N,4), uav_obs (K,3+2N+2)) from current state."""
        cfg = self.cfg
        pj_norm = float(np.clip(self._pj_hat / (cfg.P_J + 1e-9), 0.0, 2.0))

        # SU observations
        su_obs = np.zeros((cfg.N, cfg.su_obs_dim))
        for i in range(cfg.N):
            su_obs[i] = [
                float(np.clip(self._gamma_prev[i] / self._SINR_NORM, 0.0, 1.0)),
                0.0,                                    # SU energy placeholder
                float(self._mode_prev[i]) / 2.0,        # mode ∈ {0,1,2} → [0,1]
                pj_norm,
            ]

        # UAV observations
        uav_obs = np.zeros((cfg.K, cfg.uav_obs_dim))
        for k in range(cfg.K):
            px_n = self.pos_uav[k, 0] / cfg.area_size
            py_n = self.pos_uav[k, 1] / cfg.area_size
            pz_n = self.pos_uav[k, 2] / cfg.H_max
            # Log-normalise A2G gains: log10(g+eps)/10+1 clipped to [0,1]
            g_su_n = np.clip(np.log10(self.channels["g_bar_SU"][:, k] + 1e-30) / 10.0 + 1.0, 0.0, 1.0)
            g_ud_n = np.clip(np.log10(self.channels["g_bar_UD"][:, k] + 1e-30) / 10.0 + 1.0, 0.0, 1.0)
            e_n = float(np.clip(self._uav_energy[k] / cfg.E_max, 0.0, 1.0))
            uav_obs[k] = np.concatenate([[px_n, py_n, pz_n], g_su_n, g_ud_n, [pj_norm, e_n]])

        return su_obs, uav_obs

    def _uav_delta_norms(self, prev_pos_uav: np.ndarray) -> np.ndarray:
        """Normalised displacement magnitude for each UAV ∈ [0,1]."""
        cfg = self.cfg
        max_step = cfg.v_max * cfg.dt * float(np.sqrt(3))
        return np.clip(np.linalg.norm(self.pos_uav - prev_pos_uav, axis=1)
                       / max(max_step, 1e-9), 0.0, 1.0)
