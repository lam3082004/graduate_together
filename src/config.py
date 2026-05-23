"""
config.py — Centralised hyperparameters for the UAV-assisted
Ambient Backscatter Anti-Jamming Multi-Agent DRL system.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Config:
    # ── Network layout ────────────────────────────────────────────────────────
    area_size: float = 200.0          # 200×200 m²
    N: int = 5                        # SU-DU pairs
    K: int = 2                        # UAVs

    H_min: float = 20.0               # Minimum UAV altitude (m)
    H_max: float = 100.0              # Maximum UAV altitude (m)
    H_init: float = 50.0              # Initial UAV altitude (m)

    # ── Physics ───────────────────────────────────────────────────────────────
    P_J: float = 1.0                  # Jammer transmit power (W)
    N0: float = 1e-4                  # Noise floor (W)
    G: float = 1e4                    # Ambient backscatter gain constant
    eta: float = 2.7                  # Ground path-loss exponent
    a_los: float = 9.61               # LoS sigmoid parameter a
    b_los: float = 0.16               # LoS sigmoid parameter b
    xi_los: float = 1.0               # LoS additional loss factor
    xi_nlos: float = 20.0             # NLoS additional loss factor
    fc: float = 2e9                   # Carrier frequency (Hz)

    # ── UAV kinematics ────────────────────────────────────────────────────────
    v_max: float = 10.0               # Maximum speed (m/s)
    dt: float = 1.0                   # Time step (s)
    E_max: float = 1e5                # Maximum UAV energy budget (J)

    # ── SINR threshold ────────────────────────────────────────────────────────
    gamma_th_dB: float = 5.0

    @property
    def gamma_th(self) -> float:
        """Linear SINR threshold."""
        return 10 ** (self.gamma_th_dB / 10)

    # ── Reward weights ────────────────────────────────────────────────────────
    w1: float = 1.0
    w2: float = 0.5
    w3: float = 0.1
    w4: float = 0.05

    # ── Training ──────────────────────────────────────────────────────────────
    episodes: int = 600
    steps_per_episode: int = 200
    batch_size: int = 256
    buffer_size: int = 100_000
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005                # Soft-update coefficient
    lambda_il_init: float = 1.0
    lambda_il_decay: float = 0.995
    warmup_steps: int = 5_000
    per_alpha: float = 0.6
    per_beta_init: float = 0.4
    per_beta_steps: int = 100_000
    td3_noise_std: float = 0.1
    td3_noise_clip: float = 0.3
    policy_delay: int = 2

    # ── Network architecture ──────────────────────────────────────────────────
    actor_hidden: list = field(default_factory=lambda: [256, 256, 128])
    critic_hidden: list = field(default_factory=lambda: [512, 256, 128, 64])
    transformer_d_model: int = 128
    transformer_nhead: int = 4
    transformer_nlayers: int = 2
    gat_hidden: int = 64
    gat_heads: int = 4

    # ── Observation / action dimensions (derived) ─────────────────────────────
    @property
    def su_obs_dim(self) -> int:
        """[gamma_prev, E_norm, mode_prev, P_J_hat]"""
        return 4

    @property
    def uav_obs_dim(self) -> int:
        """[px, py, pz, g_SU×N, g_UD×N, P_J_hat, E_remain_norm]"""
        return 3 + 2 * self.N + 2

    @property
    def su_action_dim(self) -> int:
        """[alpha, logit_mode0, logit_mode1, logit_mode2]"""
        return 4

    @property
    def uav_action_dim(self) -> int:
        """[dx, dy, dz] normalised displacement"""
        return 3

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 42
