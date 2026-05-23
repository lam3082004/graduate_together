"""
uav_actor.py — UAV Agent Actor using numpy MLP (torch-free).

Maps local UAV observation → (dx, dy, dz) ∈ [-1, 1].
Scale by v_max * dt outside this module to get physical displacement.

UAV obs: [px, py, pz, g_SU×N, g_UD×N, P_J_hat, E_remain_norm]  dim = 3+2N+2
"""

import numpy as np
from nn.layers import MLP


class UAVActor:
    """
    UAV Agent Actor: obs (B, obs_dim) → action (B, 3) ∈ [-1, 1].

    Architecture:
        backbone    : MLP [obs_dim → 256 → 256 → 128] with ReLU
        action_head : MLP [128 → 3] with tanh output (∈ [-1, 1])
    """

    def __init__(self, obs_dim: int = 15,
                 hidden: list | None = None) -> None:
        if hidden is None:
            hidden = [256, 256, 128]
        dims_backbone = [obs_dim] + list(hidden)
        self.backbone    = MLP(dims_backbone, hidden_act='relu', out_act=None)
        self.action_head = MLP([hidden[-1], 3], hidden_act='relu', out_act='tanh')

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            action: (B, 3) in [-1, 1]
        """
        obs = np.atleast_2d(obs)
        feat = self.backbone.forward(obs)
        return self.action_head.forward(feat)

    def get_action(self, obs: np.ndarray,
                   noise_std: float = 0.0) -> np.ndarray:
        """
        Forward pass with optional Gaussian exploration noise.

        Args:
            obs       : (B, obs_dim)
            noise_std : std of additive Gaussian noise (0 = deterministic)
        Returns:
            action: (B, 3) clipped to [-1, 1]
        """
        obs = np.atleast_2d(obs)
        action = self.forward(obs)
        if noise_std > 0.0:
            noise = np.random.randn(*action.shape) * noise_std
            action = np.clip(action + noise, -1.0, 1.0)
        return action

    # ── Backward ──────────────────────────────────────────────────────────────

    def backward(self, d_action: np.ndarray) -> None:
        """Backprop through action head → backbone."""
        d_feat = self.action_head.backward(d_action)
        self.backbone.backward(d_feat)

    # ── Param access ──────────────────────────────────────────────────────────

    def params_and_grads(self) -> list:
        return self.backbone.params_and_grads() + self.action_head.params_and_grads()

    def copy_from(self, other: 'UAVActor') -> None:
        self.backbone.copy_weights_from(other.backbone)
        self.action_head.copy_weights_from(other.action_head)

    def soft_update_from(self, other: 'UAVActor', tau: float) -> None:
        self.backbone.soft_update_from(other.backbone, tau)
        self.action_head.soft_update_from(other.action_head, tau)
