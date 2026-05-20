"""
uav_actor.py — UAV Agent Actor Network for IA-MADDPG.

Maps local UAV observation → (dx, dy, dz) ∈ [-1,1].
Scale by v_max * dt outside this module to get physical displacement.

UAV obs: [px, py, pz, g_SU×N, g_UD×N, P_J_hat, E_remain_norm]  dim=3+2N+2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
        prev = h
    return nn.Sequential(*layers)


class UAVActor(nn.Module):
    """
    UAV Agent Actor: maps local observation → normalised 3-D displacement.

    Input  obs: (B, obs_dim) — [px, py, pz, g_SU×N, g_UD×N, P_J_hat, E_norm]
    Output action: (B, 3) in [-1, 1]  multiply by v_max*dt for real displacement
    """

    def __init__(self, obs_dim: int = 15, hidden: list[int] | None = None) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]

        self.backbone = _build_mlp(obs_dim, hidden)
        self.action_head = nn.Linear(hidden[-1], 3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            action: (B, 3) in [-1, 1]
        """
        feat = self.backbone(obs)
        return torch.tanh(self.action_head(feat))

    def get_action(
        self,
        obs: torch.Tensor,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass with optional additive Gaussian exploration noise.

        Args:
            obs:       (B, obs_dim)
            noise_std: standard deviation of exploration noise (0 = deterministic)

        Returns:
            action: (B, 3) clipped to [-1, 1]
        """
        action = self.forward(obs)
        if noise_std > 0.0:
            noise  = torch.randn_like(action) * noise_std
            action = (action + noise).clamp(-1.0, 1.0)
        return action
