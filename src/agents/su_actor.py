"""
su_actor.py — SU Agent Actor Network for IA-MADDPG.

Maps local SU observation → (alpha, mode_logits).
- alpha:       continuous reflection coefficient in [0,1] via sigmoid
- mode_logits: 3-dim raw logits for Gumbel-Softmax (modes 0=D2D, 1=RBS, 2=UAV)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden: list[int]) -> nn.Sequential:
    """Helper: build MLP with LayerNorm + ReLU activations."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
        prev = h
    return nn.Sequential(*layers)


class SUActor(nn.Module):
    """
    SU Agent Actor: maps local observation → (alpha, mode_logits).

    Input  obs: (B, obs_dim) — [gamma_prev, E_norm, mode_prev, P_J_hat]
    Output alpha:       (B, 1)  — sigmoid-activated reflection coefficient
           mode_logits: (B, 3)  — raw logits for Gumbel-Softmax
    """

    def __init__(self, obs_dim: int = 4, hidden: list[int] | None = None) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]

        self.backbone = _build_mlp(obs_dim, hidden)
        out_dim = hidden[-1]

        # Two heads share the backbone
        self.alpha_head = nn.Linear(out_dim, 1)        # → sigmoid → [0,1]
        self.mode_head  = nn.Linear(out_dim, 3)        # raw logits

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, obs_dim)
        Returns:
            alpha:       (B, 1)  in [0, 1]
            mode_logits: (B, 3)  unnormalised logits
        """
        feat = self.backbone(obs)
        alpha       = torch.sigmoid(self.alpha_head(feat))
        mode_logits = self.mode_head(feat)
        return alpha, mode_logits

    def get_action(
        self,
        obs: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action using Gumbel-Softmax for differentiable discrete selection.

        Args:
            obs:  (B, obs_dim)
            tau:  Gumbel-Softmax temperature (lower → more discrete)
            hard: if True, straight-through estimator (one-hot forward, soft backward)

        Returns:
            alpha:      (B, 1)  — reflection coefficient
            mode_probs: (B, 3)  — soft / hard mode probabilities
            mode_idx:   (B,)    — argmax integer mode index
        """
        alpha, mode_logits = self.forward(obs)
        mode_probs = F.gumbel_softmax(mode_logits, tau=tau, hard=hard, dim=-1)
        mode_idx   = mode_probs.argmax(dim=-1)
        return alpha, mode_probs, mode_idx
