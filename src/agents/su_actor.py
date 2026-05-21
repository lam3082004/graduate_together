"""
su_actor.py — SU Agent Actor using numpy MLP (torch-free).

Maps local SU observation → (alpha, mode_logits).
  alpha      : continuous reflection coefficient in [0,1] via sigmoid
  mode_logits: 3-dim raw logits; use softmax_np for probabilities
"""

import numpy as np
from nn.layers import MLP, sigmoid, softmax_np


class SUActor:
    """
    SU Agent Actor: obs (B, su_obs_dim) → alpha (B,1), mode_logits (B,3).

    Architecture:
        backbone  : MLP [obs_dim → 256 → 256 → 128] with ReLU
        alpha_head: MLP [128 → 1] with sigmoid output (∈ [0,1])
        mode_head : MLP [128 → 3] linear (raw logits)
    """

    def __init__(self, obs_dim: int = 4,
                 hidden: list | None = None) -> None:
        if hidden is None:
            hidden = [256, 256, 128]
        dims_backbone = [obs_dim] + list(hidden)
        self.backbone  = MLP(dims_backbone, hidden_act='relu', out_act=None)
        self.alpha_head = MLP([hidden[-1], 1], hidden_act='relu', out_act='sigmoid')
        self.mode_head  = MLP([hidden[-1], 3], hidden_act='relu', out_act=None)
        self._feat: np.ndarray | None = None

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: np.ndarray):
        """
        Args:
            obs : (B, obs_dim)
        Returns:
            alpha       : (B, 1) in [0, 1]
            mode_logits : (B, 3) raw logits
        """
        obs = np.atleast_2d(obs)
        feat = self.backbone.forward(obs)
        self._feat = feat
        alpha = self.alpha_head.forward(feat)
        mode_logits = self.mode_head.forward(feat)
        return alpha, mode_logits

    def get_action(self, obs: np.ndarray, tau: float = 1.0,
                   explore: bool = False):
        """
        Sample action; optionally add Gumbel noise for exploration.

        Returns:
            alpha      : (B,)   reflection coefficient
            mode_probs : (B, 3) softmax probabilities
            mode_idx   : (B,)   argmax integer mode index
        """
        obs = np.atleast_2d(obs)
        alpha_raw, mode_logits = self.forward(obs)
        alpha = alpha_raw.squeeze(-1)  # (B,)
        if explore:
            gumbel = -np.log(
                -np.log(np.random.uniform(1e-20, 1.0, mode_logits.shape) + 1e-20) + 1e-20
            )
            mode_logits = (mode_logits + gumbel) / max(tau, 1e-6)
        mode_probs = softmax_np(mode_logits)
        mode_idx = mode_probs.argmax(axis=-1)
        return alpha, mode_probs, mode_idx

    # ── Backward helpers ──────────────────────────────────────────────────────

    def backward_alpha(self, d_alpha: np.ndarray) -> None:
        """Backprop through alpha head → backbone."""
        d_feat = self.alpha_head.backward(d_alpha)
        self.backbone.backward(d_feat)

    def backward_mode(self, d_mode_logits: np.ndarray) -> None:
        """Backprop through mode head → backbone."""
        d_feat = self.mode_head.backward(d_mode_logits)
        self.backbone.backward(d_feat)

    # ── Param access ──────────────────────────────────────────────────────────

    def params_and_grads(self) -> list:
        return (self.backbone.params_and_grads()
                + self.alpha_head.params_and_grads()
                + self.mode_head.params_and_grads())

    def copy_from(self, other: 'SUActor') -> None:
        self.backbone.copy_weights_from(other.backbone)
        self.alpha_head.copy_weights_from(other.alpha_head)
        self.mode_head.copy_weights_from(other.mode_head)

    def soft_update_from(self, other: 'SUActor', tau: float) -> None:
        self.backbone.soft_update_from(other.backbone, tau)
        self.alpha_head.soft_update_from(other.alpha_head, tau)
        self.mode_head.soft_update_from(other.mode_head, tau)
