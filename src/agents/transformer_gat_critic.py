"""
transformer_gat_critic.py — Centralized Critic with attention-weighted agent
feature aggregation and channel-topology features (numpy-only, torch-free).

Architecture (Double-Q):
  1. Project each SU obs+action  → d-dim embedding (su_proj MLP)
  2. Project each UAV obs+action → d-dim embedding (uav_proj MLP)
  3. Stack all N+K embeddings → dot-product self-attention → context vectors
  4. For each UAV embedding: add channel-gain-weighted SU embedding (GAT-like)
  5. Mean-pool → MLP [flat_dim → 512 → 256 → 128 → 64] → twin Q heads (Q1, Q2)
"""

import numpy as np
from nn.layers import MLP, softmax_np, relu


# ── Attention helpers ─────────────────────────────────────────────────────────

def _dot_attention(emb: np.ndarray) -> np.ndarray:
    """
    Scaled dot-product self-attention.
    emb : (B, M, d)  → context : (B, M, d)
    """
    B, M, d = emb.shape
    scale = np.sqrt(max(d, 1))
    # scores: (B, M, M)
    scores = np.einsum('bmd,bnd->bmn', emb, emb) / scale
    attn = softmax_np(scores)           # (B, M, M)
    return np.einsum('bmn,bnd->bmd', attn, emb)  # (B, M, d)


def _channel_weighted_su(uav_emb: np.ndarray, su_emb: np.ndarray,
                          channel_gains: np.ndarray) -> np.ndarray:
    """
    For each UAV k: compute sum_i (g_ki / sum_j g_kj) * su_emb_i.
    uav_emb     : (B, K, d)
    su_emb      : (B, N, d)
    channel_gains: (B, K, N)
    Returns extra_uav: (B, K, d)
    """
    # Normalize gains row-wise: (B, K, N)
    g_norm = channel_gains / (channel_gains.sum(axis=-1, keepdims=True) + 1e-12)
    # Weighted sum of SU embeddings: (B, K, d)
    return np.einsum('bkn,bnd->bkd', g_norm, su_emb)


# ── Centralized Critic ────────────────────────────────────────────────────────

class CentralizedCritic:
    """
    Double-Q centralized critic with attention + channel-topology features.

    All inputs are numpy arrays; no torch dependency.
    """

    def __init__(self, N: int = 5, K: int = 2,
                 su_obs_dim: int = 4, uav_obs_dim: int = 15,
                 su_action_dim: int = 4, uav_action_dim: int = 3,
                 d: int = 128,
                 mlp_hidden: list | None = None) -> None:
        if mlp_hidden is None:
            mlp_hidden = [512, 256, 128, 64]
        self.N, self.K, self.d = N, K, d

        su_in  = su_obs_dim  + su_action_dim
        uav_in = uav_obs_dim + uav_action_dim

        # Projection MLPs (one per agent type)
        self.su_proj  = MLP([su_in, d],  hidden_act='relu', out_act=None)
        self.uav_proj = MLP([uav_in, d], hidden_act='relu', out_act=None)

        # Flat dimension after mean-pool of (N+K) context vectors + (K) channel extras
        flat_dim = (N + K) * d + K * d   # context_mean:(N+K)*d + uav_chan_extra:K*d
        # Flatten to vector before MLP
        flat_in = (N + K + K) * d        # (N+K context + K channel) * d then flatten

        # Twin Q heads
        self.q1_mlp = MLP([flat_in] + mlp_hidden + [1], hidden_act='relu', out_act=None)
        self.q2_mlp = MLP([flat_in] + mlp_hidden + [1], hidden_act='relu', out_act=None)

        # Cache for backward
        self._flat: np.ndarray | None = None

    def _embed(self, su_obs: np.ndarray, uav_obs: np.ndarray,
               su_actions: np.ndarray, uav_actions: np.ndarray,
               channel_gains: np.ndarray):
        """Build flat feature vector. Returns (B, flat_in) and caches it."""
        B = su_obs.shape[0]
        N, K, d = self.N, self.K, self.d

        # 1. Project observations+actions to embeddings
        su_in  = np.concatenate([su_obs,  su_actions],  axis=-1)   # (B, N, su_in)
        uav_in = np.concatenate([uav_obs, uav_actions], axis=-1)   # (B, K, uav_in)

        su_emb  = np.stack([self.su_proj.forward(su_in[:, i]) for i in range(N)], axis=1)
        uav_emb = np.stack([self.uav_proj.forward(uav_in[:, k]) for k in range(K)], axis=1)

        # 2. Self-attention over all N+K embeddings
        all_emb = np.concatenate([uav_emb, su_emb], axis=1)  # (B, K+N, d)
        context = _dot_attention(all_emb)                     # (B, K+N, d)

        # 3. Channel-gain weighted SU embeddings for UAV nodes
        chan_extra = _channel_weighted_su(uav_emb, su_emb, channel_gains)  # (B, K, d)

        # 4. Flatten: [context (K+N) * d, chan_extra K * d]
        flat = np.concatenate([
            context.reshape(B, -1),      # (B, (K+N)*d)
            chan_extra.reshape(B, -1),   # (B, K*d)
        ], axis=-1)                      # (B, flat_in)
        self._flat = flat
        return flat

    def forward(self, su_obs: np.ndarray, uav_obs: np.ndarray,
                su_actions: np.ndarray, uav_actions: np.ndarray,
                channel_gains: np.ndarray):
        """
        Returns q1 (B,), q2 (B,).
        All inputs: numpy float32 arrays.
        """
        flat = self._embed(su_obs, uav_obs, su_actions, uav_actions, channel_gains)
        q1 = self.q1_mlp.forward(flat).squeeze(-1)  # (B,)
        q2 = self.q2_mlp.forward(flat).squeeze(-1)  # (B,)
        return q1, q2

    def backward_critic(self, dq: np.ndarray, head: int = 1) -> None:
        """Backprop MSE gradient through selected Q head."""
        dq_col = dq.reshape(-1, 1)
        if head == 1:
            self.q1_mlp.backward(dq_col)
        else:
            self.q2_mlp.backward(dq_col)

    def params_and_grads(self) -> list:
        return (self.su_proj.params_and_grads()
                + self.uav_proj.params_and_grads()
                + self.q1_mlp.params_and_grads()
                + self.q2_mlp.params_and_grads())

    def copy_from(self, other: 'CentralizedCritic') -> None:
        self.su_proj.copy_weights_from(other.su_proj)
        self.uav_proj.copy_weights_from(other.uav_proj)
        self.q1_mlp.copy_weights_from(other.q1_mlp)
        self.q2_mlp.copy_weights_from(other.q2_mlp)

    def soft_update_from(self, other: 'CentralizedCritic', tau: float) -> None:
        self.su_proj.soft_update_from(other.su_proj, tau)
        self.uav_proj.soft_update_from(other.uav_proj, tau)
        self.q1_mlp.soft_update_from(other.q1_mlp, tau)
        self.q2_mlp.soft_update_from(other.q2_mlp, tau)


# Alias to preserve any legacy import as TransformerGATCritic
TransformerGATCritic = CentralizedCritic
