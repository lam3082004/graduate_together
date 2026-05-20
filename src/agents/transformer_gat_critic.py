"""
transformer_gat_critic.py — Centralised Critic combining Transformer + GAT.

Architecture (IA-MADDPG / Double-Q):
  1. Project each agent obs → d_model via type-specific linear layers
  2. Transformer encoder over the N+K token sequence
  3. Multi-head GAT over UAV→SU topology graph (channel_gains as edge weights)
  4. Mean-pool Transformer output + GAT output
  5. Concat with flat action vector → twin MLP heads → Q1, Q2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── GAT primitives ──────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """Single-head Graph Attention Layer (Veličković et al., 2018)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W   = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Linear(out_dim, 1, bias=False)
        self.a_dst = nn.Linear(out_dim, 1, bias=False)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (B, V, in_dim)  node features
            adj: (B, V, V)       adjacency bias (0 = connected, -inf = masked)
        Returns:
            (B, V, out_dim)
        """
        Wh   = self.W(h)                                   # (B, V, out_dim)
        e    = self.a_src(Wh) + self.a_dst(Wh).transpose(-1, -2)  # (B, V, V)
        e    = self.leaky(e) + adj
        attn = F.softmax(e, dim=-1)                        # (B, V, V)
        return F.elu(torch.bmm(attn, Wh))                  # (B, V, out_dim)


class MultiHeadGAT(nn.Module):
    """Multi-head GAT with ELU activation and output projection."""

    def __init__(self, in_dim: int, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = nn.ModuleList([GATLayer(in_dim, hidden) for _ in range(heads)])
        self.proj  = nn.Linear(hidden * heads, hidden)
        self.norm  = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   (B, V, in_dim)
            adj: (B, V, V)
        Returns:
            (B, V, hidden)
        """
        out = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        return self.norm(self.proj(out))


# ─── Centralized Critic ───────────────────────────────────────────────────────

def _build_mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class TransformerGATCritic(nn.Module):
    """
    Centralised Critic (Double-Q) combining Transformer + Multi-head GAT.

    Node indexing: UAVs first [0..K-1], then SUs [K..K+N-1].
    """

    def __init__(
        self,
        N: int = 5, K: int = 2,
        su_obs_dim: int = 4, uav_obs_dim: int = 15,
        su_action_dim: int = 4, uav_action_dim: int = 3,
        d_model: int = 128, nhead: int = 4, nlayers: int = 2,
        gat_hidden: int = 64, gat_heads: int = 4,
        mlp_hidden: list[int] | None = None,
    ) -> None:
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = [512, 256, 128, 64]

        self.N = N
        self.K = K
        num_nodes     = N + K
        total_act_dim = N * su_action_dim + K * uav_action_dim

        # Per-type input projections (obs dims differ)
        self.uav_proj = nn.Linear(uav_obs_dim, d_model)
        self.su_proj  = nn.Linear(su_obs_dim,  d_model)

        # Learnable positional encoding  (1, N+K, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, num_nodes, d_model))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        # Transformer encoder
        enc_layer     = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # Multi-head GAT
        self.gat = MultiHeadGAT(d_model, gat_hidden, heads=gat_heads)

        # MLP input: Transformer pool + GAT pool + actions
        feat_dim = d_model + gat_hidden + total_act_dim

        # Twin Q heads (Double-Q / TD3)
        self.q1 = _build_mlp(feat_dim, mlp_hidden, 1)
        self.q2 = _build_mlp(feat_dim, mlp_hidden, 1)

    # ── adjacency builder ─────────────────────────────────────────────────────

    def _build_adj(self, channel_gains: torch.Tensor) -> torch.Tensor:
        """
        Build (B, K+N, K+N) log-space adjacency bias.
        - Self-loops: 0
        - UAV k → SU i edges: log(channel_gains[:, k, i] + eps)
        - All other off-diagonal: -inf (masked)
        """
        B = channel_gains.size(0)
        V = self.K + self.N
        adj = torch.full((B, V, V), float("-inf"), device=channel_gains.device)

        # Self-loops
        idx = torch.arange(V, device=channel_gains.device)
        adj[:, idx, idx] = 0.0

        # UAV→SU directed edges (log-scale weight so large gains ≠ collapse)
        eps = 1e-9
        for k in range(self.K):
            for i in range(self.N):
                adj[:, k, self.K + i] = torch.log(channel_gains[:, k, i] + eps)

        return adj

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        su_obs_list:   list[torch.Tensor],   # N × (B, su_obs_dim)
        uav_obs_list:  list[torch.Tensor],   # K × (B, uav_obs_dim)
        su_actions:    list[torch.Tensor],   # N × (B, su_action_dim)
        uav_actions:   list[torch.Tensor],   # K × (B, uav_action_dim)
        channel_gains: torch.Tensor,         # (B, K, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q1, q2: (B, 1)
        """
        # 1. Project observations → d_model tokens; UAVs first, then SUs
        uav_tokens = [self.uav_proj(o) for o in uav_obs_list]   # K × (B, d_model)
        su_tokens  = [self.su_proj(o)  for o in su_obs_list]    # N × (B, d_model)
        tokens = torch.stack(uav_tokens + su_tokens, dim=1)      # (B, K+N, d_model)
        tokens = tokens + self.pos_enc

        # 2. Transformer encoder
        tf_out = self.transformer(tokens)                         # (B, K+N, d_model)

        # 3. GAT over UAV→SU topology
        adj     = self._build_adj(channel_gains)                  # (B, K+N, K+N)
        gat_out = self.gat(tokens, adj)                           # (B, K+N, gat_hidden)

        # 4. Mean-pool both streams
        tf_pool  = tf_out.mean(dim=1)    # (B, d_model)
        gat_pool = gat_out.mean(dim=1)   # (B, gat_hidden)

        # 5. Flatten all actions
        all_acts = torch.cat(uav_actions + su_actions, dim=-1)    # (B, total_act_dim)

        # 6. Joint feature vector → twin Q heads
        feat = torch.cat([tf_pool, gat_pool, all_acts], dim=-1)
        return self.q1(feat), self.q2(feat)
