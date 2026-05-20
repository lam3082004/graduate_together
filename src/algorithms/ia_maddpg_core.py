"""
ia_maddpg_core.py — Initialization, action selection, expert policies,
                     warmup, and persistence for IA-MADDPG.
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from src.agents.su_actor import SUActor
from src.agents.uav_actor import UAVActor
from src.algorithms.per_buffer import PERBuffer


# ── Transformer-GAT Centralized Critic ────────────────────────────────────────

class TransformerGATCritic(nn.Module):
    """
    Centralized critic: Transformer encoder over agent tokens →
    GAT-style cross-attention over UAV→SU edges → MLP → Q scalar.

    Input dims are assembled as:
        global_obs  = concat(su_obs[N], uav_obs[K])  dim=N*su_obs+K*uav_obs
        global_acts = concat(su_acts[N,4], uav_acts[K,3])
    """

    def __init__(self, su_obs_dim: int, uav_obs_dim: int,
                 su_act_dim: int, uav_act_dim: int,
                 N: int, K: int, d_model: int = 128,
                 nhead: int = 4, nlayers: int = 2,
                 gat_hidden: int = 64, gat_heads: int = 4) -> None:
        super().__init__()
        self.N, self.K = N, K
        n_agents = N + K
        # Per-agent token embedding
        su_in = su_obs_dim + su_act_dim
        uav_in = uav_obs_dim + uav_act_dim
        self.su_embed = nn.Linear(su_in, d_model)
        self.uav_embed = nn.Linear(uav_in, d_model)

        # Transformer encoder over n_agents tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # GAT: each SU attends to K UAV tokens
        self.gat_q = nn.Linear(d_model, gat_heads * gat_hidden)
        self.gat_k = nn.Linear(d_model, gat_heads * gat_hidden)
        self.gat_v = nn.Linear(d_model, gat_heads * gat_hidden)
        self.gat_out = nn.Linear(gat_heads * gat_hidden, d_model)
        self.gat_heads = gat_heads
        self.gat_hidden = gat_hidden
        self.scale = (gat_hidden) ** -0.5

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_agents * d_model, 512),
            nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self,
                su_obs: torch.Tensor, uav_obs: torch.Tensor,
                su_acts: torch.Tensor, uav_acts: torch.Tensor
                ) -> torch.Tensor:
        """
        su_obs  : (B, N, su_obs_dim)
        uav_obs : (B, K, uav_obs_dim)
        su_acts : (B, N, su_act_dim)
        uav_acts: (B, K, uav_act_dim)
        Returns : (B, 1) Q value
        """
        B = su_obs.size(0)
        su_tok = self.su_embed(torch.cat([su_obs, su_acts], dim=-1))    # (B,N,d)
        uav_tok = self.uav_embed(torch.cat([uav_obs, uav_acts], dim=-1))  # (B,K,d)

        # Concatenate all agent tokens: SU first, then UAV
        tokens = torch.cat([su_tok, uav_tok], dim=1)                    # (B,N+K,d)
        tokens = self.transformer(tokens)                               # (B,N+K,d)

        # GAT: SU tokens attend to UAV tokens
        H, Hd = self.gat_heads, self.gat_hidden
        su_q = self.gat_q(tokens[:, :self.N]).view(B, self.N, H, Hd)
        uav_k = self.gat_k(tokens[:, self.N:]).view(B, self.K, H, Hd)
        uav_v = self.gat_v(tokens[:, self.N:]).view(B, self.K, H, Hd)
        # Attention: (B, N, H, K)
        attn = torch.einsum('bnhd,bkhd->bnhk', su_q, uav_k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        agg = torch.einsum('bnhk,bkhd->bnhd', attn, uav_v)             # (B,N,H,Hd)
        agg = self.gat_out(agg.reshape(B, self.N, H * Hd))             # (B,N,d)

        # Update SU tokens with GAT output
        tokens_out = torch.cat([tokens[:, :self.N] + agg,
                                tokens[:, self.N:]], dim=1)            # (B,N+K,d)
        flat = tokens_out.reshape(B, -1)
        return self.mlp(flat)


# ── Expert policies ────────────────────────────────────────────────────────────

class SUExpert:
    """Greedy expert: maximize instantaneous SINR over modes {0,1,2}."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def act(self, channels: dict, uav_positions: np.ndarray) -> np.ndarray:
        """
        Returns su_actions: (N, 4) = [alpha, logit0, logit1, logit2]
        with one-hot logits for best mode and alpha=optimal.
        """
        cfg = self.cfg
        N = cfg.N
        actions = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            best_sinr, best_mode, best_alpha = -np.inf, 0, 0.2
            for alpha in np.linspace(0.1, 1.0, 10):
                sinr0 = _sinr_mode0(i, alpha, channels, cfg)
                sinr1 = _sinr_mode1(i, alpha, channels, cfg)
                sinr2 = _sinr_mode2_best(i, alpha, channels, uav_positions, cfg)
                for mode, sinr in enumerate([sinr0, sinr1, sinr2]):
                    if sinr > best_sinr:
                        best_sinr, best_mode, best_alpha = sinr, mode, alpha
            logits = np.full(3, -10.0, dtype=np.float32)
            logits[best_mode] = 10.0
            actions[i] = [best_alpha, *logits]
        return actions


class UAVExpert:
    """Heuristic expert: move each UAV toward best sum-SINR position."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def act(self, positions_uav: np.ndarray, channels: dict,
            su_indices: list) -> np.ndarray:
        """
        Returns uav_actions: (K, 3) normalised displacement ∈ [-1,1].
        Each UAV moves toward the midpoint of its assigned SUs.
        """
        cfg = self.cfg
        K = cfg.K
        actions = np.zeros((K, 3), dtype=np.float32)
        for k in range(K):
            assigned = su_indices[k]
            if not assigned:
                continue
            # Target: midpoint between assigned SU and DU positions
            # Simple heuristic: if SINR improvement expected, step toward SU centroid
            # Using small gradient step in direction of steepest sum-SINR
            best_delta = np.zeros(3)
            best_gain = 0.0
            for direction in _unit_directions():
                new_pos = positions_uav[k] + direction * cfg.v_max * cfg.dt
                new_pos[2] = np.clip(new_pos[2], cfg.H_min, cfg.H_max)
                gain = sum(_sinr_mode2_at_pos(i, 0.5, channels, new_pos, cfg)
                           for i in assigned)
                if gain > best_gain:
                    best_gain = gain
                    best_delta = direction
            actions[k] = best_delta
        return actions


# ── SINR helpers (used by experts) ────────────────────────────────────────────

def _sinr_mode0(i: int, alpha: float, ch: dict, cfg) -> float:
    num = cfg.G * cfg.P_J * ch["g_JS"][i] * ch["g_SD"][i] * alpha ** 2
    den = cfg.P_J * ch["g_JD"][i] + cfg.N0
    return num / max(den, 1e-30)


def _sinr_mode1(i: int, alpha: float, ch: dict, cfg) -> float:
    den_r = cfg.P_J * ch["g_JR"] + cfg.N0
    den_d = cfg.P_J * ch["g_JD"][i] + cfg.N0
    hop1 = cfg.G * cfg.P_J * ch["g_JS"][i] * ch["g_SR"][i] * alpha**2 / max(den_r, 1e-30)
    hop2 = cfg.G * cfg.P_J * ch["g_JR"] * ch["g_RD"][i] * alpha**2 / max(den_d, 1e-30)
    return min(hop1, hop2)


def _sinr_mode2_best(i: int, alpha: float, ch: dict,
                     uav_positions: np.ndarray, cfg) -> float:
    """Best SINR across all K UAVs."""
    best = 0.0
    for k in range(cfg.K):
        s = _sinr_mode2_at_pos(i, alpha, ch, uav_positions[k], cfg, k=k)
        best = max(best, s)
    return best


def _sinr_mode2_at_pos(i: int, alpha: float, ch: dict,
                        uav_pos: np.ndarray, cfg, k: int = 0) -> float:
    g_su = ch["g_bar_SU"][i, k]
    g_ud = ch["g_bar_UD"][i, k]
    g_ju = ch["g_bar_JU"][k]
    den_u = cfg.P_J * g_ju + cfg.N0
    den_d = cfg.P_J * ch["g_JD"][i] + cfg.N0
    hop1 = cfg.G * cfg.P_J * ch["g_JS"][i] * g_su * alpha**2 / max(den_u, 1e-30)
    hop2 = cfg.G * cfg.P_J * g_ju * g_ud * alpha**2 / max(den_d, 1e-30)
    return min(hop1, hop2)


def _unit_directions():
    """Six cardinal ±x, ±y, ±z unit vectors."""
    for ax in range(3):
        for sign in (1, -1):
            d = np.zeros(3)
            d[ax] = sign
            yield d


# ── Core initializer ──────────────────────────────────────────────────────────

def build_agents(cfg):
    """Construct actors, target actors, critic, target critic, and optimizers."""
    import torch.optim as optim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    su_actors = [SUActor(cfg.su_obs_dim, cfg.actor_hidden).to(device)
                 for _ in range(cfg.N)]
    uav_actors = [UAVActor(cfg.uav_obs_dim, cfg.actor_hidden).to(device)
                  for _ in range(cfg.K)]

    su_actors_tgt = [deepcopy(a) for a in su_actors]
    uav_actors_tgt = [deepcopy(a) for a in uav_actors]

    critic = TransformerGATCritic(
        su_obs_dim=cfg.su_obs_dim, uav_obs_dim=cfg.uav_obs_dim,
        su_act_dim=cfg.su_action_dim, uav_act_dim=cfg.uav_action_dim,
        N=cfg.N, K=cfg.K,
        d_model=cfg.transformer_d_model, nhead=cfg.transformer_nhead,
        nlayers=cfg.transformer_nlayers,
        gat_hidden=cfg.gat_hidden, gat_heads=cfg.gat_heads,
    ).to(device)
    critic_tgt = deepcopy(critic)

    opt_su = [optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in su_actors]
    opt_uav = [optim.Adam(a.parameters(), lr=cfg.lr_actor) for a in uav_actors]
    opt_critic = optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    return (su_actors, uav_actors, su_actors_tgt, uav_actors_tgt,
            critic, critic_tgt, opt_su, opt_uav, opt_critic, device)
