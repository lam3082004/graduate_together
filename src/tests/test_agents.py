"""
test_agents.py — pytest suite for SUActor, UAVActor, TransformerGATCritic,
expert policies, gradient flow.
"""
import sys
import os
import pytest
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from agents.su_actor import SUActor
from agents.uav_actor import UAVActor
from agents.transformer_gat_critic import TransformerGATCritic
from agents.expert_policy import SUExpertPolicy, UAVExpertPolicy
from environment.channel_model import compute_all_channels


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(seed=0)


@pytest.fixture
def su_actor(cfg):
    return SUActor(obs_dim=cfg.su_obs_dim, hidden=cfg.actor_hidden)


@pytest.fixture
def uav_actor(cfg):
    return UAVActor(obs_dim=cfg.uav_obs_dim, hidden=cfg.actor_hidden)


@pytest.fixture
def critic(cfg):
    return TransformerGATCritic(
        N=cfg.N, K=cfg.K,
        su_obs_dim=cfg.su_obs_dim, uav_obs_dim=cfg.uav_obs_dim,
        su_action_dim=cfg.su_action_dim, uav_action_dim=cfg.uav_action_dim,
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_nhead,
        nlayers=cfg.transformer_nlayers,
        gat_hidden=cfg.gat_hidden,
        gat_heads=cfg.gat_heads,
    )


@pytest.fixture
def dummy_channels(cfg):
    """Minimal channel dict for expert policies."""
    rng = np.random.default_rng(0)
    pos_su = rng.uniform(0, cfg.area_size, (cfg.N, 2))
    pos_du = rng.uniform(0, cfg.area_size, (cfg.N, 2))
    pos_rbs = np.array([cfg.area_size / 2, cfg.area_size / 2])
    pos_uav = np.column_stack([
        rng.uniform(0, cfg.area_size, cfg.K),
        rng.uniform(0, cfg.area_size, cfg.K),
        np.full(cfg.K, cfg.H_init),
    ])
    pos_jammer = rng.uniform(0, cfg.area_size, 2)
    return compute_all_channels(pos_su, pos_du, pos_rbs, pos_uav, pos_jammer, cfg, rng), pos_uav


# ── 1. SUActor output shapes ───────────────────────────────────────────────────

def test_su_actor_output_shapes(su_actor, cfg):
    B = 4
    obs = torch.randn(B, cfg.su_obs_dim)
    alpha, mode_logits = su_actor(obs)

    assert alpha.shape == (B, 1), f"alpha shape {alpha.shape} expected ({B}, 1)"
    assert mode_logits.shape == (B, 3), f"mode_logits shape {mode_logits.shape}"

    # alpha must be in [0, 1]
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0), \
        "alpha must be in [0, 1]"

    # mode_probs from get_action must sum to 1
    _, mode_probs, mode_idx = su_actor.get_action(obs, tau=1.0, hard=False)
    sums = mode_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
        f"mode_probs do not sum to 1: {sums}"
    assert mode_idx.shape == (B,)


# ── 2. UAVActor output range ───────────────────────────────────────────────────

def test_uav_actor_output_range(uav_actor, cfg):
    B = 8
    obs = torch.randn(B, cfg.uav_obs_dim)
    action = uav_actor(obs)

    assert action.shape == (B, 3), f"action shape {action.shape}"
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0), \
        "UAV action must be in [-1, 1]"


# ── 3. Critic output shape (B, 1) ─────────────────────────────────────────────

def test_critic_output_shape(critic, cfg):
    B = 4
    su_obs_list = [torch.randn(B, cfg.su_obs_dim) for _ in range(cfg.N)]
    uav_obs_list = [torch.randn(B, cfg.uav_obs_dim) for _ in range(cfg.K)]
    su_acts = [torch.randn(B, cfg.su_action_dim) for _ in range(cfg.N)]
    uav_acts = [torch.randn(B, cfg.uav_action_dim) for _ in range(cfg.K)]
    channel_gains = torch.rand(B, cfg.K, cfg.N) * 1e-4 + 1e-9  # (B, K, N)

    q1, q2 = critic(su_obs_list, uav_obs_list, su_acts, uav_acts, channel_gains)

    assert q1.shape == (B, 1), f"q1 shape {q1.shape} expected ({B}, 1)"
    assert q2.shape == (B, 1), f"q2 shape {q2.shape} expected ({B}, 1)"


# ── 4. Q1 ≠ Q2 (double heads are independent) ────────────────────────────────

def test_critic_double_q_differ(critic, cfg):
    B = 2
    su_obs_list = [torch.randn(B, cfg.su_obs_dim) for _ in range(cfg.N)]
    uav_obs_list = [torch.randn(B, cfg.uav_obs_dim) for _ in range(cfg.K)]
    su_acts = [torch.randn(B, cfg.su_action_dim) for _ in range(cfg.N)]
    uav_acts = [torch.randn(B, cfg.uav_action_dim) for _ in range(cfg.K)]
    channel_gains = torch.rand(B, cfg.K, cfg.N) * 1e-4 + 1e-9

    q1, q2 = critic(su_obs_list, uav_obs_list, su_acts, uav_acts, channel_gains)

    assert not torch.allclose(q1, q2), \
        "Q1 and Q2 should differ (independent twin heads)"


# ── 5. Gumbel-Softmax hard mode gives one-hot ─────────────────────────────────

def test_gumbel_softmax_hard(su_actor, cfg):
    B = 16
    obs = torch.randn(B, cfg.su_obs_dim)
    _, mode_probs, _ = su_actor.get_action(obs, tau=1.0, hard=True)

    # Hard one-hot: each row has exactly one 1 and rest 0
    assert torch.allclose(
        mode_probs.sum(dim=-1), torch.ones(B), atol=1e-5
    ), "Hard Gumbel-Softmax rows must sum to 1"
    assert torch.all((mode_probs == 0) | (mode_probs == 1)), \
        "Hard Gumbel-Softmax must produce one-hot vectors"


# ── 6. SUExpertPolicy: alpha in [0,1], mode in {0,1,2} ───────────────────────

def test_expert_su_valid_action(cfg, dummy_channels):
    channels, pos_uav = dummy_channels
    expert = SUExpertPolicy(cfg)
    uav_assignment = np.zeros(cfg.N, dtype=int)  # all SUs served by UAV 0

    for i in range(cfg.N):
        alpha, mode = expert.select_action(i, {
            "g_js": float(channels["g_JS"][i]),
            "g_sd": float(channels["g_SD"][i]),
            "g_jd": float(channels["g_JD"][i]),
            "g_sr": float(channels["g_SR"][i]),
            "g_jr": float(channels["g_JR"]),
            "g_rd": float(channels["g_RD"][i]),
        }, pos_uav, uav_assignment)

        assert 0.0 <= alpha <= 1.0, f"SUExpert alpha={alpha} not in [0,1]"
        assert mode in (0, 1, 2), f"SUExpert mode={mode} not in {{0,1,2}}"


# ── 7. UAVExpertPolicy: delta in [-1,1] ───────────────────────────────────────

def test_expert_uav_valid_action(cfg, dummy_channels):
    channels, pos_uav = dummy_channels
    expert = UAVExpertPolicy(cfg)

    for k in range(cfg.K):
        su_cluster = list(range(k, cfg.N, cfg.K))
        per_su_channels = {
            f"g_jd_{i}": float(channels["g_JD"][i]) for i in range(cfg.N)
        }
        per_su_channels.update({
            f"g_js_{i}": float(channels["g_JS"][i]) for i in range(cfg.N)
        })
        su_pos_3d = np.column_stack([
            np.random.uniform(0, cfg.area_size, (cfg.N, 2)),
            np.zeros(cfg.N),
        ])

        delta = expert.select_action(
            k, pos_uav, su_cluster, per_su_channels,
            np.full(cfg.N, 0.5),
            su_pos_3d,
        )
        assert delta.shape == (3,), f"UAVExpert delta shape {delta.shape}"
        assert np.all(np.abs(delta) <= 1.0 + 1e-6), \
            f"UAVExpert delta={delta} not in [-1,1]"


# ── 8. Actor gradient flow ────────────────────────────────────────────────────

def test_actor_gradient_flow(su_actor, cfg):
    obs = torch.randn(4, cfg.su_obs_dim, requires_grad=False)
    alpha, mode_logits = su_actor(obs)
    loss = alpha.mean() + mode_logits.mean()
    loss.backward()

    for name, param in su_actor.named_parameters():
        assert param.grad is not None, f"No gradient for param: {name}"
        assert not torch.any(torch.isnan(param.grad)), \
            f"NaN gradient for param: {name}"
