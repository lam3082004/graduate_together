"""
test_agents.py — pytest suite for SUActor, UAVActor, CentralizedCritic,
expert policies, gradient flow. Numpy-only, no torch.
"""
import sys
import os
import copy
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from agents.su_actor import SUActor
from agents.uav_actor import UAVActor
from agents.transformer_gat_critic import CentralizedCritic
from agents.expert_policy import SUExpertPolicy, UAVExpertPolicy
from environment.channel_model import compute_all_channels
from nn.optimizers import Adam


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(seed=0)

@pytest.fixture
def su_actor(cfg):
    return SUActor(obs_dim=cfg.su_obs_dim, hidden=[64, 32])

@pytest.fixture
def uav_actor(cfg):
    return UAVActor(obs_dim=cfg.uav_obs_dim, hidden=[64, 32])

@pytest.fixture
def critic(cfg):
    return CentralizedCritic(
        N=cfg.N, K=cfg.K,
        su_obs_dim=cfg.su_obs_dim, uav_obs_dim=cfg.uav_obs_dim,
        su_action_dim=cfg.su_action_dim, uav_action_dim=cfg.uav_action_dim,
        d=64, mlp_hidden=[128, 64],
    )

@pytest.fixture
def dummy_channels(cfg):
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


# ── 1. SUActor output shapes and range ────────────────────────────────────────

def test_su_actor_output_shapes(su_actor, cfg):
    B = 4
    obs = np.random.randn(B, cfg.su_obs_dim).astype(np.float32)
    alpha, mode_logits = su_actor.forward(obs)

    assert alpha.shape == (B, 1), f"alpha shape {alpha.shape}"
    assert mode_logits.shape == (B, 3), f"mode_logits shape {mode_logits.shape}"
    assert np.all(alpha >= 0.0) and np.all(alpha <= 1.0), "alpha out of [0,1]"

    alpha_v, mode_probs, mode_idx = su_actor.get_action(obs)
    assert np.allclose(mode_probs.sum(axis=-1), 1.0, atol=1e-5), "mode_probs not summing to 1"
    assert mode_idx.shape == (B,)
    assert np.all((mode_idx >= 0) & (mode_idx <= 2))


# ── 2. UAVActor output range ───────────────────────────────────────────────────

def test_uav_actor_output_range(uav_actor, cfg):
    B = 8
    obs = np.random.randn(B, cfg.uav_obs_dim).astype(np.float32)
    action = uav_actor.forward(obs)

    assert action.shape == (B, 3), f"action shape {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), "UAV action out of [-1,1]"


# ── 3. Critic output shape ────────────────────────────────────────────────────

def test_critic_output_shape(critic, cfg):
    B = 4
    su_obs  = np.random.randn(B, cfg.N, cfg.su_obs_dim).astype(np.float32)
    uav_obs = np.random.randn(B, cfg.K, cfg.uav_obs_dim).astype(np.float32)
    su_acts = np.random.randn(B, cfg.N, cfg.su_action_dim).astype(np.float32)
    uav_acts= np.random.randn(B, cfg.K, cfg.uav_action_dim).astype(np.float32)
    chan    = np.abs(np.random.randn(B, cfg.K, cfg.N)).astype(np.float32) * 1e-4 + 1e-9

    q1, q2 = critic.forward(su_obs, uav_obs, su_acts, uav_acts, chan)

    assert q1.shape == (B,), f"q1 shape {q1.shape} expected ({B},)"
    assert q2.shape == (B,), f"q2 shape {q2.shape}"


# ── 4. Q1 ≠ Q2 ────────────────────────────────────────────────────────────────

def test_critic_double_q_differ(critic, cfg):
    B = 4
    su_obs  = np.random.randn(B, cfg.N, cfg.su_obs_dim).astype(np.float32)
    uav_obs = np.random.randn(B, cfg.K, cfg.uav_obs_dim).astype(np.float32)
    su_acts = np.random.randn(B, cfg.N, cfg.su_action_dim).astype(np.float32)
    uav_acts= np.random.randn(B, cfg.K, cfg.uav_action_dim).astype(np.float32)
    chan    = np.abs(np.random.randn(B, cfg.K, cfg.N)).astype(np.float32) * 1e-4 + 1e-9

    q1, q2 = critic.forward(su_obs, uav_obs, su_acts, uav_acts, chan)
    assert not np.allclose(q1, q2), "Q1 and Q2 should differ"


# ── 5. Gumbel exploration changes mode probs ─────────────────────────────────

def test_gumbel_explore(su_actor, cfg):
    B = 16
    obs = np.random.randn(B, cfg.su_obs_dim).astype(np.float32)
    _, probs_det, _ = su_actor.get_action(obs, explore=False)
    _, probs_exp, _ = su_actor.get_action(obs, explore=True)
    assert not np.allclose(probs_det, probs_exp), "Exploration should change mode probs"


# ── 6. SUExpertPolicy valid action ────────────────────────────────────────────

def test_expert_su_valid_action(cfg, dummy_channels):
    channels, pos_uav = dummy_channels
    expert = SUExpertPolicy(cfg)
    uav_assignment = np.zeros(cfg.N, dtype=int)

    for i in range(cfg.N):
        ch_i = {
            "g_js": float(channels["g_JS"][i]),
            "g_sd": float(channels["g_SD"][i]),
            "g_jd": float(channels["g_JD"][i]),
            "g_sr": float(channels["g_SR"][i]),
            "g_jr": float(channels["g_JR"]),
            "g_rd": float(channels["g_RD"][i]),
        }
        alpha, mode = expert.select_action(i, ch_i, pos_uav, uav_assignment)
        assert 0.0 <= alpha <= 1.0
        assert mode in (0, 1, 2)


# ── 7. UAVExpertPolicy valid action ───────────────────────────────────────────

def test_expert_uav_valid_action(cfg, dummy_channels):
    channels, pos_uav = dummy_channels
    expert = UAVExpertPolicy(cfg)

    for k in range(cfg.K):
        su_cluster = list(range(k, cfg.N, cfg.K))
        per_su_ch = {f"g_jd_{i}": float(channels["g_JD"][i]) for i in range(cfg.N)}
        per_su_ch.update({f"g_js_{i}": float(channels["g_JS"][i]) for i in range(cfg.N)})
        su_pos_3d = np.column_stack([np.random.uniform(0, 200, (cfg.N, 2)), np.zeros(cfg.N)])
        delta = expert.select_action(k, pos_uav, su_cluster, per_su_ch,
                                     np.full(cfg.N, 0.5), su_pos_3d)
        assert delta.shape == (3,)
        assert np.all(np.abs(delta) <= 1.0 + 1e-6)


# ── 8. Actor params update after backward + Adam step ────────────────────────

def test_actor_params_update(su_actor, cfg):
    obs = np.random.randn(4, cfg.su_obs_dim).astype(np.float32)
    W_before = su_actor.backbone.layers[0].W.copy()

    su_actor.forward(obs)
    d_alpha = np.ones((4, 1), dtype=np.float32) * 0.01
    d_mode  = np.ones((4, 3), dtype=np.float32) * 0.01
    su_actor.backward_alpha(d_alpha)
    su_actor.backward_mode(d_mode)

    opt = Adam(lr=1e-3)
    opt.step(su_actor.params_and_grads())

    W_after = su_actor.backbone.layers[0].W
    assert not np.allclose(W_before, W_after), "Params should change after update"
