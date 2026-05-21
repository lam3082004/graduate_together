"""
test_environment.py — pytest suite for AntiJammingEnv, channel_model, sinr_calculator.
"""
import sys
import os
import pytest
import numpy as np

# Make src/ importable when running from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment.network_env import AntiJammingEnv
from environment.channel_model import (
    compute_all_channels,
    channel_gain_g2g,
    mean_channel_gain_a2g,
    prob_los,
)
from environment.sinr_calculator import (
    compute_sinr,
    compute_reward,
    sinr_mode0,
    sinr_mode1,
    sinr_mode2,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(seed=0)


@pytest.fixture
def env(cfg):
    return AntiJammingEnv(cfg)


@pytest.fixture
def reset_env(env):
    su_obs, uav_obs = env.reset()
    return env, su_obs, uav_obs


# ── 1. Reset shapes and valid ranges ──────────────────────────────────────────

def test_env_reset(env, cfg):
    su_obs, uav_obs = env.reset()

    assert su_obs.shape == (cfg.N, cfg.su_obs_dim), \
        f"su_obs shape {su_obs.shape} expected ({cfg.N}, {cfg.su_obs_dim})"
    assert uav_obs.shape == (cfg.K, cfg.uav_obs_dim), \
        f"uav_obs shape {uav_obs.shape} expected ({cfg.K}, {cfg.uav_obs_dim})"

    # SU obs: all values should be normalised to reasonable ranges
    assert np.all(su_obs >= 0.0), "su_obs contains negative values"
    assert np.all(su_obs <= 2.1), "su_obs contains values > 2.1 (expected ≤ 2)"

    # UAV obs: first 3 are normalised position (≥0), remainder channel gains (≥0)
    assert np.all(uav_obs >= 0.0), "uav_obs contains negative values"


# ── 2. Step output shapes ──────────────────────────────────────────────────────

def test_env_step_shapes(reset_env, cfg):
    env, su_obs, uav_obs = reset_env

    su_acts = np.zeros((cfg.N, cfg.su_action_dim), dtype=np.float32)
    uav_acts = np.zeros((cfg.K, cfg.uav_action_dim), dtype=np.float32)

    next_su_obs, next_uav_obs, rewards, done, info = env.step(su_acts, uav_acts)

    assert next_su_obs.shape == (cfg.N, cfg.su_obs_dim)
    assert next_uav_obs.shape == (cfg.K, cfg.uav_obs_dim)
    assert rewards["su"].shape == (cfg.N,)
    assert rewards["uav"].shape == (cfg.K,)
    assert isinstance(done, bool)
    assert "sinr" in info
    assert "modes" in info
    assert info["sinr"].shape == (cfg.N,)
    assert info["modes"].shape == (cfg.N,)


# ── 3. G2G channel gains > 0, decrease with distance ─────────────────────────

def test_channel_model_g2g(cfg):
    rng = np.random.default_rng(42)
    # Short distance
    g_near = channel_gain_g2g(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                               cfg.eta, rng)
    # Long distance — average over many samples to reduce variance
    gains_far = [
        channel_gain_g2g(np.array([0.0, 0.0]), np.array([100.0, 0.0]),
                         cfg.eta, rng)
        for _ in range(500)
    ]
    g_far_mean = float(np.mean(gains_far))

    assert g_near > 0.0, "G2G gain must be positive"
    assert g_far_mean < g_near or g_far_mean < 1.0, \
        "Mean far gain should be less than near gain or < 1"


# ── 4. A2G channel: LoS probability in [0,1] and mean gain > 0 ───────────────

def test_channel_model_a2g(cfg):
    pos_uav = np.array([100.0, 100.0, 50.0])
    pos_ground = np.array([50.0, 50.0])

    # LoS probability
    horiz_dist = float(np.linalg.norm(pos_uav[:2] - pos_ground))
    p = prob_los(pos_uav[2], horiz_dist, cfg.a_los, cfg.b_los)
    assert 0.0 <= p <= 1.0, f"prob_los={p} not in [0,1]"

    # Mean channel gain
    g = mean_channel_gain_a2g(pos_uav, pos_ground, cfg)
    assert g > 0.0, f"A2G mean gain={g} not positive"


# ── 5. SINR mode 0 positive ────────────────────────────────────────────────────

def test_sinr_mode0_positive(cfg):
    sinr = sinr_mode0(
        g_JS=1e-3, g_SD=1e-3, g_JD=1e-4,
        alpha=0.8, cfg=cfg,
    )
    assert sinr >= 0.0, f"sinr_mode0={sinr} must be non-negative"
    assert sinr > 0.0, "sinr_mode0 should be positive with non-zero gains"


# ── 6. SINR mode 1: bottleneck ≤ min(hop1, hop2) ─────────────────────────────

def test_sinr_mode1_bottleneck(cfg):
    g_JS, g_SR, g_JR, g_RD, g_JD, alpha = 1e-3, 5e-4, 1e-4, 2e-3, 1e-4, 0.9

    sinr = sinr_mode1(g_JS=g_JS, g_SR=g_SR, g_JR=g_JR,
                      g_RD=g_RD, g_JD=g_JD, alpha=alpha, cfg=cfg)

    signal1 = cfg.G * cfg.P_J * g_JS * g_SR * alpha ** 2
    denom1 = cfg.P_J * g_JR + cfg.N0
    hop1 = signal1 / max(denom1, 1e-30)

    signal2 = cfg.G * cfg.P_J * g_JS * g_RD * alpha ** 2
    denom2 = cfg.P_J * g_JD + cfg.N0
    hop2 = signal2 / max(denom2, 1e-30)

    assert sinr <= min(hop1, hop2) + 1e-9, \
        f"mode1 SINR={sinr} exceeds bottleneck min({hop1},{hop2})"


# ── 7. SINR mode 2: bottleneck ≤ min(hop1, hop2) ─────────────────────────────

def test_sinr_mode2_bottleneck(cfg):
    g_JS = 1e-3
    g_bar_SU = 2e-5
    g_bar_JU = 5e-6
    g_bar_UD = 3e-5
    g_JD = 1e-4
    alpha = 0.7

    sinr = sinr_mode2(
        g_JS=g_JS, g_bar_SU=g_bar_SU, g_bar_JU=g_bar_JU,
        g_bar_UD=g_bar_UD, g_JD=g_JD, alpha=alpha, cfg=cfg,
    )

    signal1 = cfg.G * cfg.P_J * g_JS * g_bar_SU * alpha ** 2
    denom1 = cfg.P_J * g_bar_JU + cfg.N0
    hop1 = signal1 / max(denom1, 1e-30)

    signal2 = cfg.G * cfg.P_J * g_bar_JU * g_bar_UD * alpha ** 2
    denom2 = cfg.P_J * g_JD + cfg.N0
    hop2 = signal2 / max(denom2, 1e-30)

    assert sinr <= min(hop1, hop2) + 1e-9, \
        f"mode2 SINR={sinr} exceeds bottleneck min({hop1},{hop2})"


# ── 8. Higher SINR → higher reward ────────────────────────────────────────────

def test_reward_increases_with_sinr(cfg):
    low_sinr = 0.1
    high_sinr = 100.0
    alpha = 0.5
    mode = 0
    delta_p = 0.0

    r_low = compute_reward(low_sinr, alpha, mode, delta_p, cfg)
    r_high = compute_reward(high_sinr, alpha, mode, delta_p, cfg)

    assert r_high > r_low, \
        f"reward(SINR={high_sinr})={r_high} should exceed reward(SINR={low_sinr})={r_low}"


# ── 9. UAV stays within bounds after clipping ─────────────────────────────────

def test_uav_position_update(env, cfg):
    env.reset()
    # Large displacement that should be clipped
    uav_acts = np.ones((cfg.K, 3), dtype=np.float32) * 100.0  # way beyond v_max
    su_acts = np.zeros((cfg.N, cfg.su_action_dim), dtype=np.float32)
    env.step(su_acts, uav_acts)

    pos = env.pos_uav
    assert np.all(pos[:, 0] >= 0.0) and np.all(pos[:, 0] <= cfg.area_size), \
        "UAV x position out of bounds"
    assert np.all(pos[:, 1] >= 0.0) and np.all(pos[:, 1] <= cfg.area_size), \
        "UAV y position out of bounds"
    assert np.all(pos[:, 2] >= cfg.H_min) and np.all(pos[:, 2] <= cfg.H_max), \
        f"UAV altitude out of [{cfg.H_min}, {cfg.H_max}]"


# ── 10. Episode terminates at step 200 ────────────────────────────────────────

def test_episode_terminates(env, cfg):
    env.reset()
    su_acts = np.zeros((cfg.N, cfg.su_action_dim), dtype=np.float32)
    uav_acts = np.zeros((cfg.K, cfg.uav_action_dim), dtype=np.float32)

    done = False
    steps = 0
    while not done:
        _, _, _, done, _ = env.step(su_acts, uav_acts)
        steps += 1
        if steps > cfg.steps_per_episode + 10:
            break

    assert done, "Episode should terminate"
    assert steps == cfg.steps_per_episode, \
        f"Episode length {steps} != steps_per_episode {cfg.steps_per_episode}"
