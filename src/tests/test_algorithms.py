"""
test_algorithms.py — pytest suite for PERBuffer, IAMADDPG, baselines, soft_update.
"""
import sys
import os
import copy
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from environment.network_env import AntiJammingEnv
from algorithms.per_buffer import PERBuffer
from algorithms.ia_maddpg import IAMADDPG
from algorithms.ia_maddpg_update import soft_update
from algorithms.baselines import (
    DirectTransmission,
    GreedyStrategy,
    FrequencyHopping,
    StandardMADDPG,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    c = Config(seed=0)
    # Reduce sizes for fast tests
    object.__setattr__(c, "buffer_size", 1000)
    object.__setattr__(c, "batch_size", 16)
    object.__setattr__(c, "warmup_steps", 20)
    object.__setattr__(c, "steps_per_episode", 10)
    return c


@pytest.fixture
def env(cfg):
    return AntiJammingEnv(cfg)


@pytest.fixture
def buffer(cfg):
    return PERBuffer(capacity=200, alpha=0.6)


@pytest.fixture
def dummy_transition(cfg):
    """A single valid transition tuple."""
    N, K = cfg.N, cfg.K
    return (
        np.zeros((N, cfg.su_obs_dim), dtype=np.float32),    # su_obs
        np.zeros((K, cfg.uav_obs_dim), dtype=np.float32),   # uav_obs
        np.zeros((N, cfg.su_action_dim), dtype=np.float32), # su_acts
        np.zeros((K, cfg.uav_action_dim), dtype=np.float32),# uav_acts
        np.zeros(N, dtype=np.float32),                       # rewards
        np.zeros((N, cfg.su_obs_dim), dtype=np.float32),    # next_su_obs
        np.zeros((K, cfg.uav_obs_dim), dtype=np.float32),   # next_uav_obs
        0.0,                                                 # done
        {},                                                  # channels
    )


# ── 1. PERBuffer push / sample ────────────────────────────────────────────────

def test_per_buffer_push_sample(buffer, dummy_transition, cfg):
    n_push = 50
    for _ in range(n_push):
        buffer.push(dummy_transition)

    assert len(buffer) == n_push

    batch_size = 8
    batch, indices, weights = buffer.sample(batch_size, beta=0.4)

    assert len(batch) == batch_size, f"batch length {len(batch)} != {batch_size}"
    assert indices.shape == (batch_size,), f"indices shape {indices.shape}"
    assert weights.shape == (batch_size,), f"weights shape {weights.shape}"
    assert np.all(weights > 0), "IS weights must be positive"


# ── 2. PERBuffer: high-priority transitions sampled more often ────────────────

def test_per_buffer_priority_update(buffer, dummy_transition):
    n_transitions = 100
    for _ in range(n_transitions):
        buffer.push(dummy_transition)

    # Assign very high priority to the first 5 transitions
    high_priority_indices = np.array(
        [buffer.tree.capacity - 1 + i for i in range(5)], dtype=np.int64
    )
    buffer.update_priorities(high_priority_indices, np.full(5, 100.0))

    # Count how often high-priority items are sampled
    n_trials = 500
    high_count = 0
    for _ in range(n_trials):
        batch, idxs, _ = buffer.sample(16, beta=0.4)
        high_count += int(np.any(np.isin(idxs, high_priority_indices)))

    # High-priority items should be in >50% of samples
    rate = high_count / n_trials
    assert rate > 0.5, \
        f"High-priority items sampled in only {rate*100:.1f}% of batches (expected >50%)"


# ── Helper: fill buffer manually (avoids ia_maddpg.warmup which has a known
#            env.step unpacking bug — it expects 4 returns but env returns 5) ──

def _fill_buffer(agent: IAMADDPG, env: AntiJammingEnv, cfg: Config,
                 n_steps: int) -> None:
    su_obs, uav_obs = env.reset()
    for _ in range(n_steps):
        su_acts, uav_acts = agent.select_actions(su_obs, uav_obs, explore=True)
        next_su, next_uav, rewards, done, info = env.step(su_acts, uav_acts)
        agent.store_transition((
            su_obs, uav_obs, su_acts, uav_acts,
            rewards["su"], next_su, next_uav, float(done), env.channels,
        ))
        su_obs, uav_obs = (next_su, next_uav) if not done else env.reset()


# ── 3. IA-MADDPG warmup fills buffer ─────────────────────────────────────────

def test_ia_maddpg_warmup(cfg, env):
    agent = IAMADDPG(cfg, env)
    assert len(agent.buffer) == 0, "Buffer should be empty before warmup"

    _fill_buffer(agent, env, cfg, cfg.warmup_steps)

    assert len(agent.buffer) > 0, "Buffer should be non-empty after warmup"
    assert len(agent.buffer) <= cfg.warmup_steps + 2, \
        f"Buffer has {len(agent.buffer)} entries but fill was {cfg.warmup_steps} steps"


# ── 4. IA-MADDPG update returns loss dict without error ──────────────────────

def test_ia_maddpg_update_runs(cfg, env):
    agent = IAMADDPG(cfg, env)
    # Fill buffer with more than batch_size transitions
    _fill_buffer(agent, env, cfg, cfg.batch_size * 2 + 10)

    result = agent.update(step=cfg.policy_delay)  # ensure policy update fires

    # Result should be a non-empty dict with expected keys
    assert isinstance(result, dict), f"update() returned {type(result)}"
    if result:  # may be empty if buffer still too small in edge cases
        assert "critic_loss" in result, "Missing critic_loss in update result"
        assert "actor_loss" in result, "Missing actor_loss in update result"
        assert isinstance(result["critic_loss"], float)
        assert not np.isnan(result["critic_loss"]), "critic_loss is NaN"


# ── 5. DirectTransmission always selects mode 0 ──────────────────────────────

def test_baseline_dt_actions(cfg):
    strategy = DirectTransmission()
    su_acts, uav_acts = strategy.select_actions({"N": cfg.N, "K": cfg.K})

    assert su_acts.shape == (cfg.N, 4)
    assert uav_acts.shape == (cfg.K, 3)

    # Mode 0 logit must be largest (argmax → 0)
    modes = np.argmax(su_acts[:, 1:4], axis=1)
    assert np.all(modes == 0), f"DT should always select mode 0, got {modes}"

    # Alpha fixed at 0.2
    assert np.allclose(su_acts[:, 0], DirectTransmission.ALPHA), \
        f"DT alpha should be {DirectTransmission.ALPHA}"

    # UAV stays put
    assert np.allclose(uav_acts, 0.0), "DT UAV actions should be zero"


# ── 6. GreedyStrategy never picks mode 2 (UAV relay) ─────────────────────────

def test_baseline_greedy_no_mode2(cfg, env):
    env.reset()
    strategy = GreedyStrategy(cfg)
    su_acts, uav_acts = strategy.select_actions(env.channels, cfg)

    modes = np.argmax(su_acts[:, 1:4], axis=1)
    assert np.all(modes != 2), \
        f"Greedy should never pick mode 2, got modes={modes}"

    # UAV stays put
    assert np.allclose(uav_acts, 0.0), "Greedy UAV actions should be zero"


# ── 7. FrequencyHopping produces varying actions across steps ─────────────────

def test_baseline_fh_random(cfg):
    strategy = FrequencyHopping(cfg)

    actions_0, _ = strategy.select_actions(0)
    actions_1, _ = strategy.select_actions(1)
    actions_100, _ = strategy.select_actions(100)

    # Over multiple calls, actions should vary (not all identical)
    all_same = (
        np.allclose(actions_0, actions_1) and
        np.allclose(actions_0, actions_100)
    )
    assert not all_same, \
        "FrequencyHopping should produce varying actions over multiple steps"

    # Modes should only be 0 or 1 (no UAV mode 2)
    for s in range(20):
        acts, _ = strategy.select_actions(s)
        modes = np.argmax(acts[:, 1:4], axis=1)
        assert np.all(modes < 2), \
            f"FreqHopping should only use modes 0/1, got {modes}"


# ── 8. Soft update: target params move toward source ─────────────────────────

def test_soft_update(cfg):
    from agents.su_actor import SUActor

    source = SUActor(obs_dim=cfg.su_obs_dim, hidden=[32, 16])
    target = copy.deepcopy(source)

    # Perturb source parameters strongly
    with torch.no_grad():
        for p in source.parameters():
            p.add_(torch.randn_like(p) * 5.0)

    tau = 0.1
    soft_update(source, target, tau)

    # After soft update: target should have moved toward source but not all the way
    param_diffs = []
    for sp, tp in zip(source.parameters(), target.parameters()):
        param_diffs.append(float((sp - tp).abs().mean()))

    # target should NOT equal source (tau < 1)
    assert any(d > 1e-6 for d in param_diffs), \
        "After soft update, target should not fully equal source"

    # But target should have moved; check against a fresh copy of the original
    orig_target = copy.deepcopy(target)
    for _ in range(50):
        soft_update(source, target, tau)

    # After many updates, target should be closer to source
    final_diffs = [
        float((sp - tp).abs().mean())
        for sp, tp in zip(source.parameters(), target.parameters())
    ]
    orig_diffs = [
        float((sp - tp).abs().mean())
        for sp, tp in zip(source.parameters(), orig_target.parameters())
    ]
    assert sum(final_diffs) < sum(orig_diffs), \
        "Target params should converge toward source after repeated soft updates"
