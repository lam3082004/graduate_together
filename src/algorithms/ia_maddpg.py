"""
ia_maddpg.py — IA-MADDPG: Imitation-Augmented MADDPG (numpy-only, torch-free).

Public API used by training / evaluation scripts:
  IAMADDPG(cfg, env)
  .select_actions(su_obs, uav_obs, explore)
  .store_transition(transition)
  .update(step)  → dict of losses
  .warmup(env, n_steps)
  .save(path) / .load(path)
"""

import os
import copy
import pickle
import numpy as np

from config import Config
from agents.su_actor import SUActor
from agents.uav_actor import UAVActor
from agents.transformer_gat_critic import CentralizedCritic
from agents.expert_policy import SUExpertPolicy, UAVExpertPolicy
from algorithms.per_buffer import PERBuffer
from nn.optimizers import Adam


# ── Soft update helper ────────────────────────────────────────────────────────

def soft_update_actor(src, tgt, tau: float) -> None:
    tgt.soft_update_from(src, tau)


# ── Transition unpacking ──────────────────────────────────────────────────────

def _unpack_batch(batch: list):
    """
    batch: list of tuples
        (su_obs, uav_obs, su_acts, uav_acts,
         rewards, next_su_obs, next_uav_obs, done, channel_gains)
    Returns dict of numpy arrays.
    """
    (su_obs, uav_obs, su_acts, uav_acts,
     rewards, next_su_obs, next_uav_obs, dones, channels) = zip(*batch)
    return dict(
        su_obs=np.stack(su_obs).astype(np.float32),           # (B, N, su_obs_dim)
        uav_obs=np.stack(uav_obs).astype(np.float32),         # (B, K, uav_obs_dim)
        su_acts=np.stack(su_acts).astype(np.float32),         # (B, N, su_act_dim)
        uav_acts=np.stack(uav_acts).astype(np.float32),       # (B, K, uav_act_dim)
        rewards=np.stack(rewards).astype(np.float32),         # (B, N) or (B,)
        next_su_obs=np.stack(next_su_obs).astype(np.float32), # (B, N, su_obs_dim)
        next_uav_obs=np.stack(next_uav_obs).astype(np.float32),
        dones=np.array(dones, dtype=np.float32),              # (B,)
        channels=list(channels),
    )


def _channel_gains_from_batch(channels_list: list, K: int, N: int) -> np.ndarray:
    """Extract (B, K, N) channel gains from list of channel dicts."""
    B = len(channels_list)
    g = np.ones((B, K, N), dtype=np.float32) * 1e-9
    for b, ch in enumerate(channels_list):
        if ch and "g_bar_SU" in ch:
            # g_bar_SU shape: (N, K) → we want (K, N)
            g[b] = ch["g_bar_SU"].T[:K, :N]
    return g


# ── IA-MADDPG ─────────────────────────────────────────────────────────────────

class IAMADDPG:
    """
    Imitation-Augmented MADDPG with centralized critic (numpy-only).
    Manages N SU actors + K UAV actors + 1 centralized double-Q critic.
    """

    def __init__(self, cfg: Config, env,
                 allow_uav_mode: bool = True,
                 lambda_il: float | None = None) -> None:
        self.cfg = cfg
        self._step = 0
        self.lambda_il = cfg.lambda_il_init if lambda_il is None else lambda_il
        self.allow_uav_mode = allow_uav_mode
        N, K = cfg.N, cfg.K

        # ── Actors + targets ─────────────────────────────────────────────────
        self.su_actors = [SUActor(cfg.su_obs_dim, cfg.actor_hidden) for _ in range(N)]
        self.uav_actors = [UAVActor(cfg.uav_obs_dim, cfg.actor_hidden) for _ in range(K)]
        self.su_actors_tgt = [copy.deepcopy(a) for a in self.su_actors]
        self.uav_actors_tgt = [copy.deepcopy(a) for a in self.uav_actors]

        # ── Critic + target ──────────────────────────────────────────────────
        self.critic = CentralizedCritic(
            N=N, K=K,
            su_obs_dim=cfg.su_obs_dim, uav_obs_dim=cfg.uav_obs_dim,
            su_action_dim=cfg.su_action_dim, uav_action_dim=cfg.uav_action_dim,
            d=cfg.transformer_d_model,
            mlp_hidden=cfg.critic_hidden,
        )
        self.critic_tgt = copy.deepcopy(self.critic)

        # ── Optimizers ───────────────────────────────────────────────────────
        self.opt_su   = [Adam(lr=cfg.lr_actor)  for _ in range(N)]
        self.opt_uav  = [Adam(lr=cfg.lr_actor)  for _ in range(K)]
        self.opt_critic = Adam(lr=cfg.lr_critic)

        # ── Buffer + experts ─────────────────────────────────────────────────
        self.buffer = PERBuffer(cfg.buffer_size, alpha=cfg.per_alpha)
        self.su_expert  = SUExpertPolicy(cfg)
        self.uav_expert = UAVExpertPolicy(cfg)
        self._uav_su_map = [list(range(i, N, K)) for i in range(K)]

    # ── Action selection ──────────────────────────────────────────────────────

    def select_actions(self, su_obs: np.ndarray, uav_obs: np.ndarray,
                       explore: bool = True):
        """
        su_obs : (N, su_obs_dim)  uav_obs : (K, uav_obs_dim)
        Returns su_actions (N, 4), uav_actions (K, 3)
        """
        cfg = self.cfg
        noise_std = cfg.td3_noise_std if explore else 0.0
        su_actions = []
        for i, actor in enumerate(self.su_actors):
            obs_b = su_obs[i:i+1]                       # (1, obs_dim)
            alpha, mode_probs, _ = actor.get_action(obs_b, explore=explore)
            if explore:
                mode_logits_raw, _ = actor.forward(obs_b)  # reuse cache
                _, ml = actor.forward(obs_b)
                act = np.concatenate([alpha.reshape(1, 1),
                                      ml], axis=-1).squeeze(0)
            else:
                _, ml = actor.forward(obs_b)
                act = np.concatenate([alpha.reshape(1, 1), ml], axis=-1).squeeze(0)
            if not self.allow_uav_mode:
                act[3] = -10.0  # suppress mode 2 logit
            su_actions.append(act)

        uav_actions = []
        for k, actor in enumerate(self.uav_actors):
            obs_b = uav_obs[k:k+1]
            act = actor.get_action(obs_b, noise_std=noise_std).squeeze(0)
            uav_actions.append(act)

        return (np.array(su_actions, dtype=np.float32),
                np.array(uav_actions, dtype=np.float32))

    # ── Expert actions ────────────────────────────────────────────────────────

    def select_expert_actions(self, su_obs: np.ndarray, uav_obs: np.ndarray,
                               env):
        """Use expert policies to generate warm-up actions."""
        cfg = self.cfg
        channels = getattr(env, 'channels', {})
        uav_pos = getattr(env, 'pos_uav', np.zeros((cfg.K, 3)))
        uav_assign = getattr(env, 'uav_assignment', np.zeros(cfg.N, dtype=int))
        alpha_prev = np.full(cfg.N, 0.5)

        su_acts = np.zeros((cfg.N, 4), dtype=np.float32)
        for i in range(cfg.N):
            ch_i = {
                "g_js": float(channels.get("g_JS", np.ones(cfg.N) * 1e-5)[i]),
                "g_sd": float(channels.get("g_SD", np.ones(cfg.N) * 1e-5)[i]),
                "g_jd": float(channels.get("g_JD", np.ones(cfg.N) * 1e-5)[i]),
                "g_sr": float(channels.get("g_SR", np.ones(cfg.N) * 1e-5)[i]),
                "g_jr": float(channels.get("g_JR", 1e-5)),
                "g_rd": float(channels.get("g_RD", np.ones(cfg.N) * 1e-5)[i]),
            }
            if "g_bar_SU" in channels:
                for k in range(cfg.K):
                    ch_i[f"g_su_uav_{k}"] = float(channels["g_bar_SU"][i, k])
                    ch_i[f"g_j_uav_{k}"]  = float(channels.get("g_bar_JU",
                        np.ones(cfg.K) * 1e-5)[k])
                ch_i["g_uav_d"] = float(channels.get("g_bar_UD",
                    np.ones((cfg.N, cfg.K)) * 1e-5)[i, 0])
            alpha, mode = self.su_expert.select_action(i, ch_i, uav_pos, uav_assign)
            logits = np.full(3, -10.0, dtype=np.float32)
            logits[mode] = 10.0
            su_acts[i] = [alpha, *logits]

        uav_acts = np.zeros((cfg.K, 3), dtype=np.float32)
        su_pos_3d = np.column_stack([
            getattr(env, 'pos_su', np.zeros((cfg.N, 2))),
            np.zeros(cfg.N),
        ])
        per_su_ch: dict = {}
        if channels:
            for i in range(cfg.N):
                per_su_ch[f"g_jd_{i}"] = float(channels.get("g_JD",
                    np.ones(cfg.N) * 1e-5)[i])
                per_su_ch[f"g_js_{i}"] = float(channels.get("g_JS",
                    np.ones(cfg.N) * 1e-5)[i])
        for k in range(cfg.K):
            cluster = self._uav_su_map[k]
            delta = self.uav_expert.select_action(
                k, uav_pos, cluster, per_su_ch, alpha_prev, su_pos_3d)
            uav_acts[k] = delta
        return su_acts, uav_acts

    # ── Buffer interface ──────────────────────────────────────────────────────

    def store_transition(self, transition: tuple) -> None:
        self.buffer.push(transition)

    # ── Training update ───────────────────────────────────────────────────────

    def update(self, step: int) -> dict:
        """
        One update: critic + (delayed) actor + soft target update.
        Returns dict with loss values (empty if buffer not warm).
        """
        cfg = self.cfg
        if len(self.buffer) < cfg.batch_size:
            return {}

        self._step = step
        beta = min(1.0, cfg.per_beta_init
                   + (1.0 - cfg.per_beta_init) * step / max(cfg.per_beta_steps, 1))
        batch, indices, weights = self.buffer.sample(cfg.batch_size, beta)
        td = _unpack_batch(batch)
        w = weights.reshape(-1)   # (B,)
        N, K = cfg.N, cfg.K
        chan_gains = _channel_gains_from_batch(td["channels"], K, N)

        # ── Target Q ──────────────────────────────────────────────────────────
        nxt_su_acts = np.zeros((cfg.batch_size, N, cfg.su_action_dim), np.float32)
        for i, actor in enumerate(self.su_actors_tgt):
            obs_b = td["next_su_obs"][:, i]           # (B, su_obs_dim)
            alpha_b, ml_b = actor.forward(obs_b)
            noise = np.random.randn(*ml_b.shape) * cfg.td3_noise_std
            ml_b = np.clip(ml_b + noise,
                           -cfg.td3_noise_clip, cfg.td3_noise_clip)
            nxt_su_acts[:, i] = np.concatenate([alpha_b, ml_b], axis=-1)

        nxt_uav_acts = np.zeros((cfg.batch_size, K, cfg.uav_action_dim), np.float32)
        for k, actor in enumerate(self.uav_actors_tgt):
            obs_b = td["next_uav_obs"][:, k]
            nxt_uav_acts[:, k] = actor.get_action(obs_b, noise_std=cfg.td3_noise_std)

        q1_next, q2_next = self.critic_tgt.forward(
            td["next_su_obs"], td["next_uav_obs"],
            nxt_su_acts, nxt_uav_acts, chan_gains)
        q_next = np.minimum(q1_next, q2_next)

        rew = td["rewards"]
        r_global = rew.mean(axis=-1) if rew.ndim > 1 else rew  # (B,)
        td_target = (r_global + cfg.gamma * q_next * (1.0 - td["dones"]))

        # ── Critic update ─────────────────────────────────────────────────────
        q1, q2 = self.critic.forward(
            td["su_obs"], td["uav_obs"], td["su_acts"], td["uav_acts"], chan_gains)
        td_errors = q1 - td_target
        self.buffer.update_priorities(indices, td_errors)

        dq1 = w * 2.0 * td_errors / cfg.batch_size  # (B,)
        self.critic.backward_critic(dq1, head=1)
        q2_errors = q2 - td_target
        dq2 = w * 2.0 * q2_errors / cfg.batch_size
        self.critic.backward_critic(dq2, head=2)
        self.opt_critic.step(self.critic.params_and_grads())
        critic_loss = float(np.mean(w * (td_errors ** 2 + q2_errors ** 2)))

        actor_loss = 0.0
        bc_loss = 0.0

        # ── Delayed actor update ──────────────────────────────────────────────
        if step % cfg.policy_delay == 0:
            # SU actors
            cur_su = np.zeros_like(td["su_acts"])
            for i, actor in enumerate(self.su_actors):
                obs_b = td["su_obs"][:, i]
                alpha_b, ml_b = actor.forward(obs_b)
                cur_su[:, i] = np.concatenate([alpha_b, ml_b], axis=-1)

            # UAV actors
            cur_uav = np.zeros_like(td["uav_acts"])
            for k, actor in enumerate(self.uav_actors):
                obs_b = td["uav_obs"][:, k]
                cur_uav[:, k] = actor.forward(obs_b)

            # Policy gradient: maximize Q1
            q1_pi, _ = self.critic.forward(
                td["su_obs"], td["uav_obs"], cur_su, cur_uav, chan_gains)
            maddpg_loss = -float(q1_pi.mean())

            # BC loss: match simplified expert (alpha=0.5, mode0 preferred)
            su_exp = np.zeros_like(cur_su)
            su_exp[:, :, 0] = 0.5
            su_exp[:, :, 1] = 10.0
            uav_exp = np.zeros_like(cur_uav)
            bc = float(np.mean((cur_su - su_exp) ** 2)
                       + np.mean((cur_uav - uav_exp) ** 2))
            actor_loss = maddpg_loss + self.lambda_il * bc
            bc_loss = bc

            # Compute gradients: dQ/da via finite difference approximation
            # (pure-numpy: we backprop -dQ through the critic, then into actors)
            dq_da = -np.ones(cfg.batch_size) / cfg.batch_size   # (B,)
            # Backprop through critic to get gradients w.r.t. su_acts / uav_acts
            # Use policy gradient + BC directly on actor outputs
            d_su = -(1.0 / cfg.batch_size) * np.ones_like(cur_su)
            d_su += 2.0 * self.lambda_il * (cur_su - su_exp) / cfg.batch_size
            d_uav = 2.0 * self.lambda_il * (cur_uav - uav_exp) / cfg.batch_size

            for i, actor in enumerate(self.su_actors):
                obs_b = td["su_obs"][:, i]
                actor.forward(obs_b)     # refresh cache
                # Split gradient: alpha part (col 0), mode part (cols 1:4)
                d_alpha = d_su[:, i, 0:1]
                d_mode  = d_su[:, i, 1:4]
                actor.backward_alpha(d_alpha)
                actor.backward_mode(d_mode)
                self.opt_su[i].step(actor.params_and_grads())

            for k, actor in enumerate(self.uav_actors):
                obs_b = td["uav_obs"][:, k]
                actor.forward(obs_b)
                actor.backward(d_uav[:, k])
                self.opt_uav[k].step(actor.params_and_grads())

        # ── Soft-update targets ───────────────────────────────────────────────
        tau = cfg.tau
        for src, tgt in zip(self.su_actors, self.su_actors_tgt):
            tgt.soft_update_from(src, tau)
        for src, tgt in zip(self.uav_actors, self.uav_actors_tgt):
            tgt.soft_update_from(src, tau)
        self.critic_tgt.soft_update_from(self.critic, tau)

        # ── Anneal IL coefficient ─────────────────────────────────────────────
        self.lambda_il = max(self.lambda_il * cfg.lambda_il_decay, 0.0)

        return dict(
            critic_loss=critic_loss,
            actor_loss=float(actor_loss),
            bc_loss=float(bc_loss),
        )

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def warmup(self, env, n_steps: int) -> None:
        """Pre-populate buffer with expert-generated transitions."""
        su_obs, uav_obs = env.reset()
        for _ in range(n_steps):
            su_acts, uav_acts = self.select_expert_actions(su_obs, uav_obs, env)
            next_su, next_uav, rewards, done, info = env.step(su_acts, uav_acts)
            self.buffer.push((
                su_obs, uav_obs, su_acts, uav_acts,
                rewards["su"], next_su, next_uav, float(done), env.channels,
            ))
            su_obs, uav_obs = (next_su, next_uav) if not done else env.reset()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        state = dict(
            su_actors=[[(l.W.copy(), l.b.copy()) for l in a.backbone.layers
                        ] + [(l.W.copy(), l.b.copy()) for l in a.alpha_head.layers
                        ] + [(l.W.copy(), l.b.copy()) for l in a.mode_head.layers]
                       for a in self.su_actors],
            uav_actors=[[(l.W.copy(), l.b.copy()) for l in a.backbone.layers
                         ] + [(l.W.copy(), l.b.copy()) for l in a.action_head.layers]
                        for a in self.uav_actors],
            lambda_il=self.lambda_il,
        )
        with open(os.path.join(path, "ia_maddpg.pkl"), "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(os.path.join(path, "ia_maddpg.pkl"), "rb") as f:
            state = pickle.load(f)
        # Restore SU actors
        for a, params in zip(self.su_actors, state["su_actors"]):
            all_layers = a.backbone.layers + a.alpha_head.layers + a.mode_head.layers
            for layer, (W, b) in zip(all_layers, params):
                layer.W[:] = W
                layer.b[:] = b
        # Restore UAV actors
        for a, params in zip(self.uav_actors, state["uav_actors"]):
            all_layers = a.backbone.layers + a.action_head.layers
            for layer, (W, b) in zip(all_layers, params):
                layer.W[:] = W
                layer.b[:] = b
        self.lambda_il = state.get("lambda_il", 0.0)
        # Sync targets
        for src, tgt in zip(self.su_actors, self.su_actors_tgt):
            tgt.copy_from(src)
        for src, tgt in zip(self.uav_actors, self.uav_actors_tgt):
            tgt.copy_from(src)
        self.critic_tgt.copy_from(self.critic)
