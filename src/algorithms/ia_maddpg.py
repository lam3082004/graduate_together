"""
ia_maddpg.py — IA-MADDPG facade: public API used by training scripts.

Delegates initialization to ia_maddpg_core and updates to ia_maddpg_update.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.algorithms.per_buffer import PERBuffer
from src.algorithms.ia_maddpg_core import (
    build_agents, SUExpert, UAVExpert,
    _sinr_mode0, _sinr_mode1, _sinr_mode2_best,
)
from src.algorithms.ia_maddpg_update import update_step, soft_update


class IAMADDPG:
    """
    Imitation-Augmented MADDPG with Transformer-GAT Centralized Critic.
    Manages N SU actors + K UAV actors + 1 centralized critic.
    """

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self._step = 0
        self.lambda_il = cfg.lambda_il_init

        (self.su_actors, self.uav_actors,
         self.su_actors_tgt, self.uav_actors_tgt,
         self.critic, self.critic_tgt,
         self.opt_su, self.opt_uav, self.opt_critic,
         self.device) = build_agents(cfg)

        self.buffer = PERBuffer(cfg.buffer_size, alpha=cfg.per_alpha)
        self.su_expert = SUExpert(cfg)
        self.uav_expert = UAVExpert(cfg)

        # Assign SU indices to UAVs round-robin
        self._uav_su_map = [list(range(i, cfg.N, cfg.K)) for i in range(cfg.K)]

    # ── Action selection ───────────────────────────────────────────────────────

    def select_actions(self, su_obs: np.ndarray, uav_obs: np.ndarray,
                       explore: bool = True):
        """
        su_obs : (N, su_obs_dim)  uav_obs : (K, uav_obs_dim)
        Returns su_actions (N,4), uav_actions (K,3)
        """
        cfg = self.cfg
        noise_std = cfg.td3_noise_std if explore else 0.0

        su_actions = []
        for i, actor in enumerate(self.su_actors):
            obs_t = torch.tensor(su_obs[i], dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            with torch.no_grad():
                alpha, mode_logits = actor(obs_t)
                if explore:
                    mode_logits = mode_logits + torch.randn_like(mode_logits) * noise_std
                act = torch.cat([alpha, mode_logits], dim=-1).squeeze(0)
            su_actions.append(act.cpu().numpy())

        uav_actions = []
        for k, actor in enumerate(self.uav_actors):
            obs_t = torch.tensor(uav_obs[k], dtype=torch.float32,
                                 device=self.device).unsqueeze(0)
            with torch.no_grad():
                act = actor.get_action(obs_t, noise_std=noise_std).squeeze(0)
            uav_actions.append(act.cpu().numpy())

        return np.array(su_actions, dtype=np.float32), \
               np.array(uav_actions, dtype=np.float32)

    def select_expert_actions(self, env):
        """Use expert policies to generate actions for warm-up transitions."""
        channels = env.channels if hasattr(env, 'channels') else {}
        uav_pos = env.positions_uav if hasattr(env, 'positions_uav') else \
                  np.zeros((self.cfg.K, 3))
        su_act = self.su_expert.act(channels, uav_pos)
        uav_act = self.uav_expert.act(uav_pos, channels, self._uav_su_map)
        return su_act, uav_act

    # ── Buffer interface ───────────────────────────────────────────────────────

    def store_transition(self, transition: tuple) -> None:
        self.buffer.push(transition)

    # ── Training update ────────────────────────────────────────────────────────

    def update(self, step: int) -> dict:
        """
        One update iteration: critic + (delayed) actor + soft target update.
        Anneals lambda_IL after each call.
        Returns dict with loss values (empty if buffer not warm).
        """
        self._step = step
        result = update_step(
            buffer=self.buffer,
            su_actors=self.su_actors,
            uav_actors=self.uav_actors,
            su_actors_tgt=self.su_actors_tgt,
            uav_actors_tgt=self.uav_actors_tgt,
            critic=self.critic,
            critic_tgt=self.critic_tgt,
            opt_su=self.opt_su,
            opt_uav=self.opt_uav,
            opt_critic=self.opt_critic,
            su_expert=self.su_expert,
            uav_expert=self.uav_expert,
            cfg=self.cfg,
            lambda_il=self.lambda_il,
            step=step,
            device=self.device,
        )

        if result:
            # Soft-update target networks
            tau = self.cfg.tau
            for src, tgt in zip(self.su_actors, self.su_actors_tgt):
                soft_update(src, tgt, tau)
            for src, tgt in zip(self.uav_actors, self.uav_actors_tgt):
                soft_update(src, tgt, tau)
            soft_update(self.critic, self.critic_tgt, tau)

            # Anneal IL coefficient
            self.lambda_il *= self.cfg.lambda_il_decay
            self.lambda_il = max(self.lambda_il, 0.0)

        return result

    # ── Warm-up pre-population ─────────────────────────────────────────────────

    def warmup(self, env, n_steps: int) -> None:
        """Pre-populate buffer with expert-generated transitions."""
        obs = env.reset()
        for _ in range(n_steps):
            su_obs, uav_obs = obs
            su_acts, uav_acts = self.select_expert_actions(env)
            next_obs, rewards, done, info = env.step(su_acts, uav_acts)
            channels = getattr(env, 'channels', {})
            self.store_transition((
                su_obs, uav_obs, su_acts, uav_acts,
                rewards, next_obs[0], next_obs[1], float(done), channels,
            ))
            obs = next_obs if not done else env.reset()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        state = dict(
            su_actors=[a.state_dict() for a in self.su_actors],
            uav_actors=[a.state_dict() for a in self.uav_actors],
            critic=self.critic.state_dict(),
            lambda_il=self.lambda_il,
        )
        torch.save(state, os.path.join(path, "ia_maddpg.pt"))

    def load(self, path: str) -> None:
        ckpt = torch.load(os.path.join(path, "ia_maddpg.pt"),
                          map_location=self.device)
        for a, sd in zip(self.su_actors, ckpt["su_actors"]):
            a.load_state_dict(sd)
        for a, sd in zip(self.uav_actors, ckpt["uav_actors"]):
            a.load_state_dict(sd)
        self.critic.load_state_dict(ckpt["critic"])
        self.lambda_il = ckpt.get("lambda_il", 0.0)
        # Sync targets
        from copy import deepcopy
        self.su_actors_tgt = [deepcopy(a) for a in self.su_actors]
        self.uav_actors_tgt = [deepcopy(a) for a in self.uav_actors]
        self.critic_tgt = deepcopy(self.critic)
