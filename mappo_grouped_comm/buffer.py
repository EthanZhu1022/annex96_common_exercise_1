"""
Grouped replay buffer utilities for communication-aware MAPPO.

The default on-policy SharedReplayBuffer.feed_forward_generator() shuffles every
agent-time sample independently. That breaks communication training because the
actor's communication module expects all agents from the same group and the same
timestep to stay together.

This subclass keeps the agent dimension intact when sampling feed-forward PPO
mini-batches:
  - sample unit: one timestep across all agents in the group
  - shuffle axis: timestep/thread dimension only
  - yielded tensors: flattened back to (T_mb * n_agents, ...)

This preserves exact group structure for CommNet during PPO updates while
remaining compatible with R_MAPPO's existing update code.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch

from onpolicy.utils.shared_buffer import SharedReplayBuffer


class GroupedSharedReplayBuffer(SharedReplayBuffer):
    """Replay buffer whose feed-forward batches preserve per-timestep agent groups."""

    def feed_forward_generator(
        self,
        advantages,
        num_mini_batch: Optional[int] = None,
        mini_batch_size: Optional[int] = None,
    ) -> Iterator[Tuple]:
        """Yield PPO mini-batches without destroying intra-group communication structure."""
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({}).".format(
                    n_rollout_threads,
                    episode_length,
                    batch_size,
                    num_mini_batch,
                )
            )
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(
            batch_size, num_agents, *self.share_obs.shape[3:]
        )
        obs = self.obs[:-1].reshape(
            batch_size, num_agents, *self.obs.shape[3:]
        )
        rnn_states = self.rnn_states[:-1].reshape(
            batch_size, num_agents, *self.rnn_states.shape[3:]
        )
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            batch_size, num_agents, *self.rnn_states_critic.shape[3:]
        )
        actions = self.actions.reshape(
            batch_size, num_agents, self.actions.shape[-1]
        )
        value_preds = self.value_preds[:-1].reshape(batch_size, num_agents, 1)
        returns = self.returns[:-1].reshape(batch_size, num_agents, 1)
        masks = self.masks[:-1].reshape(batch_size, num_agents, 1)
        active_masks = self.active_masks[:-1].reshape(batch_size, num_agents, 1)
        action_log_probs = self.action_log_probs.reshape(
            batch_size, num_agents, self.action_log_probs.shape[-1]
        )
        if advantages is not None:
            advantages = advantages.reshape(batch_size, num_agents, 1)

        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(
                batch_size, num_agents, self.available_actions.shape[-1]
            )
        else:
            available_actions = None

        for indices in sampler:
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(
                -1, *rnn_states_critic.shape[2:]
            )
            actions_batch = actions[indices].reshape(-1, actions.shape[-1])
            value_preds_batch = value_preds[indices].reshape(-1, 1)
            return_batch = returns[indices].reshape(-1, 1)
            masks_batch = masks[indices].reshape(-1, 1)
            active_masks_batch = active_masks[indices].reshape(-1, 1)
            old_action_log_probs_batch = action_log_probs[indices].reshape(
                -1, action_log_probs.shape[-1]
            )
            adv_targ = advantages[indices].reshape(-1, 1) if advantages is not None else None

            if available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(
                    -1, available_actions.shape[-1]
                )
            else:
                available_actions_batch = None

            yield (
                share_obs_batch,
                obs_batch,
                rnn_states_batch,
                rnn_states_critic_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                active_masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                available_actions_batch,
            )
