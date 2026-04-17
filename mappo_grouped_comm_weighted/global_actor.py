"""
Weighted grouped communication actor controller.

Actor path:
  1. Each group uses its own actor encoder.
  2. Encoded features from all agents are assembled into a global tensor.
  3. Communication uses weighted same-group / other-group means.
  4. The fused features are split back to each group.
  5. Each group uses its own action head to produce actions / log-probs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import check

from mappo_grouped_comm_weighted.weighted_comm import WeightedGroupCommunication


class WeightedGlobalCommActorController(nn.Module):
    """Run grouped encoders with simple weighted communication."""

    def __init__(
        self,
        actors: Sequence[nn.Module],
        group_indices: Sequence[Sequence[int]],
        comm_hidden_dim: int = 64,
        alpha: float = 0.75,
        beta: float = 0.25,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.group_actors = nn.ModuleList(list(actors))
        self.group_indices = [list(map(int, idx)) for idx in group_indices]
        self.group_sizes = [len(idx) for idx in self.group_indices]
        self.n_agents_total = sum(self.group_sizes)

        ref_actor = self.group_actors[0]
        self.hidden_size = ref_actor.hidden_size
        self.tpdv = ref_actor.tpdv
        self._use_policy_active_masks = ref_actor._use_policy_active_masks
        self.comm = WeightedGroupCommunication(
            hidden_dim=self.hidden_size,
            group_indices=self.group_indices,
            comm_hidden_dim=comm_hidden_dim,
            alpha=alpha,
            beta=beta,
            use_residual=use_residual,
            dropout=dropout,
        )

    def _encode_group(
        self,
        actor: nn.Module,
        obs,
        rnn_states,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        features = actor.base(obs)
        if actor._use_naive_recurrent_policy or actor._use_recurrent_policy:
            features, rnn_states = actor.rnn(features, rnn_states, masks)

        return features, rnn_states

    def _apply_weighted_comm(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.comm(global_features)

    def act(
        self,
        obs,
        rnn_states,
        masks,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one rollout / eval step with weighted communication."""
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        n_agents = obs.shape[0]
        if n_agents != self.n_agents_total:
            raise ValueError(
                f"Expected {self.n_agents_total} agents, got {n_agents}."
            )

        global_features = torch.zeros(
            n_agents, self.hidden_size, dtype=obs.dtype, device=obs.device
        )
        new_rnn_states = rnn_states.clone()

        for actor, idx_k in zip(self.group_actors, self.group_indices):
            feats_k, rnn_k = self._encode_group(
                actor=actor,
                obs=obs[idx_k],
                rnn_states=rnn_states[idx_k],
                masks=masks[idx_k],
            )
            global_features[idx_k] = feats_k
            new_rnn_states[idx_k] = rnn_k

        global_features = self._apply_weighted_comm(global_features.unsqueeze(0)).squeeze(0)

        act_dim = None
        all_actions = None
        all_log_probs = None

        for actor, idx_k in zip(self.group_actors, self.group_indices):
            actions_k, logp_k = actor.act(global_features[idx_k], None, deterministic)
            if all_actions is None:
                act_dim = actions_k.shape[-1]
                all_actions = torch.zeros(
                    n_agents, act_dim, dtype=actions_k.dtype, device=actions_k.device
                )
                all_log_probs = torch.zeros(
                    n_agents, logp_k.shape[-1], dtype=logp_k.dtype, device=logp_k.device
                )
            all_actions[idx_k] = actions_k
            all_log_probs[idx_k] = logp_k

        return all_actions, all_log_probs, new_rnn_states

    def evaluate_actions(
        self,
        group_batches: Sequence[Dict[str, torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Evaluate grouped actions after weighted communication."""
        if len(group_batches) != len(self.group_actors):
            raise ValueError("group_batches length must match number of group actors.")

        encoded_groups: List[torch.Tensor] = []
        timesteps_in_batch: Optional[int] = None

        for actor, idx_k, batch in zip(self.group_actors, self.group_indices, group_batches):
            obs = batch["obs"]
            rnn_states = batch["rnn_states"]
            masks = batch["masks"]

            feats_k, _ = self._encode_group(actor=actor, obs=obs, rnn_states=rnn_states, masks=masks)
            n_k = len(idx_k)
            if feats_k.shape[0] % n_k != 0:
                raise ValueError(
                    f"Batch size {feats_k.shape[0]} is not divisible by group size {n_k}."
                )
            t_mb = feats_k.shape[0] // n_k
            if timesteps_in_batch is None:
                timesteps_in_batch = t_mb
            elif timesteps_in_batch != t_mb:
                raise ValueError("All group mini-batches must contain the same sampled timesteps.")

            encoded_groups.append(feats_k.view(t_mb, n_k, self.hidden_size))

        assert timesteps_in_batch is not None
        global_features = torch.zeros(
            timesteps_in_batch,
            self.n_agents_total,
            self.hidden_size,
            dtype=encoded_groups[0].dtype,
            device=encoded_groups[0].device,
        )
        for idx_k, feats_k in zip(self.group_indices, encoded_groups):
            global_features[:, idx_k, :] = feats_k

        global_features = self._apply_weighted_comm(global_features)

        action_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        for actor, idx_k, batch in zip(self.group_actors, self.group_indices, group_batches):
            features_k = global_features[:, idx_k, :].reshape(-1, self.hidden_size)
            available_actions = batch.get("available_actions")
            active_masks = batch.get("active_masks")

            logp_k, entropy_k = actor.act.evaluate_actions(
                features_k,
                batch["action"],
                available_actions,
                active_masks=(
                    active_masks if self._use_policy_active_masks else None
                ),
            )
            action_log_probs.append(logp_k)
            entropies.append(entropy_k)

        return action_log_probs, entropies
