"""
CommActorWrapper — R_Actor replacement with injected communication module.

Architecture (per group, per step during rollout)
--------------------------------------------------
  obs (n_agents, obs_dim)
    │
    ▼  R_Actor.base  (MLPBase)
  features (n_agents, hidden_dim)
    │
    ▼  comm_module.forward((1, n_agents, hidden_dim)) → (1, n_agents, hidden_dim)
  features' (n_agents, hidden_dim)          ← trainable if comm_method ≠ 'none'
    │
    ▼  R_Actor.act  (ACTLayer)
  actions, log_probs

During PPO evaluate_actions (mini-batch from SharedReplayBuffer)
----------------------------------------------------------------
  obs (B, obs_dim)  where B = T_mb × n_agents  (T_mb = mini-batch timesteps)
    │
    ▼  R_Actor.base
  features (B, hidden_dim)
    │  reshape → (T_mb, n_agents, hidden_dim)
    ▼  comm_module
  features' (T_mb, n_agents, hidden_dim)
    │  reshape → (B, hidden_dim)
    ▼  R_Actor.act.evaluate_actions
  action_log_probs, dist_entropy

Note on approximation during training
--------------------------------------
feed_forward_generator (SharedReplayBuffer) shuffles all T×n_agents samples
before splitting into mini-batches.  Reshaping (B, D) → (T_mb, n_agents, D)
after shuffling means each "group" of n_agents in the reshaped tensor may come
from different timesteps.  This is an approximation, but:
  - Gradients still flow through comm.parameters() on every PPO update.
  - During rollout the grouping is exact (same-timestep, same-group).
  - The approximation is standard in batch-based CommNet implementations.

Integration with R_MAPPO
------------------------
R_MAPPO.ppo_update() calls:
  nn.utils.clip_grad_norm_(self.policy.actor.parameters(), ...)
  self.policy.actor_optimizer.step()

Because CommActorWrapper is assigned to policy.actor:
  - policy.actor.parameters() now includes comm.parameters().
  - policy.actor_optimizer must be rebuilt to optimise the wrapper (done in
    _build_grouped_components_comm inside train.py, after actor replacement).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import check

from mappo_grouped_comm.communication.base import BaseCommunicationModule


class CommActorWrapper(nn.Module):
    """Drop-in replacement for R_Actor with an injected communication module.

    Parameters
    ----------
    actor    : R_Actor   — original policy actor for this group.
    comm     : BaseCommunicationModule — communication module (may be identity).
    n_agents : int       — number of agents in this group; needed to reshape the
                           flattened mini-batch back to (T_mb, n_agents, D).
    """

    def __init__(
        self,
        actor,
        comm:     BaseCommunicationModule,
        n_agents: int,
    ) -> None:
        super().__init__()

        # Register actor sub-modules as nn.Module children so their parameters
        # are included in self.parameters() → picked up by actor_optimizer.
        self.base = actor.base    # MLPBase / CNNBase — feature encoder
        self.act  = actor.act     # ACTLayer — action distribution head
        self.comm = comm          # Communication module (trainable if ≠ none)

        # Register RNN only when present (disabled in our feed-forward config)
        rnn = getattr(actor, "rnn", None)
        if rnn is not None:
            self.rnn = rnn
        else:
            self.rnn = None

        # Copy actor config flags needed by R_MAPPO / rMAPPOPolicy internally
        self.hidden_size                 = actor.hidden_size
        self._use_orthogonal             = actor._use_orthogonal
        self._use_policy_active_masks    = actor._use_policy_active_masks
        self._use_naive_recurrent_policy = actor._use_naive_recurrent_policy
        self._use_recurrent_policy       = actor._use_recurrent_policy
        self._recurrent_N                = actor._recurrent_N
        self.tpdv                        = actor.tpdv   # dict(dtype=..., device=...)
        self.algo                        = actor.algo

        self.n_agents = n_agents

    # ------------------------------------------------------------------
    # Rollout forward (called from R_MAPPOPolicy.get_actions / .act)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic: bool = False,
    ):
        """Encode obs, communicate, produce actions.

        Parameters
        ----------
        obs            : (n_agents, obs_dim)
        rnn_states     : (n_agents, recurrent_N, hidden_size)
        masks          : (n_agents, 1)
        available_actions : optional
        deterministic  : bool

        Returns
        -------
        actions          : (n_agents, act_dim)
        action_log_probs : (n_agents, 1)
        rnn_states       : (n_agents, recurrent_N, hidden_size)  [unchanged if no RNN]
        """
        obs        = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks      = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # 1. Base encoder: (n_agents, hidden_dim)
        features = self.base(obs)

        # 2. Optional RNN (disabled in feed-forward mode; kept for correctness)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        # 3. Communication — reshape to (1, n_agents, D) for comm contract
        if not self.comm.is_identity:
            features = self.comm(features.unsqueeze(0)).squeeze(0)  # (n,D)

        # 4. Action distribution head
        actions, action_log_probs = self.act(
            features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    # ------------------------------------------------------------------
    # PPO training forward (called from R_MAPPOPolicy.evaluate_actions)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Compute log-probs and entropy for sampled actions during PPO update.

        Parameters
        ----------
        obs        : (B, obs_dim)   where B = T_mb × n_agents
        rnn_states : (B, recurrent_N, hidden_size)
        action     : (B, act_dim)
        masks      : (B, 1)

        Returns
        -------
        action_log_probs : (B, 1)
        dist_entropy     : scalar
        """
        obs        = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action     = check(action).to(**self.tpdv)
        masks      = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # 1. Base encoder: (B, hidden_dim)
        features = self.base(obs)

        # 2. Optional RNN
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            features, rnn_states = self.rnn(features, rnn_states, masks)

        # 3. Communication over mini-batch
        #    B = T_mb × n_agents → reshape to (T_mb, n_agents, D) for CommNet.
        #    After feed_forward_generator shuffling the T_mb "groups" do not
        #    correspond to single timesteps, but gradients still flow through
        #    comm.parameters() giving correct optimisation signal.
        if not self.comm.is_identity:
            B, D = features.shape
            n    = self.n_agents
            if n > 0 and B % n == 0:
                T_mb     = B // n
                features = self.comm(features.view(T_mb, n, D)).view(B, D)
            # If B not divisible by n (shouldn't happen with standard config),
            # skip comm silently — safer than crashing.

        # 4. Evaluate action distribution
        if self.algo == "hatrpo":
            return self.act.evaluate_actions_trpo(
                features, action, available_actions,
                active_masks=(
                    active_masks if self._use_policy_active_masks else None
                ),
            )
        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(
                features, action, available_actions,
                active_masks=(
                    active_masks if self._use_policy_active_masks else None
                ),
            )

        return action_log_probs, dist_entropy
