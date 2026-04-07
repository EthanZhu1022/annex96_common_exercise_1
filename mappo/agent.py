"""
Actor-Critic networks for Hierarchical MAPPO.

Architecture follows the CTDE (Centralized Training, Decentralized Execution) paradigm:

  DECENTRALIZED ACTORS
  --------------------
  One Actor network per cluster. During *execution*, each actor only sees:
    - The concatenated local observations of the buildings in its cluster.
    - (Optional) The concatenated communication messages M from all agents.
  This makes the policy fully executable without a central coordinator.

  CENTRALIZED CRITIC
  ------------------
  A single shared Critic that receives the *global state*: all buildings'
  observations concatenated in CityLearn's building order. The critic is used
  ONLY during training (advantage estimation) and is never called at deployment.
  Global state access lets the critic provide lower-variance value estimates,
  which speeds up policy optimization.

  SHARED COOPERATIVE REWARD
  -------------------------
  All cluster agents receive the same district-level global reward (sum of
  per-building rewards). This turns the problem into a cooperative game where
  every agent benefits from reducing the district's electricity cost/peaks.
  Using a shared reward with CTDE is the standard MAPPO setup for cooperative
  multi-agent control tasks.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Decentralized Gaussian policy for one cluster of buildings.

    Input:  local_obs (+ concatenated comm messages if communication is on)
            → shape: (actor_input_dim,)
    Output: mean and log_std of a Gaussian, from which we sample a joint
            action vector for *all* buildings in this cluster.

    The output mean is squashed to [-1, 1] via tanh.  CityLearn's internal
    action clipping then constrains each building-level action to its true
    bounds (e.g. heating_device → [0, 1]).

    Args:
        input_dim:  Dimensionality of actor input (obs + optional messages).
        action_dim: Total action dimension for all buildings in this cluster
                    (= sum of per-building action dims in CityLearn ordering).
        hidden_dim: Width of each hidden layer.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Separate heads for mean and log-std allow independent expressivity
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _distribution(self, x: torch.Tensor) -> Normal:
        h       = self.backbone(x)
        mean    = torch.tanh(self.mean_head(h))
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob).

        Used during *rollout collection*. The sampled action is clipped to
        [-1, 1]; log_prob is computed before clipping so gradients are valid.

        Args:
            x: shape (1, input_dim) or (input_dim,)
        Returns:
            action:   shape (..., action_dim), clipped to [-1, 1]
            log_prob: scalar — sum of per-dimension log probs
        """
        dist     = self._distribution(x)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        action   = action.clamp(-1.0, 1.0)
        return action, log_prob

    def evaluate_actions(
        self, x: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log_prob and entropy for *stored* actions.

        Used during *PPO update*. We pass the actions collected during the
        rollout and evaluate them under the *current* (updated) policy to get
        the importance-sampling ratio.

        Args:
            x:       shape (batch, input_dim)
            actions: shape (batch, action_dim)
        Returns:
            log_prob: shape (batch,)
            entropy:  shape (batch,)
        """
        dist     = self._distribution(x)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized value function V(s_global).

    Receives the *full global state* (all buildings' observations concatenated
    in CityLearn's building order) and produces a scalar value estimate.

    Because the global state is available only at training time (CTDE), the
    critic can see information from all clusters simultaneously, which reduces
    variance in the advantage estimates compared to a per-agent critic that
    only sees local observations.

    Args:
        global_obs_dim: Total dimension of the global state vector
                        (= sum of per-building obs dims over ALL buildings).
        hidden_dim:     Width of each hidden layer.
    """

    def __init__(self, global_obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch, global_obs_dim) or (global_obs_dim,)
        Returns:
            value: shape (batch,) — scalar value per sample
        """
        return self.net(x).squeeze(-1)
