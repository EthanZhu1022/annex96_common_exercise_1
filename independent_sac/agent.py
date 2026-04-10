"""
Per-building SAC agent for Independent SAC baseline.
=====================================================

Each building receives its own SACAgent instance with:
  - A squashed-Gaussian actor (tanh reparameterization trick)
  - Twin soft Q-networks (clipped double-Q reduces overestimation bias)
  - Automatic entropy temperature tuning via a learnable log_alpha
  - An independent replay buffer

No information is shared with other buildings at any point.
This mirrors the spirit of CityLearn's built-in SAC but removes any
central-agent coupling and adds per-building independent training.

Changes vs CityLearn's citylearn/agents/sac.py:
  - No RLC/RBC inheritance chain; purely standalone PyTorch.
  - Automatic temperature tuning (CityLearn SAC uses fixed alpha).
  - Twin Q-networks with soft target updates (polyak averaging).
  - Clean save/load API matching the checkpoint convention of mappo/.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import math

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Neural-network modules
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Squashed-Gaussian policy for a single building.

    Outputs a tanh-squashed action in (-1, 1)^action_dim.  The reparameterization
    trick (rsample) enables gradient flow through sampled actions for the SAC
    actor update.

    Architecture: two ReLU hidden layers → separate mean and log_std heads.
    ReLU is used (rather than Tanh as in mappo/agent.py) because the actor
    receives normalized inputs and ReLU avoids saturation in deeper SAC nets.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) of the pre-tanh Gaussian."""
        h       = self.backbone(obs)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action with reparameterization + tanh squashing.

        Returns:
            action:   tanh-squashed sample ∈ (-1, 1)^action_dim, shape (B, A)
            log_prob: log probability under squashed distribution, shape (B, 1)
        """
        mean, log_std = self(obs)
        std      = log_std.exp()
        dist     = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()          # reparameterized sample
        action   = torch.tanh(pre_tanh)
        # Numerically stable tanh log-prob correction (softplus identity):
        #   log(1 - tanh²(x)) = 2*(log 2 - x - softplus(-2x))
        # This avoids catastrophic cancellation when |x| is large, unlike
        # the naive torch.log(1 - tanh(x)^2 + eps) formulation.
        log_prob = dist.log_prob(pre_tanh) - (
            2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return tanh(mean) for greedy/test-time evaluation."""
        mean, _ = self(obs)
        return torch.tanh(mean)


class TwinQNetwork(nn.Module):
    """Twin Q-networks Q1(s, a) and Q2(s, a).

    Using the minimum of the two Q-values as the Bellman target
    (clipped double-Q) reduces overestimation bias compared to a single critic.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()

        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.min(q1, q2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple ring-buffer replay memory for one building.

    Stores (obs, action, reward, next_obs, done) tuples.
    Uses collections.deque for O(1) push with automatic eviction.
    """

    def __init__(self, capacity: int) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        self.buffer.append((
            obs.copy(), action.copy(), float(reward), next_obs.copy(), float(done)
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(action,   dtype=np.float32),
            np.array(reward,   dtype=np.float32).reshape(-1, 1),
            np.array(next_obs, dtype=np.float32),
            np.array(done,     dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Per-building SAC agent
# ---------------------------------------------------------------------------

class SACAgent:
    """Independent SAC agent for a single building.

    Owns its actor, twin Q-networks, target Q-networks, replay buffer,
    and learnable entropy temperature log_alpha.  No information is shared
    with other agents at any point — truly decentralized.

    Args:
        obs_dim:         Observation dimensionality.
        action_dim:      Action dimensionality.
        hidden_dim:      Hidden layer width for all networks (default 256).
        lr:              Learning rate for actor, critics, and alpha (default 3e-4).
        gamma:           Discount factor (default 0.99).
        tau:             Soft target-network update coefficient ρ (default 5e-3).
                         target ← (1-ρ)·target + ρ·online after each update.
        alpha_init:      Initial entropy temperature (default 0.2).
        target_entropy:  Desired policy entropy; defaults to -action_dim
                         (the standard heuristic from the SAC paper).
        buffer_capacity: Max transitions stored per agent (default 100 000).
        device:          Torch device (auto-detected if None).
    """

    def __init__(
        self,
        obs_dim:         int,
        action_dim:      int,
        hidden_dim:      int            = 256,
        lr:              float          = 3e-4,
        gamma:           float          = 0.99,
        tau:             float          = 5e-3,
        alpha_init:      float          = 0.2,
        target_entropy:  Optional[float] = None,
        buffer_capacity: int            = 100_000,
        max_grad_norm:   float          = 1.0,
        device:          Optional[torch.device] = None,
    ) -> None:
        self.gamma         = gamma
        self.tau           = tau
        self.action_dim    = action_dim
        self.max_grad_norm = max_grad_norm
        self.device        = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.actor         = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic        = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = TwinQNetwork(obs_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Target network is never directly optimized — updated via polyak averaging only.
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Automatic entropy temperature tuning
        self.target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )
        self.log_alpha = torch.tensor(
            np.log(alpha_init), dtype=torch.float32,
            requires_grad=True, device=self.device,
        )

        # Optimizers
        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt  = Adam([self.log_alpha],          lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature (always positive via exp)."""
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Interaction API
    # ------------------------------------------------------------------

    def select_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """Select an action for one environment step.

        Returns a numpy array ∈ (-1, 1)^action_dim (before action-space scaling).

        Args:
            obs:           Normalized observation from CityLearn, shape (obs_dim,).
            deterministic: If True, return tanh(mean) instead of sampling.
                           Use deterministic=True during test evaluation.
        """
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add a transition to this building's replay buffer."""
        self.replay_buffer.push(obs, action, reward, next_obs, done)

    # ------------------------------------------------------------------
    # SAC gradient update
    # ------------------------------------------------------------------

    def update(self, batch_size: int) -> Dict[str, float]:
        """Sample a batch and perform one SAC gradient step.

        Update order (standard SAC):
          1. Critic update  — minimize Bellman error with clipped double-Q target.
          2. Actor update   — maximize E[Q(s, a)] - α·log π(a|s).
          3. Alpha update   — minimize -log_α · (log π + target_entropy).
          4. Soft target    — polyak-average critic_target toward critic.

        Returns:
            Dict with scalar loss values for logging.
        """
        obs, action, reward, next_obs, done = self.replay_buffer.sample(batch_size)

        obs_t      = torch.from_numpy(obs).to(self.device)
        action_t   = torch.from_numpy(action).to(self.device)
        reward_t   = torch.from_numpy(reward).to(self.device)
        next_obs_t = torch.from_numpy(next_obs).to(self.device)
        done_t     = torch.from_numpy(done).to(self.device)

        # ── 1. Critic update ────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs_t)
            q1_next, q2_next = self.critic_target(next_obs_t, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q   = reward_t + (1.0 - done_t) * self.gamma * min_q_next

        q1, q2 = self.critic(obs_t, action_t)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # ── 2. Actor update (freeze critic for efficiency) ───────────────
        for p in self.critic.parameters():
            p.requires_grad_(False)

        new_action, log_prob = self.actor.sample(obs_t)
        q1_pi, q2_pi = self.critic(obs_t, new_action)
        min_q_pi     = torch.min(q1_pi, q2_pi)
        actor_loss   = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ── 3. Temperature (alpha) update ────────────────────────────────
        # Automatic entropy tuning: adjust α so that policy entropy ≈ target_entropy.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── 4. Soft target update ────────────────────────────────────────
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "entropy":     -log_prob.mean().item(),
        }

    # ------------------------------------------------------------------
    # Mode switching & persistence
    # ------------------------------------------------------------------

    def eval_mode(self) -> None:
        """Switch to evaluation mode (disables dropout etc.)."""
        self.actor.eval()
        self.critic.eval()

    def train_mode(self) -> None:
        """Switch back to training mode."""
        self.actor.train()
        self.critic.train()

    def save(self, path_prefix: str) -> None:
        """Save all network weights and log_alpha to disk.

        Files written:
          {path_prefix}_actor.pt
          {path_prefix}_critic.pt
          {path_prefix}_critic_target.pt
          {path_prefix}_log_alpha.pt
        """
        torch.save(self.actor.state_dict(),         f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(),        f"{path_prefix}_critic.pt")
        torch.save(self.critic_target.state_dict(), f"{path_prefix}_critic_target.pt")
        torch.save(self.log_alpha,                  f"{path_prefix}_log_alpha.pt")

    def load(self, path_prefix: str) -> None:
        """Load saved weights from disk."""
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pt", map_location=self.device)
        )
        self.critic.load_state_dict(
            torch.load(f"{path_prefix}_critic.pt", map_location=self.device)
        )
        self.critic_target.load_state_dict(
            torch.load(f"{path_prefix}_critic_target.pt", map_location=self.device)
        )
        saved_log_alpha = torch.load(f"{path_prefix}_log_alpha.pt", map_location=self.device)
        self.log_alpha = saved_log_alpha.detach().requires_grad_(True)
        lr = self.alpha_opt.param_groups[0]["lr"]
        self.alpha_opt = Adam([self.log_alpha], lr=lr)
