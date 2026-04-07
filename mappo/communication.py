"""
Communication module for Hierarchical MAPPO.

Each cluster agent generates a fixed-size message vector from its local observations:
    m_i = f_i(local_obs_i)

All messages are then concatenated (NOT averaged) into a global message tensor:
    M = concat(m_0, m_1, ..., m_{K-1})   shape: (K * msg_dim,)

Each actor then receives (local_obs_i, M) as its full input, so every agent
can condition its policy on what all other agents are "saying" about their clusters.

Concatenation (vs. mean-pooling) preserves each agent's individual signal,
letting downstream actors distinguish which cluster sent which message.

To disable communication entirely, set `use_communication=False` in Config.
The actor input dimension is then just `local_obs_dim` with no message appended.
"""

import torch
import torch.nn as nn


class CommunicationNet(nn.Module):
    """Maps a cluster's local observation to a bounded message vector.

    One independent CommunicationNet per cluster agent. Parameters are
    optimized together with the cluster's Actor (same optimizer group).

    Args:
        local_obs_dim: Dimensionality of the cluster's concatenated local obs.
        msg_dim:       Size of the output message vector (default: 32).
    """

    def __init__(self, local_obs_dim: int, msg_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(local_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, msg_dim),
            nn.Tanh(),   # Bounded messages in [-1, 1] for training stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, local_obs_dim) or (local_obs_dim,)
        Returns:
            message: Tensor of shape (batch, msg_dim) or (msg_dim,)
        """
        return self.net(x)
