"""
CommNet communication module.

Reference: Sukhbaatar et al., "Learning Multiagent Communication with
Backpropagation", NeurIPS 2016.  https://arxiv.org/abs/1605.07736

Architecture (per round r)
--------------------------
  1. msg_i = mean_{j ≠ i}(h_j)          — mean-pool other agents' features
  2. h_i   = h_i + dropout(W_r(msg_i))  — residual update via 2-layer MLP
     (if use_residual=False: h_i = W_r(msg_i), less stable but sometimes useful)

The module processes a (batch, n_agents, hidden_dim) tensor and returns a
tensor of the same shape.  batch may represent independent timesteps so the
communication is computed over the n_agents axis only.

Gradient flow
-------------
W_r (msg_layers) are learned jointly with the actor encoder because
CommActorWrapper registers this module as self.comm, making its parameters
part of policy.actor.parameters() which the actor optimizer updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseCommunicationModule


class CommNetCommunicationModule(BaseCommunicationModule):
    """CommNet: per-round mean-pooling message passing with residual updates.

    Parameters
    ----------
    hidden_dim      : int
        Dimensionality of incoming actor features (= actor hidden_size).
    comm_hidden_dim : int
        Internal projection size for the message MLP (default 64).
    comm_rounds     : int
        Number of sequential communication rounds (default 1).
    use_residual    : bool
        If True, h = h + msg_proj (residual connection).
        If False, h = msg_proj (full replacement — less stable).
    dropout         : float
        Dropout probability applied to each projected message (default 0.0).
    """

    def __init__(
        self,
        hidden_dim:      int,
        comm_hidden_dim: int   = 64,
        comm_rounds:     int   = 1,
        use_residual:    bool  = True,
        dropout:         float = 0.0,
    ) -> None:
        super().__init__()
        self.comm_rounds  = comm_rounds
        self.use_residual = use_residual
        self.hidden_dim   = hidden_dim

        # One 2-layer MLP per round: hidden_dim → comm_hidden_dim → hidden_dim
        # Projects the mean-pooled message into the hidden feature space.
        self.msg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, comm_hidden_dim),
                nn.ReLU(),
                nn.Linear(comm_hidden_dim, hidden_dim),
            )
            for _ in range(comm_rounds)
        ])

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply comm_rounds rounds of CommNet communication.

        Parameters
        ----------
        features : (batch, n_agents, hidden_dim)

        Returns
        -------
        (batch, n_agents, hidden_dim) — communication-enhanced features
        """
        h        = features                           # (B, N, D)
        n_agents = h.shape[1]

        for r in range(self.comm_rounds):
            # Sum of all agents: (B, 1, D) → broadcast subtract self
            sum_h = h.sum(dim=1, keepdim=True)        # (B, 1, D)

            if n_agents > 1:
                # Mean of OTHER agents (exclude self): (B, N, D)
                msg = (sum_h - h) / (n_agents - 1)
            else:
                # Single agent: no peers → zero message (no-op round)
                msg = torch.zeros_like(h)

            # Apply message MLP over (B*N, D) then reshape back
            B, N, D = h.shape
            msg_proj = self.msg_layers[r](
                msg.reshape(B * N, D)
            ).reshape(B, N, D)
            msg_proj = self.dropout(msg_proj)

            if self.use_residual:
                h = h + msg_proj
            else:
                h = msg_proj

        return h
