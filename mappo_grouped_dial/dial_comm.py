"""
Simplified DIAL-style communication for grouped MAPPO.

The module keeps a global communication scope across all agents:

  h_i -> message_mlp -> m_i
  m_i_train = sigmoid(m_i + noise)
  m_i_eval  = 1[m_i > 0]
  recv_i    = mean(m_j, j != i)
  out_i     = h_i + recv_mlp(recv_i)

This is not a paper-faithful DQN/C-Net implementation of DIAL. It is a MAPPO
adaptation that preserves the key DIAL property we care about here: messages are
continuous and differentiable during training, but discretised at execution.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mappo_grouped_comm.communication.base import BaseCommunicationModule


class DIALCommunicationModule(BaseCommunicationModule):
    """Global DIAL-style differentiable communication with DRU."""

    def __init__(
        self,
        hidden_dim: int,
        comm_hidden_dim: int = 64,
        message_dim: int = 32,
        comm_rounds: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
        dru_noise: float = 0.5,
        discretize_eval: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.message_dim = int(message_dim)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.dru_noise = float(dru_noise)
        self.discretize_eval = bool(discretize_eval)

        self.message_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, int(comm_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(comm_hidden_dim), self.message_dim),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.receive_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.message_dim, int(comm_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(comm_hidden_dim), self.hidden_dim),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _apply_dru(self, message_logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.dru_noise > 0.0:
                noise = torch.randn_like(message_logits) * self.dru_noise
                message_logits = message_logits + noise
            return torch.sigmoid(message_logits)

        if self.discretize_eval:
            return (message_logits > 0.0).to(message_logits.dtype)
        return torch.sigmoid(message_logits)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply comm_rounds rounds of DIAL-style message passing."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        h = features
        batch_size, n_agents, hidden_dim = h.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )

        for message_mlp, receive_mlp in zip(self.message_mlps, self.receive_mlps):
            message_logits = message_mlp(h.reshape(batch_size * n_agents, hidden_dim)).reshape(
                batch_size, n_agents, self.message_dim
            )
            messages = self._apply_dru(message_logits)

            if n_agents > 1:
                recv = (messages.sum(dim=1, keepdim=True) - messages) / (n_agents - 1)
            else:
                recv = torch.zeros_like(messages)

            recv_proj = receive_mlp(
                recv.reshape(batch_size * n_agents, self.message_dim)
            ).reshape(batch_size, n_agents, hidden_dim)
            recv_proj = self.dropout(recv_proj)

            if self.use_residual:
                h = h + recv_proj
            else:
                h = recv_proj

        return h

