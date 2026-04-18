"""
TarMAC-style communication for grouped MAPPO.

This is a MAPPO adaptation of TarMAC's signature-based targeted communication:

  q_i = W_q h_i
  k_i = W_k h_i
  v_i = W_v h_i
  a_ij = softmax_j(q_i^T k_j / sqrt(d_k))
  c_i = sum_j a_ij v_j
  h_i' = tanh(W[h_i ; c_i])

The communication is global across all agents and can be repeated for multiple
rounds before the grouped action heads are applied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mappo_grouped_comm.communication.base import BaseCommunicationModule


class TarMACCommunicationModule(BaseCommunicationModule):
    """TarMAC-style targeted communication with soft attention."""

    def __init__(
        self,
        hidden_dim: int,
        comm_hidden_dim: int = 64,
        key_dim: int = 32,
        value_dim: int = 64,
        comm_rounds: int = 1,
        use_residual: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.scale = 1.0 / math.sqrt(max(self.key_dim, 1))

        self.query_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.key_dim) for _ in range(self.comm_rounds)]
        )
        self.key_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.key_dim) for _ in range(self.comm_rounds)]
        )
        self.value_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.value_dim) for _ in range(self.comm_rounds)]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim + self.value_dim, int(comm_hidden_dim)),
                    nn.Tanh(),
                    nn.Linear(int(comm_hidden_dim), self.hidden_dim),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply TarMAC-style targeted communication."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        h = features
        for query_layer, key_layer, value_layer, update_layer in zip(
            self.query_layers,
            self.key_layers,
            self.value_layers,
            self.update_layers,
        ):
            queries = query_layer(h)
            keys = key_layer(h)
            values = value_layer(h)

            attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, values)

            updated = update_layer(torch.cat([h, context], dim=-1))
            updated = torch.tanh(self.dropout(updated))

            if self.use_residual:
                h = h + updated
            else:
                h = updated

        return h

