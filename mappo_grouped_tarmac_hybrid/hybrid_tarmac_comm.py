"""
Hybrid TarMAC communication for grouped MAPPO.

This module keeps TarMAC's global targeted attention while using separate local
and message encoders before the PowerNet-style concatenation and residual
update:

  local_i = ReLU(W_local h_i)
  q_i = W_q h_i
  k_i = W_k h_i
  v_i = W_v h_i
  a_ij = softmax_j(q_i^T k_j / sqrt(d_k))
  context_i = sum_j a_ij v_j
  delta_i = update([local_i ; context_i])
  h_i' = h_i + delta_i
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mappo_grouped_comm.communication.base import BaseCommunicationModule


class HybridTarMACCommunicationModule(BaseCommunicationModule):
    """TarMAC attention with separately encoded local features."""

    def __init__(
        self,
        hidden_dim: int,
        comm_hidden_dim: int = 64,
        key_dim: int = 32,
        value_dim: int = 64,
        comm_rounds: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.local_dim = int(comm_hidden_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.scale = 1.0 / math.sqrt(max(self.key_dim, 1))

        self.local_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.local_dim),
                    nn.ReLU(),
                )
                for _ in range(self.comm_rounds)
            ]
        )
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
                    nn.Linear(self.local_dim + self.value_dim, self.local_dim),
                    nn.Tanh(),
                    nn.Linear(self.local_dim, self.hidden_dim),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply global attention followed by a local/context residual update."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        h = features
        for local_encoder, query_layer, key_layer, value_layer, update_layer in zip(
            self.local_encoders,
            self.query_layers,
            self.key_layers,
            self.value_layers,
            self.update_layers,
        ):
            local = local_encoder(h)
            queries = query_layer(h)
            keys = key_layer(h)
            values = value_layer(h)

            attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, values)

            delta = update_layer(torch.cat([local, context], dim=-1))
            delta = torch.tanh(self.dropout(delta))

            if self.use_residual:
                h = h + delta
            else:
                h = delta

        return h
