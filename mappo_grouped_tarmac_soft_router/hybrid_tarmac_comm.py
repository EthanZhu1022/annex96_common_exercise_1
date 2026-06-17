"""
Hybrid TarMAC communication for grouped MAPPO.

This module keeps TarMAC's global targeted attention and exposes three fusion
modes for controlled ablations:

  relu:
    local_i = ReLU(W_local h_i)
    delta_i = update([local_i ; context_i])

  linear:
    local_i = W_local h_i
    delta_i = update([local_i ; context_i])

  gated:
    projected_i = W_context context_i
    gate_i = sigmoid(W_gate [h_i ; projected_i])
    delta_i = gate_i * projected_i

All modes use the same TarMAC attention:

  q_i = W_q h_i
  k_i = W_k h_i
  v_i = W_v h_i
  a_ij = softmax_j(q_i^T k_j / sqrt(d_k))
  context_i = sum_j a_ij v_j
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
        fusion_mode: str = "relu",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.local_dim = int(comm_hidden_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.fusion_mode = str(fusion_mode).lower().strip()
        if self.fusion_mode not in {"relu", "linear", "gated"}:
            raise ValueError(
                f"Unknown fusion_mode='{fusion_mode}'. "
                "Valid options are: relu, linear, gated."
            )
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

        if self.fusion_mode in {"relu", "linear"}:
            activation = nn.ReLU if self.fusion_mode == "relu" else nn.Identity
            self.local_encoders = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.local_dim),
                        activation(),
                    )
                    for _ in range(self.comm_rounds)
                ]
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
            self.context_projections = nn.ModuleList()
            self.gate_layers = nn.ModuleList()
        else:
            self.local_encoders = nn.ModuleList()
            self.update_layers = nn.ModuleList()
            self.context_projections = nn.ModuleList(
                [
                    nn.Linear(self.value_dim, self.hidden_dim)
                    for _ in range(self.comm_rounds)
                ]
            )
            self.gate_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                        nn.Sigmoid(),
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
        for round_idx, (query_layer, key_layer, value_layer) in enumerate(
            zip(self.query_layers, self.key_layers, self.value_layers)
        ):
            queries = query_layer(h)
            keys = key_layer(h)
            values = value_layer(h)

            attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, values)

            if self.fusion_mode in {"relu", "linear"}:
                local = self.local_encoders[round_idx](h)
                delta = self.update_layers[round_idx](
                    torch.cat([local, context], dim=-1)
                )
                delta = torch.tanh(self.dropout(delta))
            else:
                projected_context = self.context_projections[round_idx](context)
                gate = self.gate_layers[round_idx](
                    torch.cat([h, projected_context], dim=-1)
                )
                delta = gate * self.dropout(projected_context)

            if self.use_residual:
                h = h + delta
            else:
                h = delta

        return h
