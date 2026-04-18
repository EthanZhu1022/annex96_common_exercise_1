"""
PowerNet-style neighbor communication module.

This is a grouped-MAPPO adaptation of PowerNet's local-neighbor communication:

  local_i    = q_o(h_i)
  neigh_i    = mean_j q_h(h_j), for j in N(i)
  out_i      = update([local_i ; neigh_i])

The original PowerNet uses explicit grid neighbors and recurrent hidden-state
updates. In this repo we approximate the neighbor graph within each group using
a configurable local topology and update the current actor features directly.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .base import BaseCommunicationModule


class PowerNetCommunicationModule(BaseCommunicationModule):
    """Neighbor-restricted communication inspired by PowerNet."""

    def __init__(
        self,
        hidden_dim: int,
        comm_hidden_dim: int = 64,
        comm_rounds: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
        comm_num_agents: int = 1,
        comm_topology: Literal["ring", "chain", "full"] = "ring",
        comm_neighbors: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.n_agents = int(comm_num_agents)
        self.comm_topology = str(comm_topology)
        self.comm_neighbors = int(comm_neighbors)

        self.local_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, int(comm_hidden_dim)),
                    nn.ReLU(),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.neighbor_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, int(comm_hidden_dim)),
                    nn.ReLU(),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(comm_hidden_dim) * 2, int(comm_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(comm_hidden_dim), self.hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(self.comm_rounds)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        adj = self._build_adjacency(
            n_agents=self.n_agents,
            topology=self.comm_topology,
            neighbors=self.comm_neighbors,
        )
        self.register_buffer("adjacency", adj, persistent=False)

    @staticmethod
    def _build_adjacency(
        n_agents: int,
        topology: str,
        neighbors: int,
    ) -> torch.Tensor:
        adj = torch.zeros(n_agents, n_agents, dtype=torch.float32)
        if n_agents <= 1:
            return adj

        if topology == "full":
            adj.fill_(1.0)
            adj.fill_diagonal_(0.0)
            return adj

        for i in range(n_agents):
            for step in range(1, max(neighbors, 1) + 1):
                left = i - step
                right = i + step

                if topology == "ring":
                    adj[i, left % n_agents] = 1.0
                    adj[i, right % n_agents] = 1.0
                elif topology == "chain":
                    if left >= 0:
                        adj[i, left] = 1.0
                    if right < n_agents:
                        adj[i, right] = 1.0
                else:
                    raise ValueError(
                        f"Unknown PowerNet topology '{topology}'. Valid options: ring, chain, full."
                    )

        adj.fill_diagonal_(0.0)
        return adj

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply neighbor-restricted communication over group members."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        batch_size, n_agents, hidden_dim = features.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )
        if n_agents != self.n_agents:
            raise ValueError(
                f"Expected n_agents={self.n_agents}, got {n_agents}."
            )

        h = features
        adj = self.adjacency.to(device=h.device, dtype=h.dtype)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        adj_batch = adj.unsqueeze(0).expand(batch_size, -1, -1)

        for local_encoder, neighbor_encoder, update_layer in zip(
            self.local_encoders,
            self.neighbor_encoders,
            self.update_layers,
        ):
            local = local_encoder(h)
            neighbor_hidden = neighbor_encoder(h)
            neigh_mean = torch.bmm(adj_batch, neighbor_hidden) / degree.unsqueeze(0)

            update_in = torch.cat([local, neigh_mean], dim=-1)
            updated = self.dropout(update_layer(update_in))

            if self.use_residual:
                h = h + updated
            else:
                h = updated

        return h

