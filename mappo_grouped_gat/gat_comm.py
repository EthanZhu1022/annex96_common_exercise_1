"""
Graph Attention Network communication for grouped MAPPO.

Reference:
  Velickovic et al., "Graph Attention Networks", ICLR 2018.

For each communication round and attention head, this module applies the GAT
masked self-attention update over a fixed building-similarity graph:

  z_i = W h_i
  e_ij = LeakyReLU(a^T [z_i || z_j])
  alpha_ij = softmax_j(e_ij), only for j in N_i
  h'_i = concat_heads(sum_j alpha_ij z_j)

The graph is not a CE1 physical feeder graph. It is a constructed similarity
graph: K-means groups are fully connected internally, and sparse inter-group
edges connect the closest groups. This preserves GAT's masked-neighborhood
attention while matching the data available in CE1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mappo_grouped_comm.communication.base import BaseCommunicationModule


class GroupSimilarityGATCommunicationModule(BaseCommunicationModule):
    """Multi-head masked graph attention over a fixed similarity graph."""

    def __init__(
        self,
        hidden_dim: int,
        adjacency_mask: torch.Tensor,
        num_heads: int = 4,
        comm_rounds: int = 1,
        use_residual: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        activation: str = "elu",
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.comm_rounds = int(comm_rounds)
        self.use_residual = bool(use_residual)
        self.negative_slope = float(negative_slope)
        self.activation = str(activation).lower().strip()

        if self.num_heads <= 0:
            raise ValueError(f"Expected num_heads > 0, got {self.num_heads}.")
        if self.comm_rounds <= 0:
            raise ValueError(f"Expected comm_rounds > 0, got {self.comm_rounds}.")

        if head_dim is None:
            if self.hidden_dim % self.num_heads != 0:
                raise ValueError(
                    "hidden_dim must be divisible by num_heads when head_dim is not set; "
                    f"got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
                )
            head_dim = self.hidden_dim // self.num_heads
        self.head_dim = int(head_dim)
        self.concat_dim = self.num_heads * self.head_dim

        adjacency_mask = torch.as_tensor(adjacency_mask, dtype=torch.bool)
        if adjacency_mask.dim() != 2 or adjacency_mask.shape[0] != adjacency_mask.shape[1]:
            raise ValueError(
                f"Expected square adjacency mask, got shape {tuple(adjacency_mask.shape)}."
            )
        if not torch.all(adjacency_mask.any(dim=-1)):
            raise ValueError("Every node must have at least one attention neighbor.")

        self.n_agents = int(adjacency_mask.shape[0])
        self.register_buffer("adjacency_mask", adjacency_mask)

        self.feature_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, self.concat_dim, bias=False)
                for _ in range(self.comm_rounds)
            ]
        )
        self.attn_src = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.num_heads, self.head_dim))
                for _ in range(self.comm_rounds)
            ]
        )
        self.attn_dst = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.num_heads, self.head_dim))
                for _ in range(self.comm_rounds)
            ]
        )
        self.output_layers = nn.ModuleList(
            [
                nn.Identity()
                if self.concat_dim == self.hidden_dim
                else nn.Linear(self.concat_dim, self.hidden_dim, bias=False)
                for _ in range(self.comm_rounds)
            ]
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.feature_layers:
            nn.init.xavier_uniform_(layer.weight)
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        for src, dst in zip(self.attn_src, self.attn_dst):
            nn.init.xavier_uniform_(src)
            nn.init.xavier_uniform_(dst)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "elu":
            return F.elu(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation in {"none", "identity"}:
            return x
        raise ValueError(f"Unknown GAT activation '{self.activation}'.")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply GAT communication to features shaped (batch, n_agents, hidden_dim)."""
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        batch_size, n_agents, hidden_dim = features.shape
        if n_agents != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} agents, got {n_agents}.")
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}.")

        h = features
        mask = self.adjacency_mask.view(1, n_agents, n_agents, 1)

        for layer, src_attn, dst_attn, output_layer in zip(
            self.feature_layers,
            self.attn_src,
            self.attn_dst,
            self.output_layers,
        ):
            z = layer(h).view(batch_size, n_agents, self.num_heads, self.head_dim)
            z = self.dropout(z)

            src_score = (z * src_attn.view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1)
            dst_score = (z * dst_attn.view(1, 1, self.num_heads, self.head_dim)).sum(dim=-1)
            logits = src_score.unsqueeze(2) + dst_score.unsqueeze(1)
            logits = F.leaky_relu(logits, negative_slope=self.negative_slope)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

            attention = torch.softmax(logits, dim=2)
            attention = self.dropout(attention)
            context = torch.einsum("bijh,bjhf->bihf", attention, z)
            updated = context.reshape(batch_size, n_agents, self.concat_dim)
            updated = output_layer(updated)
            updated = self._activate(updated)

            h = h + updated if self.use_residual else updated

        return h
