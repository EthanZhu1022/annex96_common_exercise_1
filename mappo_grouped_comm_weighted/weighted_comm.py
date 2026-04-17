"""
Simple same-group / other-group weighted communication.

The communication rule is intentionally minimal and stable:

  msg_same  = mean(h_same_group)
  msg_other = mean(h_other_groups)
  new_msg   = alpha * msg_same + beta * msg_other
  out_i     = h_i + MLP(new_msg)

This keeps the structure close to the original CommNet-style path in
`mappo_grouped_comm_v2`: the self feature is preserved, while only the message
is transformed before the residual update.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class WeightedGroupCommunication(nn.Module):
    """Group-aware weighted communication without attention."""

    def __init__(
        self,
        hidden_dim: int,
        group_indices: Sequence[Sequence[int]],
        comm_hidden_dim: int = 64,
        alpha: float = 0.75,
        beta: float = 0.25,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if alpha <= beta:
            raise ValueError(
                f"Expected alpha > beta for grouped communication, got alpha={alpha}, beta={beta}."
            )
        if alpha < 0.0 or beta < 0.0:
            raise ValueError(
                f"Expected non-negative communication weights, got alpha={alpha}, beta={beta}."
            )

        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_residual = bool(use_residual)
        self.group_indices = [list(map(int, idx)) for idx in group_indices]
        self.group_sizes = [len(idx) for idx in self.group_indices]
        self.n_agents_total = sum(self.group_sizes)

        self.msg_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, int(comm_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(comm_hidden_dim), self.hidden_dim),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply weighted same-group / other-group message passing.

        Parameters
        ----------
        features : torch.Tensor
            Shape (batch, n_agents, hidden_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, n_agents, hidden_dim).
        """
        if features.dim() != 3:
            raise ValueError(
                f"Expected features with shape (batch, n_agents, hidden_dim), got {tuple(features.shape)}."
            )

        batch_size, n_agents, hidden_dim = features.shape
        if n_agents != self.n_agents_total:
            raise ValueError(
                f"Expected {self.n_agents_total} agents, got {n_agents}."
            )
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, got {hidden_dim}."
            )

        communicated = torch.zeros_like(features)
        global_sum = features.sum(dim=1, keepdim=True)

        for idx_k, n_k in zip(self.group_indices, self.group_sizes):
            group_feats = features[:, idx_k, :]
            group_sum = group_feats.sum(dim=1, keepdim=True)

            msg_same = group_sum / max(n_k, 1)
            if n_agents > n_k:
                msg_other = (global_sum - group_sum) / (n_agents - n_k)
            else:
                msg_other = torch.zeros_like(msg_same)

            new_msg = self.alpha * msg_same + self.beta * msg_other
            msg_proj = self.msg_mlp(
                new_msg.expand(batch_size, n_k, hidden_dim).reshape(batch_size * n_k, hidden_dim)
            )
            msg_proj = self.dropout(msg_proj).reshape(batch_size, n_k, hidden_dim)

            if self.use_residual:
                communicated[:, idx_k, :] = group_feats + msg_proj
            else:
                communicated[:, idx_k, :] = msg_proj

        return communicated
