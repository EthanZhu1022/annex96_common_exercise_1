"""
Abstract base class for all inter-agent communication modules.

Contract
--------
  Input  : features  — torch.Tensor  (batch, n_agents, hidden_dim)
  Output : torch.Tensor              (batch, n_agents, hidden_dim)

The module is stateless in the first version.  Future implementations may
introduce per-episode hidden state (recurrent communication); they should
override forward() and document the state management explicitly.

Extension guide
---------------
To add a new communication method:
  1. Subclass BaseCommunicationModule.
  2. Implement forward(self, features) → features (same shape).
  3. Add an elif branch in communication/factory.py.
  4. Expose it in communication/__init__.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseCommunicationModule(nn.Module, ABC):
    """Abstract base for inter-agent communication modules.

    Subclasses must implement forward(features) and may override is_identity.
    """

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply inter-agent communication.

        Parameters
        ----------
        features : torch.Tensor  shape (batch, n_agents, hidden_dim)
            Per-agent encoded features before the action head.

        Returns
        -------
        torch.Tensor  shape (batch, n_agents, hidden_dim)
            Communication-enhanced per-agent features.
        """
        ...

    @property
    def is_identity(self) -> bool:
        """True if this module is a no-op.

        Used by CommActorWrapper to skip the reshape overhead when
        comm_method='none'.  Subclasses that are not no-ops must return False.
        """
        return False
