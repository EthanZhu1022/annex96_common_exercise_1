"""
No-communication identity module.

When comm_method='none', CommActorWrapper uses this module.  The forward pass
returns features unchanged (is_identity=True shortcuts the reshape overhead),
so the training loop degrades exactly to mappo_grouped behavior.

No learnable parameters are added.
"""

from __future__ import annotations

import torch

from .base import BaseCommunicationModule


class NoCommunicationModule(BaseCommunicationModule):
    """Identity communication — passes features through unchanged.

    No extra parameters; checkpoint state_dict is empty for this module.
    """

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features

    @property
    def is_identity(self) -> bool:
        return True
