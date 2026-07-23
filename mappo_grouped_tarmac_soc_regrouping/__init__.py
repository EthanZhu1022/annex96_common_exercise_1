"""
Policy-induced electrical-storage SOC regrouping for grouped TarMAC MAPPO.

The experiment is intentionally split into two stages:

1. Run a pretrained fixed-group TarMAC policy deterministically and extract
   per-building hourly electrical-storage SOC trajectories and statistics.
2. Recluster buildings from those statistics and train a new grouped TarMAC
   policy from scratch.
"""

from .features import ENERGY_4F_FEATURES, SOC_6F_FEATURES

__all__ = ["ENERGY_4F_FEATURES", "SOC_6F_FEATURES"]
