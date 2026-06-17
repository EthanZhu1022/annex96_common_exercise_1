"""
mappo_grouped_tarmac_soft_router - Grouped MAPPO with hybrid TarMAC
communication and state-conditioned soft actor routing.

The training/evaluation pipeline mirrors `mappo_grouped_tarmac_hybrid`.
The experimental difference is that the grouped action heads are combined
with a soft router instead of being fixed by the offline building group.
"""

