"""
mappo_grouped_powernet_global - Grouped MAPPO with global PowerNet-style communication.

This package keeps the grouped-encoder / grouped-action-head structure from
`mappo_grouped_comm_v2`, but replaces CommNet with a PowerNet-inspired
neighbor communication module over the full set of agents.

The k-means grouping is unchanged. The only intended experimental difference
from the intra-group PowerNet variant is that the communication graph is built
over all agents instead of being restricted to each group independently.
"""

