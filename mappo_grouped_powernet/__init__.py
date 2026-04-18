"""
mappo_grouped_powernet - Grouped MAPPO with PowerNet-style neighbor communication.

This package reuses the grouped communication training pipeline from
`mappo_grouped_comm`, but switches the communication module to a
PowerNet-inspired local-neighbor update inside each group.
"""

