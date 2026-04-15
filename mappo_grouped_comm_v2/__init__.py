"""
mappo_grouped_comm_v2 - Experimental grouped MAPPO with global communication.

This package keeps the grouped-actor / shared-critic structure from
`mappo_grouped`, but changes the actor path to:

  per-group encoder -> global communication over all agents -> per-group action head

The communication module is global across groups, while action heads remain
group-specific.
"""
