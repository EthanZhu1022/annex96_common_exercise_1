"""
mappo_grouped_comm — Grouped-actor MAPPO with inter-actor communication
=======================================================================

Extends mappo_grouped with an inter-actor communication module injected
between the actor's base encoder and its action distribution head.

Communication is abstracted behind a pluggable interface so methods can be
swapped without touching the training loop.  Currently supported:

  comm_method='none'    — identity (exact mappo_grouped baseline)
  comm_method='commnet' — CommNet (Sukhbaatar et al., NeurIPS 2016)

Key components
--------------
cluster.py              — K-means clustering (identical to mappo_grouped).
env.py                  — Re-exports CityLearnMAPPOEnv (identical to mappo_grouped).
communication/          — Pluggable communication abstraction.
  base.py               — BaseCommunicationModule (abstract).
  none.py               — NoCommunicationModule (identity, no-op).
  commnet.py            — CommNetCommunicationModule.
  factory.py            — build_communication_module() factory.
actor_wrapper.py        — CommActorWrapper: drops in for R_Actor with comm injected.
train.py                — Grouped training loop with comm integration.

Run
---
  python -m mappo_grouped_comm.train --climate VT
  python -m mappo_grouped_comm.train --climate VT --comm_method commnet
  python -m mappo_grouped_comm.train --climate VT --comm_method commnet --comm_rounds 2
  python -m mappo_grouped_comm.train --test_only --checkpoint_dir results/mappo_grouped_comm_vt
"""
