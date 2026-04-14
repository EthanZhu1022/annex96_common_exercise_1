"""
mappo_grouped — Grouped-actor MAPPO for CityLearn Annex96-CE1
==============================================================

Extends the standard MAPPO baseline (mappo_standard) with building clustering:
25 buildings are partitioned into K=4 or K=5 groups by K-means on normalised
[BES_capacity, HVAC_total_capacity] features.  Each group has its own actor
network; all groups share a single centralised critic.

Key components
--------------
cluster.py  — K-means clustering: extracts features, selects K & seed,
              saves CSV/JSON cluster artifacts.
env.py      — Re-exports CityLearnMAPPOEnv from mappo_standard (identical env).
train.py    — Grouped training loop: K actors + 1 shared critic, fully aligned
              with mappo_standard's W&B/checkpoint/test conventions.

Reuses directly from on-policy-main
-------------------------------------
  from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
  from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
  from onpolicy.utils.shared_buffer import SharedReplayBuffer

Run
---
  python -m mappo_grouped.train --climate VT
  python -m mappo_grouped.train --climate TX --group_k_candidates 4 5 --cluster_seed 0
  python -m mappo_grouped.train --climate VT --no_test
  python -m mappo_grouped.train --test_only --checkpoint_dir results/mappo_grouped_vt
"""
