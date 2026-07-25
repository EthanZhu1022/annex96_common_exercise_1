# Policy-induced SOC regrouping

This package implements a separate two-stage experiment:

1. Load the final January 3-feature grouped TarMAC checkpoint and run its actor
   deterministically over the January window.
2. Compute per-building electrical-storage SOC statistics, recluster the
   buildings, train a new grouped TarMAC model from scratch on January, and
   evaluate it on February.

Two regrouping variants are supported:

- `soc6f`: `soc_mean`, `soc_std`, `soc_q10`, `soc_low_fraction`,
  `soc_high_fraction`, `soc_daily_range_mean`.
- `energy4f`: `bes_capacity_kwh`, `soc_q10`, `heating_mean`, `nsl_mean`.

The source checkpoint is used only to generate the behavior-derived SOC
features. Its actor, critic, and optimizer weights are never loaded into the
newly grouped models.

Run both variants sequentially from the repository root:

```bash
bash scripts/run_tarmac_soc_regrouping_two_stage.sh
```

The queue collects the deterministic source SOC trajectory once, then trains
each regrouping variant with seeds `42`, `0`, and `1` (six fresh models total).

Stage 1 writes:

- `soc_hourly_trajectory.csv`
- `soc_statistics.csv`
- `soc_collection_metadata.json`

The 744 SOC samples include the reset state at January hour 0 followed by 743
deterministic policy transitions, which is the episode convention used by the
current CityLearn environment for the inclusive `0..743` monthly window.

Each stage-2 result directory contains the normal grouped TarMAC checkpoint and
February test reports, plus the SOC-based cluster assignment and a copied
`source_soc_statistics.csv` and `source_soc_collection_metadata.json`.
