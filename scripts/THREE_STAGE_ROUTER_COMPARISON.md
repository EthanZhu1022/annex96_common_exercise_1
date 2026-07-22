# Three-stage router architecture comparison

The two launchers answer different initialization questions while keeping the
climate, grouping features, router schedule, and stage-3 learning rates aligned.

| Variant | Stage 1 | New training episodes | TarMAC parameters | Approx. actor forward cost |
|---|---|---:|---:|---:|
| Shared latent | Retrain shared encoder + K heads for 500 episodes | 1500 | One set | 1x |
| Full expert | Load the existing 500-episode grouped checkpoint | 1000 | One shared set, called K+1 times | 5-6x |

## Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_shared_router.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_three_stage_full_expert_router.ps1
```

## Linux

```bash
bash scripts/run_three_stage_shared_router.sh
bash scripts/run_three_stage_full_expert_router.sh
```

Count the pretrained full-expert checkpoint's original 500 episodes when
reporting total environment interactions. Both methods therefore represent
1500 actor-training episodes end to end.

Compare the fixed baseline, the stage-1 shared checkpoint, both final models,
and the intermediate router-only checkpoints. Use the same test month only for
the final comparison; select intermediate checkpoints using a separate
validation window. Report reward, CV-RMSE, NMBE, comfort exceedance, wall-clock
time, and mean/standard deviation across matched seeds.

The shared model's episode-500 checkpoint is kept as
`checkpoints/checkpoint_ep0500_static_actor.pt`. The full-expert model's
episode-500 checkpoint is kept as `checkpoints/checkpoint_ep0500_router_only.pt`.
These snapshots make it possible to separate stage-1 architecture quality from
router and dynamic-fine-tuning quality instead of judging only the final run.
