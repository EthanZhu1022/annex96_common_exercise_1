# Stable full-expert and router-GRU experiments

The comparison script runs two matched variants:

1. `full_expert_stable_heads`: feed-forward router control.
2. `full_expert_routergru_stable_heads`: temporal GRU router.

Both variants load the same fixed-3f expert checkpoint and use:

- 500 router-only episodes;
- 500 dynamic actor episodes;
- a frozen router throughout all 500 dynamic actor episodes;
- frozen expert encoders and global TarMAC in stage 3;
- action-head-only adaptation at learning rate `1e-5`;
- temperature `0.5` and the original `0.5 -> 1.0` static-prior schedule.

The GRU is used only by the router. Experts remain feed-forward. One GRU
hidden state is maintained per building in the existing rollout state buffer,
and PPO updates use aligned contiguous time chunks instead of shuffled
timesteps.

Run all three seeds sequentially:

```bash
bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Run a seed-42 smoke/comparison first:

```bash
SEEDS="42" bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Run a custom seed list:

```bash
SEEDS="3 4 5" bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Run only the GRU variant:

```bash
VARIANTS="gru" bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Run only the matched feed-forward control:

```bash
VARIANTS="ff" bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Set `PYTHON_EXE` or `EXPERT_DIR` when the environment or checkpoint path is
different:

```bash
PYTHON_EXE=python3 \
EXPERT_DIR=results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final \
bash scripts/run_soft_router_full_expert_stable_and_gru.sh
```

Completed result directories are skipped. If `checkpoint.pt` exists without
`latest_metrics.json`, the script automatically restores the model, router,
critic, optimizers, and episode number and continues from that checkpoint.
Any failed run stops the queue.
