#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_EXE="${PYTHON_EXE:-python}"
EXPERT_DIR="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
SAVE_DIR="results/mappo_grouped_tarmac_soft_router_three_stage_full_expert_3f_vt_seed42"

"$PYTHON_EXE" -m mappo_grouped_tarmac_soft_router.train \
  --climate VT \
  --seed 42 \
  --train_month 1 \
  --test_month 2 \
  --group_k_candidates 4 5 \
  --cluster_seed 0 \
  --cluster_retries 10 \
  --grouping_method agglomerative \
  --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean \
  --comm_fusion_mode linear \
  --training_schedule pretrained_full_expert \
  --static_actor_episodes 0 \
  --router_only_episodes 500 \
  --dynamic_actor_episodes 500 \
  --dynamic_actor_router_freeze_episodes 100 \
  --expert_checkpoint_dir "$EXPERT_DIR" \
  --router_temperature 0.5 \
  --router_prior_start 0.5 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.05 \
  --router_balance_coef 0.01 \
  --router_only_lr 1e-4 \
  --dynamic_actor_lr 3e-5 \
  --router_finetune_lr 2e-5 \
  --checkpoint_keep_every 50 \
  --wandb_name "$(basename "$SAVE_DIR")" \
  --save_dir "$SAVE_DIR"
