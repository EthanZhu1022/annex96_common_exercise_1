#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

PYTHON_EXE="${PYTHON_EXE:-python}"
EXPERT_DIR="${EXPERT_DIR:-results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final}"
SEEDS_TEXT="${SEEDS:-42 0 1}"
read -r -a RUN_SEEDS <<< "$SEEDS_TEXT"
VARIANTS_TEXT="${VARIANTS:-ff gru}"
read -r -a RUN_VARIANTS <<< "$VARIANTS_TEXT"

if [[ ! -f "${EXPERT_DIR}/checkpoint.pt" ]]; then
  echo "Missing fixed-3f expert checkpoint: ${EXPERT_DIR}/checkpoint.pt" >&2
  exit 1
fi

run_variant() {
  local variant="$1"
  local seed="$2"
  local save_dir
  local -a extra_args=()

  if [[ "$variant" == "ff" ]]; then
    save_dir="results/mappo_grouped_tarmac_soft_router_full_expert_stable_heads_3f_vt_seed${seed}"
  elif [[ "$variant" == "gru" ]]; then
    save_dir="results/mappo_grouped_tarmac_soft_router_full_expert_routergru_stable_heads_3f_vt_seed${seed}"
    extra_args+=(--router_gru)
  else
    echo "Unknown variant: $variant" >&2
    exit 1
  fi

  if [[ -f "${save_dir}/checkpoint.pt" && -f "${save_dir}/latest_metrics.json" ]]; then
    echo "[skip] completed ${variant} seed=${seed}: ${save_dir}"
    return
  fi
  if [[ -f "${save_dir}/checkpoint.pt" ]]; then
    echo "[resume] ${variant} seed=${seed}: ${save_dir}/checkpoint.pt"
    extra_args+=(--resume_checkpoint "${save_dir}/checkpoint.pt")
  fi

  echo "[run] ${variant} seed=${seed}: ${save_dir}"
  "$PYTHON_EXE" -m mappo_grouped_tarmac_soft_router.train \
    --climate VT \
    --seed "$seed" \
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
    --dynamic_actor_router_freeze_episodes 500 \
    --dynamic_actor_update_scope heads \
    --expert_checkpoint_dir "$EXPERT_DIR" \
    --router_temperature 0.5 \
    --router_prior_start 0.5 \
    --router_warmup_episodes 150 \
    --router_entropy_scale 0.05 \
    --router_balance_coef 0.01 \
    --router_only_lr 1e-4 \
    --dynamic_actor_lr 1e-5 \
    --router_finetune_lr 2e-5 \
    --checkpoint_keep_every 50 \
    --wandb_name "$(basename "$save_dir")" \
    --save_dir "$save_dir" \
    "${extra_args[@]}"
}

# By default the matched feed-forward control runs first, then the GRU router.
# set -e guarantees that a failed run stops the queue before the next experiment.
for variant in "${RUN_VARIANTS[@]}"; do
  for seed in "${RUN_SEEDS[@]}"; do
    run_variant "$variant" "$seed"
  done
done

echo "All stable full-expert and router-GRU experiments completed."
