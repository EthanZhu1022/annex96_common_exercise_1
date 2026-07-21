#!/usr/bin/env bash
set -u

PYTHON_EXE="${PYTHON_EXE:-python}"
NO_SKIP_COMPLETED="${NO_SKIP_COMPLETED:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR" || exit 1

LOG_DIR="$REPO_DIR/experiment_queue_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/twostage_router_followup_queue_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Repository: $REPO_DIR"
echo "Python: $PYTHON_EXE"
echo "Queue log: $LOG_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

completed=()
skipped=()
failed=()

common=(
  --climate VT
  --n_episodes 500
  --train_month 1
  --test_month 2
  --group_k_candidates 4 5
  --cluster_seed 0
  --cluster_retries 10
  --grouping_method agglomerative
  --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean
  --comm_fusion_mode linear
)

run_exp() {
  local name="$1"
  local save_dir="$2"
  shift 2

  echo ""
  echo "============================================================"
  echo "$name"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "SaveDir: $save_dir"
  echo "============================================================"

  if [[ "$NO_SKIP_COMPLETED" != "1" && -f "$save_dir/latest_metrics.json" ]]; then
    echo "Skipping completed experiment: $save_dir/latest_metrics.json"
    skipped+=("$name")
    return 0
  fi

  "$PYTHON_EXE" -m mappo_grouped_tarmac_soft_router.train "$@"
  exit_code=$?

  if [[ $exit_code -eq 0 ]]; then
    echo "Completed: $name at $(date '+%Y-%m-%d %H:%M:%S')"
    completed+=("$name")
  else
    echo "FAILED: $name exit_code=$exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
    failed+=("$name exit_code=$exit_code")
  fi
}

run_routeronly500() {
  local seed="$1"
  local expert_dir="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed${seed}"
  local save_dir="results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_routeronly500_temp05_prior1_vt_500_seed${seed}"
  run_exp \
    "twostage_routeronly500_temp05_prior1_seed${seed}" \
    "$save_dir" \
    "${common[@]}" \
    --seed "$seed" \
    --expert_checkpoint_dir "$expert_dir" \
    --router_temperature 0.5 \
    --router_prior_start 0.5 \
    --router_prior_end 1.0 \
    --router_warmup_episodes 150 \
    --router_entropy_scale 0.005 \
    --router_freeze_experts_episodes 500 \
    --router_only_lr 1e-4 \
    --router_finetune_lr 5e-5 \
    --wandb_name "$(basename "$save_dir")" \
    --save_dir "$save_dir"
}

run_no_capacity_freeze200() {
  local seed="$1"
  local expert_dir="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed${seed}"
  local save_dir="results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_no_capacity_vt_500_seed${seed}"
  run_exp \
    "twostage_freeze200_temp05_prior1_no_capacity_seed${seed}" \
    "$save_dir" \
    "${common[@]}" \
    --seed "$seed" \
    --expert_checkpoint_dir "$expert_dir" \
    --router_temperature 0.5 \
    --router_prior_start 0.5 \
    --router_prior_end 1.0 \
    --router_warmup_episodes 150 \
    --router_entropy_scale 0.005 \
    --router_freeze_experts_episodes 200 \
    --router_only_lr 1e-4 \
    --router_finetune_lr 5e-5 \
    --no_router_capacity_features \
    --wandb_name "$(basename "$save_dir")" \
    --save_dir "$save_dir"
}

for seed in 0 1 2 3; do
  run_routeronly500 "$seed"
done

for seed in 0 1 2 3; do
  run_no_capacity_freeze200 "$seed"
done

echo ""
echo "================ Queue Summary ================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Completed: ${#completed[@]}"
printf '  OK   %s\n' "${completed[@]}"
echo "Skipped: ${#skipped[@]}"
printf '  SKIP %s\n' "${skipped[@]}"
echo "Failed: ${#failed[@]}"
printf '  FAIL %s\n' "${failed[@]}"
echo "Log: $LOG_FILE"

if [[ ${#failed[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
