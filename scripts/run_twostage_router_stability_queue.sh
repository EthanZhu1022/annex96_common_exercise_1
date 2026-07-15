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
LOG_FILE="$LOG_DIR/twostage_router_stability_queue_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Repository: $REPO_DIR"
echo "Python: $PYTHON_EXE"
echo "Queue log: $LOG_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

run_exp() {
  local name="$1"
  local module="$2"
  local save_dir="$3"
  shift 3

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

  "$PYTHON_EXE" -m "$module" "$@"
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    echo "Completed: $name at $(date '+%Y-%m-%d %H:%M:%S')"
    completed+=("$name")
  else
    echo "FAILED: $name exit_code=$exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
    failed+=("$name exit_code=$exit_code")
  fi
  return $exit_code
}

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
)

run_tarmac_3f() {
  local seed="$1"
  local suffix="seed${seed}"
  local save_dir="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_${suffix}"
  run_exp \
    "tarmac_hybrid_agglomerative_3f_${suffix}" \
    "mappo_grouped_tarmac_hybrid_grouping.train" \
    "$save_dir" \
    "${common[@]}" \
    --seed "$seed" \
    --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean \
    --comm_fusion_mode linear \
    --wandb_name "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_${suffix}" \
    --save_dir "$save_dir"
}

run_tarmac_5f() {
  local seed="$1"
  local suffix="seed${seed}"
  local save_dir="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_5f_linear_vt_500_${suffix}"
  run_exp \
    "tarmac_hybrid_agglomerative_5f_${suffix}" \
    "mappo_grouped_tarmac_hybrid_grouping.train" \
    "$save_dir" \
    "${common[@]}" \
    --seed "$seed" \
    --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean \
    --comm_fusion_mode linear \
    --wandb_name "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_5f_linear_vt_500_${suffix}" \
    --save_dir "$save_dir"
}

run_twostage() {
  local seed="$1"
  local variant="$2"
  local expert_dir="$3"
  local save_dir="$4"
  shift 4
  run_exp \
    "soft_router_twostage_3f_${variant}_seed${seed}" \
    "mappo_grouped_tarmac_soft_router.train" \
    "$save_dir" \
    "${common[@]}" \
    --seed "$seed" \
    --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean \
    --comm_fusion_mode linear \
    --expert_checkpoint_dir "$expert_dir" \
    --wandb_name "$(basename "$save_dir")" \
    --save_dir "$save_dir" \
    "$@"
}

# 1) Main stability check: does the strong 3f result survive other seeds?
for seed in 0 1 2 3; do
  run_tarmac_3f "$seed" || true
done

# 2) Direct comparison against the older 5f feature set on the same seeds.
for seed in 0 1 2 3; do
  run_tarmac_5f "$seed" || true
done

# 3) Two-stage router: fixed 3f experts first, then router selector.
# Seed 42 reuses the already strong expert run.
run_twostage \
  42 \
  "freeze200_temp05_prior1" \
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final" \
  "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_vt_500_final" \
  --router_temperature 0.5 \
  --router_prior_start 0.5 \
  --router_prior_end 1.0 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.005 \
  --router_freeze_experts_episodes 200 \
  --router_only_lr 1e-4 \
  --router_finetune_lr 5e-5 || true

run_twostage \
  42 \
  "freeze200_temp07_prior07" \
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final" \
  "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp07_prior07_vt_500_final" \
  --router_temperature 0.7 \
  --router_prior_start 0.3 \
  --router_prior_end 0.7 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.01 \
  --router_freeze_experts_episodes 200 \
  --router_only_lr 1e-4 \
  --router_finetune_lr 5e-5 || true

run_twostage \
  42 \
  "routeronly500_temp05_prior1" \
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final" \
  "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_routeronly500_temp05_prior1_vt_500_final" \
  --router_temperature 0.5 \
  --router_prior_start 0.5 \
  --router_prior_end 1.0 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.005 \
  --router_freeze_experts_episodes 500 \
  --router_only_lr 1e-4 \
  --router_finetune_lr 5e-5 || true

run_twostage \
  42 \
  "freeze200_temp05_prior1_no_capacity" \
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final" \
  "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_no_capacity_vt_500_final" \
  --router_temperature 0.5 \
  --router_prior_start 0.5 \
  --router_prior_end 1.0 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.005 \
  --router_freeze_experts_episodes 200 \
  --router_only_lr 1e-4 \
  --router_finetune_lr 5e-5 \
  --no_router_capacity_features || true

# 4) If the seed-0 3f expert has completed above, test whether two-stage
# router behavior is still reasonable outside seed 42.
run_twostage \
  0 \
  "freeze200_temp05_prior1" \
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_seed0" \
  "results/mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_vt_500_seed0" \
  --router_temperature 0.5 \
  --router_prior_start 0.5 \
  --router_prior_end 1.0 \
  --router_warmup_episodes 150 \
  --router_entropy_scale 0.005 \
  --router_freeze_experts_episodes 200 \
  --router_only_lr 1e-4 \
  --router_finetune_lr 5e-5 || true

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
