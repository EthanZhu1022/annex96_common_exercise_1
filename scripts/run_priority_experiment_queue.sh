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
LOG_FILE="$LOG_DIR/priority_experiment_queue_${STAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Repository: $REPO_DIR"
echo "Python: $PYTHON_EXE"
echo "Queue log: $LOG_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

names=(
  "soft_router_agglomerative_5f_capacity_router"
  "soft_router_agglomerative_5f_capacity_router_prior05"
  "soft_router_agglomerative_5f_capacity_router_temp05"
  "soft_router_agglomerative_5f_no_capacity_router"
  "tarmac_hybrid_agglomerative_3f"
  "tarmac_hybrid_agglomerative_4f"
  "powernet_global_agglomerative_3f"
  "powernet_global_agglomerative_4f"
)

modules=(
  "mappo_grouped_tarmac_soft_router.train"
  "mappo_grouped_tarmac_soft_router.train"
  "mappo_grouped_tarmac_soft_router.train"
  "mappo_grouped_tarmac_soft_router.train"
  "mappo_grouped_tarmac_hybrid_grouping.train"
  "mappo_grouped_tarmac_hybrid_grouping.train"
  "mappo_grouped_powernet_global_grouping.train"
  "mappo_grouped_powernet_global_grouping.train"
)

save_dirs=(
  "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_vt_500_final"
  "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_prior05_vt_500_final"
  "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_temp05_vt_500_final"
  "results/mappo_grouped_tarmac_soft_router_agglomerative_5f_no_capacity_router_vt_500_final"
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
  "results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final"
  "results/mappo_grouped_powernet_global_agglomerative_capacity_load_3f_vt_500_final"
  "results/mappo_grouped_powernet_global_agglomerative_capacity_load_4f_vt_500_final"
)

common_args=(
  --climate VT
  --n_episodes 500
  --train_month 1
  --test_month 2
  --seed 42
  --group_k_candidates 4 5
  --cluster_seed 0
  --cluster_retries 10
  --grouping_method agglomerative
)

args_0=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean
  --comm_fusion_mode linear
  --router_temperature 0.7
  --router_prior_end 0.7
  --router_warmup_episodes 100
  --router_entropy_scale 0.02
  --wandb_name mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_vt_500_final
  --save_dir "${save_dirs[0]}"
)

args_1=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean
  --comm_fusion_mode linear
  --router_temperature 0.7
  --router_prior_end 0.5
  --router_warmup_episodes 100
  --router_entropy_scale 0.02
  --wandb_name mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_prior05_vt_500_final
  --save_dir "${save_dirs[1]}"
)

args_2=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean
  --comm_fusion_mode linear
  --router_temperature 0.5
  --router_prior_end 0.7
  --router_warmup_episodes 100
  --router_entropy_scale 0.02
  --wandb_name mappo_grouped_tarmac_soft_router_agglomerative_5f_capacity_router_temp05_vt_500_final
  --save_dir "${save_dirs[2]}"
)

args_3=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean comfort_lower_excess_mean
  --comm_fusion_mode linear
  --router_temperature 0.7
  --router_prior_end 0.7
  --router_warmup_episodes 100
  --router_entropy_scale 0.02
  --no_router_capacity_features
  --wandb_name mappo_grouped_tarmac_soft_router_agglomerative_5f_no_capacity_router_vt_500_final
  --save_dir "${save_dirs[3]}"
)

args_4=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean
  --comm_fusion_mode linear
  --wandb_name mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final
  --save_dir "${save_dirs[4]}"
)

args_5=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean
  --comm_fusion_mode linear
  --wandb_name mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final
  --save_dir "${save_dirs[5]}"
)

args_6=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean
  --wandb_name mappo_grouped_powernet_global_agglomerative_capacity_load_3f_vt_500_final
  --save_dir "${save_dirs[6]}"
)

args_7=(
  "${common_args[@]}"
  --grouping_feature_columns bes_capacity_kwh hvac_total_kw heating_mean nsl_mean
  --wandb_name mappo_grouped_powernet_global_agglomerative_capacity_load_4f_vt_500_final
  --save_dir "${save_dirs[7]}"
)

completed=()
skipped=()
failed=()

for i in "${!names[@]}"; do
  name="${names[$i]}"
  module="${modules[$i]}"
  save_dir="${save_dirs[$i]}"
  metrics_path="$save_dir/latest_metrics.json"
  args_var="args_$i[@]"
  args=("${!args_var}")

  echo ""
  echo "============================================================"
  echo "[$((i + 1))/${#names[@]}] $name"
  echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "SaveDir: $save_dir"
  echo "============================================================"

  if [[ "$NO_SKIP_COMPLETED" != "1" && -f "$metrics_path" ]]; then
    echo "Skipping completed experiment: $metrics_path"
    skipped+=("$name")
    continue
  fi

  "$PYTHON_EXE" -m "$module" "${args[@]}"
  exit_code=$?

  if [[ $exit_code -eq 0 ]]; then
    echo "Completed: $name at $(date '+%Y-%m-%d %H:%M:%S')"
    completed+=("$name")
  else
    echo "FAILED: $name exit_code=$exit_code at $(date '+%Y-%m-%d %H:%M:%S')"
    failed+=("$name exit_code=$exit_code")
  fi
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
