#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
MAX_JOBS="${MAX_JOBS:-16}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
EPISODES="${EPISODES:-500}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v parallel >/dev/null 2>&1; then
  echo "GNU Parallel is required. Install it with: sudo apt-get install parallel" >&2
  exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required. Install it with: sudo apt-get install util-linux" >&2
  exit 1
fi

LOGICAL_CPUS="$(nproc)"
REQUIRED_CPUS="$((MAX_JOBS * THREADS_PER_JOB))"
if ((LOGICAL_CPUS < REQUIRED_CPUS)); then
  echo "Need at least $REQUIRED_CPUS logical CPUs, but nproc reports $LOGICAL_CPUS." >&2
  exit 1
fi

"$PYTHON_EXE" -c "import torch, sklearn, pandas, numpy"

# The completed baseline is intentionally omitted:
# bes_capacity_kwh + heating_mean + nsl_mean
#
# Format: label|feature_1|feature_2|feature_3
VARIANTS=(
  "B_hvac_nslmean|bes_capacity_kwh|hvac_total_kw|nsl_mean"
  "C_heatmax_nslmean|bes_capacity_kwh|heating_max|nsl_mean"
  "D_heatmean_nslmax|bes_capacity_kwh|heating_mean|nsl_max"
  "E_hvac_nslmax|bes_capacity_kwh|hvac_total_kw|nsl_max"
  "F_heatmax_nslmax|bes_capacity_kwh|heating_max|nsl_max"
  "G_heatmean_comfort|bes_capacity_kwh|heating_mean|comfort_lower_excess_mean"
  "H_heatmean_tempstd|bes_capacity_kwh|heating_mean|indoor_temp_std"
  "I_no_capacity_comfort|heating_mean|nsl_mean|comfort_lower_excess_mean"
)
SEEDS=(0 1 2)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/experiment_queue_logs/3f_shortlist_$STAMP"
JOB_FILE="$LOG_DIR/jobs.tsv"
JOB_LOG="$LOG_DIR/parallel_joblog.tsv"
mkdir -p "$LOG_DIR"
: >"$JOB_FILE"

JOB_COUNT=0
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r label f1 f2 f3 <<<"$variant"
    printf '%s\t%s\t%s\t%s\t%s\n' "$seed" "$label" "$f1" "$f2" "$f3" >>"$JOB_FILE"
    JOB_COUNT="$((JOB_COUNT + 1))"
  done
done

EXPECTED_JOBS="$((${#VARIANTS[@]} * ${#SEEDS[@]}))"
if ((JOB_COUNT != EXPECTED_JOBS)); then
  echo "Expected $EXPECTED_JOBS jobs, generated $JOB_COUNT." >&2
  exit 1
fi

run_one() {
  local seed="$1"
  local label="$2"
  local f1="$3"
  local f2="$4"
  local f3="$5"
  local slot="$6"

  local core_start="$(((slot - 1) * THREADS_PER_JOB))"
  local core_end="$((core_start + THREADS_PER_JOB - 1))"
  local run_name="mappo_grouped_tarmac_hybrid_agglomerative_3f_${label}_vt_500_seed${seed}"
  local save_dir="$REPO_DIR/results/$run_name"
  local stdout_path="$LOG_DIR/$run_name.stdout.log"
  local stderr_path="$LOG_DIR/$run_name.stderr.log"

  if [[ -f "$save_dir/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] $run_name"
    return 0
  fi

  echo "[start] slot=$slot cores=$core_start-$core_end $run_name"

  local -a env_args=(
    "PYTHONUNBUFFERED=1"
    "OMP_NUM_THREADS=$THREADS_PER_JOB"
    "MKL_NUM_THREADS=$THREADS_PER_JOB"
    "OPENBLAS_NUM_THREADS=$THREADS_PER_JOB"
    "NUMEXPR_NUM_THREADS=$THREADS_PER_JOB"
  )
  if [[ "$USE_GPU" != "1" ]]; then
    env_args+=("CUDA_VISIBLE_DEVICES=")
  fi

  if taskset -c "$core_start-$core_end" env "${env_args[@]}" \
    "$PYTHON_EXE" -m mappo_grouped_tarmac_hybrid_grouping.train \
      --climate VT \
      --n_episodes "$EPISODES" \
      --train_month 1 \
      --test_month 2 \
      --seed "$seed" \
      --group_k_candidates 4 5 \
      --cluster_seed 0 \
      --cluster_retries 10 \
      --grouping_method agglomerative \
      --grouping_feature_columns "$f1" "$f2" "$f3" \
      --comm_fusion_mode linear \
      --wandb_name "$run_name" \
      --save_dir "$save_dir" \
      >"$stdout_path" 2>"$stderr_path"; then
    echo "[done]  $run_name"
  else
    local exit_code=$?
    echo "[fail]  exit=$exit_code $run_name stderr=$stderr_path" >&2
    return "$exit_code"
  fi
}

export -f run_one
export PYTHON_EXE THREADS_PER_JOB EPISODES USE_GPU FORCE REPO_DIR LOG_DIR

WAVES="$(((JOB_COUNT + MAX_JOBS - 1) / MAX_JOBS))"

echo "Repository: $REPO_DIR"
echo "Feature combinations: ${#VARIANTS[@]} (completed baseline excluded)"
echo "Seeds: ${SEEDS[*]}"
echo "Total jobs: $JOB_COUNT"
echo "Concurrent jobs: $MAX_JOBS"
echo "CPU cores per job: $THREADS_PER_JOB"
echo "Logical CPUs available: $LOGICAL_CPUS"
echo "Queue waves: $WAVES"
echo "GPU enabled: $USE_GPU"
echo "Logs: $LOG_DIR"
echo
echo "Previous 500-episode runs took about 8.1 hours per job."
echo "With $WAVES queue waves, the ideal lower-bound estimate is about 16.2 hours."
echo "Plan for roughly 18-24 hours because concurrent simulation and I/O can add overhead."
echo "GNU Parallel will display job-level progress and ETA below."
echo

parallel \
  --jobs "$MAX_JOBS" \
  --bar \
  --eta \
  --joblog "$JOB_LOG" \
  --colsep '\t' \
  run_one '{1}' '{2}' '{3}' '{4}' '{5}' '{%}' \
  :::: "$JOB_FILE"

FAILED_JOBS="$(awk 'NR > 1 && $7 != 0 {count++} END {print count + 0}' "$JOB_LOG")"
if ((FAILED_JOBS > 0)); then
  echo "$FAILED_JOBS jobs failed. Inspect $JOB_LOG and the corresponding stderr logs." >&2
  exit 1
fi

echo "All jobs completed successfully."
echo "GNU Parallel job log: $JOB_LOG"
