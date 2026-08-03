#!/usr/bin/env bash
set -Eeuo pipefail

THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
CPU_OFFSET="${CPU_OFFSET:-0}"
EPISODES="${EPISODES:-500}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"
UV_EXE="${UV_EXE:-uv}"

if [[ "$THREADS_PER_JOB" != "2" ]]; then
  echo "This experiment requires exactly two CPU cores per training run." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if ! command -v "$UV_EXE" >/dev/null 2>&1; then
  echo "uv is required but was not found: $UV_EXE" >&2
  exit 2
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required to reserve two non-overlapping CPU cores per run." >&2
  exit 2
fi

"$UV_EXE" run --frozen python -c "import torch, sklearn, pandas, numpy, scipy"

read -r -a CPU_IDS <<<"$("$UV_EXE" run --frozen python -c \
  'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
RUN_COUNT=6
REQUIRED_CORES=$((CPU_OFFSET + RUN_COUNT * THREADS_PER_JOB))
if ((${#CPU_IDS[@]} < REQUIRED_CORES)); then
  echo "Need at least $REQUIRED_CORES allowed CPU cores for CPU_OFFSET=$CPU_OFFSET; only ${#CPU_IDS[@]} are available." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/experiment_queue_logs/balanced_spectral_5f_3fA_3seeds_$STAMP"
mkdir -p "$LOG_DIR"

echo "Balanced Spectral VT comparison"
echo "Variants: 5f primary comparison; 3fA exploratory comparison"
echo "Seeds: 0 1 42"
echo "Runs: $RUN_COUNT concurrent jobs"
echo "CPU cores per run: $THREADS_PER_JOB"
echo "Episodes per run: $EPISODES"
echo "K candidates: 4 5 (same protocol as the historical comparisons)"
echo "Reward weights: NMBE=1 CV-RMSE=1 comfort=0.8 binary=1.3 degree=0.3"
echo "Logs: $LOG_DIR"

run_job() {
  local variant="$1"
  local seed="$2"
  local slot="$3"
  local cpu_spec=""
  local run_name=""
  local -a feature_columns=()

  if [[ "$variant" == "5f" ]]; then
    feature_columns=(
      bes_capacity_kwh
      hvac_total_kw
      heating_mean
      nsl_mean
      comfort_lower_excess_mean
    )
    run_name="mappo_grouped_tarmac_hybrid_balanced_spectral_capacity_load_5f_linear_vt_500_seed${seed}"
  elif [[ "$variant" == "3fA" ]]; then
    feature_columns=(bes_capacity_kwh heating_mean nsl_mean)
    run_name="mappo_grouped_tarmac_hybrid_balanced_spectral_capacity_load_3f_linear_vt_500_seed${seed}"
  else
    echo "Unknown variant: $variant" >&2
    return 2
  fi

  local offset
  for ((offset = 0; offset < THREADS_PER_JOB; offset++)); do
    cpu_spec+="${cpu_spec:+,}${CPU_IDS[$((CPU_OFFSET + slot * THREADS_PER_JOB + offset))]}"
  done

  local save_dir="$REPO_DIR/results/$run_name"
  local stdout_path="$LOG_DIR/$run_name.stdout.log"
  local stderr_path="$LOG_DIR/$run_name.stderr.log"
  if [[ -f "$save_dir/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] completed variant=$variant seed=$seed $run_name"
    return 0
  fi

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

  local -a command=(
    "$UV_EXE" run --frozen python -m mappo_grouped_tarmac_hybrid_grouping.train
    --climate VT
    --n_episodes "$EPISODES"
    --train_month 1
    --test_month 2
    --grouping_feature_month 1
    --seed "$seed"
    --group_k_candidates 4 5
    --cluster_seed 0
    --cluster_retries 10
    --grouping_method balanced_spectral
    --grouping_feature_columns "${feature_columns[@]}"
    --comm_fusion_mode linear
    --weight_nmbe 1.0
    --weight_cv_rmse 1.0
    --weight_comfort 0.8
    --comfort_binary_weight 1.3
    --comfort_degree_weight 0.3
    --wandb_name "$run_name"
    --save_dir "$save_dir"
  )

  echo "[start] variant=$variant seed=$seed cpus=$cpu_spec $run_name"
  echo "        stdout=$stdout_path"
  echo "        stderr=$stderr_path"
  taskset -c "$cpu_spec" env "${env_args[@]}" "${command[@]}" \
    >"$stdout_path" 2>"$stderr_path"
  echo "[done]  variant=$variant seed=$seed $run_name"
}

pids=()
slot=0
for variant in 5f 3fA; do
  for seed in 0 1 42; do
    run_job "$variant" "$seed" "$slot" &
    pids+=("$!")
    slot=$((slot + 1))
  done
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if ((failed)); then
  echo "One or more runs failed. Inspect logs under $LOG_DIR." >&2
  exit 1
fi

echo "All six Balanced Spectral runs completed."
