#!/usr/bin/env bash
set -Eeuo pipefail

THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
CPU_OFFSET="${CPU_OFFSET:-0}"
EPISODES="${EPISODES:-500}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"

if [[ "$THREADS_PER_JOB" != "2" ]]; then
  echo "This experiment is designed for exactly two CPU cores per seed." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [[ -z "${PYTHON_EXE:-}" ]]; then
  if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON_EXE="$REPO_DIR/.venv/bin/python"
  else
    PYTHON_EXE="python3"
  fi
fi

if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required to reserve two non-overlapping CPU cores per seed." >&2
  exit 2
fi

"$PYTHON_EXE" -c "import torch, sklearn, pandas, numpy"

read -r -a CPU_IDS <<<"$("$PYTHON_EXE" -c \
  'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
REQUIRED_CORES=$((CPU_OFFSET + 3 * THREADS_PER_JOB))
if ((${#CPU_IDS[@]} < REQUIRED_CORES)); then
  echo "Need at least $REQUIRED_CORES allowed CPU cores for CPU_OFFSET=$CPU_OFFSET, but only ${#CPU_IDS[@]} are available." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/experiment_queue_logs/vt_3fA_degree_hours_$STAMP"
mkdir -p "$LOG_DIR"

echo "VT 3fA degree-hours reward experiment"
echo "Seeds: 0 1 2 (parallel)"
echo "CPU cores per seed: $THREADS_PER_JOB"
echo "Episodes per seed: $EPISODES"
echo "Features: bes_capacity_kwh heating_mean nsl_mean"
echo "Train/test/grouping months: January/February/January"
echo "Reward: NMBE=5 CV-RMSE=5 comfort=1 binary=1 degree=1.5"
echo "Comfort bounds: official seasonal VT band, 20-24C"
echo "Logs: $LOG_DIR"

run_seed() {
  local seed="$1"
  local slot="$2"
  local cpu_spec=""
  local offset
  for ((offset = 0; offset < THREADS_PER_JOB; offset++)); do
    cpu_spec+="${cpu_spec:+,}${CPU_IDS[$((CPU_OFFSET + slot * THREADS_PER_JOB + offset))]}"
  done

  local run_name="mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_nmbe5_cvrmse5_comfort10_binary10_degree15_seed${seed}"
  local save_dir="$REPO_DIR/results/$run_name"
  local stdout_path="$LOG_DIR/$run_name.stdout.log"
  local stderr_path="$LOG_DIR/$run_name.stderr.log"

  if [[ -f "$save_dir/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] completed seed=$seed $run_name"
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
    "$PYTHON_EXE" -m mappo_grouped_tarmac_hybrid_grouping.train
    --climate VT
    --n_episodes "$EPISODES"
    --train_month 1
    --test_month 2
    --grouping_feature_month 1
    --seed "$seed"
    --group_k_candidates 4 5
    --cluster_seed 0
    --cluster_retries 10
    --grouping_method agglomerative
    --grouping_feature_columns bes_capacity_kwh heating_mean nsl_mean
    --comm_fusion_mode linear
    --weight_nmbe 5.0
    --weight_cv_rmse 5.0
    --weight_comfort 1.0
    --comfort_binary_weight 1.0
    --comfort_degree_weight 1.5
    --wandb_name "$run_name"
    --save_dir "$save_dir"
  )

  echo "[start] seed=$seed cpus=$cpu_spec $run_name"
  taskset -c "$cpu_spec" env "${env_args[@]}" "${command[@]}" \
    >"$stdout_path" 2>"$stderr_path"
  echo "[done]  seed=$seed $run_name"
}

pids=()
for seed in 0 1 2; do
  run_seed "$seed" "$seed" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if ((failed)); then
  echo "One or more seeds failed. Inspect $LOG_DIR." >&2
  exit 1
fi

echo "All three VT seeds completed."
