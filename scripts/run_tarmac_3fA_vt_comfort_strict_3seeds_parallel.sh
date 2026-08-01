#!/usr/bin/env bash
set -Eeuo pipefail

THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
EPISODES="${EPISODES:-500}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"

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

"$PYTHON_EXE" -c "import torch, sklearn, pandas, numpy"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/experiment_queue_logs/3fA_vt_comfort_strict_$STAMP"
mkdir -p "$LOG_DIR"

AVAILABLE_CPU_IDS=""
if command -v taskset >/dev/null 2>&1; then
  AVAILABLE_CPU_IDS="$("$PYTHON_EXE" -c \
    'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
fi

echo "VT/TX 3fA strict comfort reward run"
echo "VT features: bes_capacity_kwh heating_mean nsl_mean"
echo "TX features: bes_capacity_kwh cooling_mean nsl_mean"
echo "VT train/test months: 1/2, grouping feature month: 1"
echo "TX train/test months: 8/9, grouping feature month: 8"
echo "Seeds: 0 1 2 for each climate, launched concurrently"
echo "Episodes per seed: $EPISODES"
echo "GPU enabled: $USE_GPU"
echo "Threads per job: $THREADS_PER_JOB"
echo "Reward weights are read from annex96_rewards/ce1.py"
echo "Expected current comfort weights: weight_comfort=1.5, comfort_binary_weight=3.0, comfort_degree_weight=1.0"
echo "Logs: $LOG_DIR"
echo

pids=()
failed=0

run_job() {
  local climate="$1"
  local seed="$2"
  local slot="$3"
  local train_month="$4"
  local test_month="$5"
  local feature_month="$6"
  local feature_2="$7"
  local run_name="$8"
  local save_dir="$REPO_DIR/results/$run_name"
  local stdout_path="$LOG_DIR/$run_name.stdout.log"
  local stderr_path="$LOG_DIR/$run_name.stderr.log"
  local cpu_spec=""

  if [[ -f "$save_dir/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] completed climate=$climate seed=$seed $run_name"
    return 0
  fi

  if [[ -n "$AVAILABLE_CPU_IDS" ]]; then
    read -r -a cpu_ids <<<"$AVAILABLE_CPU_IDS"
    local required_index=$((slot * THREADS_PER_JOB + THREADS_PER_JOB - 1))
    if ((${#cpu_ids[@]} <= required_index)); then
      echo "Need at least $((required_index + 1)) allowed CPU IDs, but affinity exposes ${#cpu_ids[@]}." >&2
      return 1
    fi
    for ((offset = 0; offset < THREADS_PER_JOB; offset++)); do
      local cpu_id="${cpu_ids[$((slot * THREADS_PER_JOB + offset))]}"
      cpu_spec+="${cpu_spec:+,}${cpu_id}"
    done
  fi

  echo "[start] climate=$climate seed=$seed slot=$slot cpus=${cpu_spec:-all} $run_name"
  echo "        stdout=$stdout_path"
  echo "        stderr=$stderr_path"

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

  local -a cmd=(
    "$PYTHON_EXE" -m mappo_grouped_tarmac_hybrid_grouping.train
    --climate "$climate"
    --n_episodes "$EPISODES"
    --train_month "$train_month"
    --test_month "$test_month"
    --grouping_feature_month "$feature_month"
    --seed "$seed"
    --group_k_candidates 4 5
    --cluster_seed 0
    --cluster_retries 10
    --grouping_method agglomerative
    --grouping_feature_columns bes_capacity_kwh "$feature_2" nsl_mean
    --comm_fusion_mode linear
    --wandb_name "$run_name"
    --save_dir "$save_dir"
  )

  if [[ -n "$cpu_spec" ]]; then
    taskset -c "$cpu_spec" env "${env_args[@]}" "${cmd[@]}" >"$stdout_path" 2>"$stderr_path"
  else
    env "${env_args[@]}" "${cmd[@]}" >"$stdout_path" 2>"$stderr_path"
  fi

  echo "[done]  climate=$climate seed=$seed $run_name"
}

slot=0
for seed in 0 1 2; do
  run_job \
    "VT" \
    "$seed" \
    "$slot" \
    1 \
    2 \
    1 \
    "heating_mean" \
    "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_comfort15_binary30_degree10_seed${seed}" &
  pids+=("$!")
  slot=$((slot + 1))
done

for seed in 0 1 2; do
  run_job \
    "TX" \
    "$seed" \
    "$slot" \
    8 \
    9 \
    8 \
    "cooling_mean" \
    "mappo_grouped_tarmac_hybrid_agglomerative_capacity_cooling_3f_linear_tx_aug_sep_500_comfort15_binary30_degree10_seed${seed}" &
  pids+=("$!")
  slot=$((slot + 1))
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if ((failed)); then
  echo "One or more seeds failed. Check stderr logs under $LOG_DIR." >&2
  exit 1
fi

echo "All requested seeds finished."
