#!/usr/bin/env bash
set -Eeuo pipefail

THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
EPISODES="${EPISODES:-500}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"
JOB_HOURS_ESTIMATE="${JOB_HOURS_ESTIMATE:-8.1}"

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
LOG_DIR="$REPO_DIR/experiment_queue_logs/cooling_3f_tx_aug_sep_$STAMP"
mkdir -p "$LOG_DIR"

if command -v taskset >/dev/null 2>&1; then
  AVAILABLE_CPU_IDS="$("$PYTHON_EXE" -c \
    'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
  read -r -a AVAILABLE_CPU_ID_ARRAY <<<"$AVAILABLE_CPU_IDS"
  if ((${#AVAILABLE_CPU_ID_ARRAY[@]} < THREADS_PER_JOB)); then
    echo "Need $THREADS_PER_JOB allowed CPU IDs, but affinity exposes ${#AVAILABLE_CPU_ID_ARRAY[@]}." >&2
    exit 1
  fi
  CPU_SPEC=""
  for ((i = 0; i < THREADS_PER_JOB; i++)); do
    CPU_SPEC+="${CPU_SPEC:+,}${AVAILABLE_CPU_ID_ARRAY[$i]}"
  done
else
  CPU_SPEC=""
fi

echo "TX cooling 3f sequential run"
echo "Features: bes_capacity_kwh cooling_mean nsl_mean"
echo "Train month: 8, test month: 9, grouping feature month: 8"
echo "Seeds: 0 1 2"
echo "Episodes per seed: $EPISODES"
echo "GPU enabled: $USE_GPU"
echo "Threads per job: $THREADS_PER_JOB"
echo "Logs: $LOG_DIR"
echo "Planning estimate: about $(awk "BEGIN {printf \"%.1f\", 3 * $JOB_HOURS_ESTIMATE}") hours total."
echo

for SEED in 0 1 2; do
  RUN_NAME="mappo_grouped_tarmac_hybrid_agglomerative_capacity_cooling_3f_linear_tx_aug_sep_500_seed${SEED}"
  SAVE_DIR="$REPO_DIR/results/$RUN_NAME"
  STDOUT_PATH="$LOG_DIR/$RUN_NAME.stdout.log"
  STDERR_PATH="$LOG_DIR/$RUN_NAME.stderr.log"

  if [[ -f "$SAVE_DIR/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] completed seed=$SEED $RUN_NAME"
    continue
  fi

  echo "[start] seed=$SEED $RUN_NAME"
  echo "        stdout=$STDOUT_PATH"
  echo "        stderr=$STDERR_PATH"

  ENV_ARGS=(
    "PYTHONUNBUFFERED=1"
    "OMP_NUM_THREADS=$THREADS_PER_JOB"
    "MKL_NUM_THREADS=$THREADS_PER_JOB"
    "OPENBLAS_NUM_THREADS=$THREADS_PER_JOB"
    "NUMEXPR_NUM_THREADS=$THREADS_PER_JOB"
  )
  if [[ "$USE_GPU" != "1" ]]; then
    ENV_ARGS+=("CUDA_VISIBLE_DEVICES=")
  fi

  CMD=(
    "$PYTHON_EXE" -m mappo_grouped_tarmac_hybrid_grouping.train
    --climate TX
    --n_episodes "$EPISODES"
    --train_month 8
    --test_month 9
    --grouping_feature_month 8
    --seed "$SEED"
    --group_k_candidates 4 5
    --cluster_seed 0
    --cluster_retries 10
    --grouping_method agglomerative
    --grouping_feature_columns bes_capacity_kwh cooling_mean nsl_mean
    --comm_fusion_mode linear
    --wandb_name "$RUN_NAME"
    --save_dir "$SAVE_DIR"
  )

  if [[ -n "$CPU_SPEC" ]]; then
    taskset -c "$CPU_SPEC" env "${ENV_ARGS[@]}" "${CMD[@]}" >"$STDOUT_PATH" 2>"$STDERR_PATH"
  else
    env "${ENV_ARGS[@]}" "${CMD[@]}" >"$STDOUT_PATH" 2>"$STDERR_PATH"
  fi

  echo "[done]  seed=$SEED $RUN_NAME"
  echo
done

echo "All requested seeds finished."
