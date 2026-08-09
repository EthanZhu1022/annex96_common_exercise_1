#!/usr/bin/env bash
set -Eeuo pipefail

# Eight missing fair-comparison runs for Report 2:
#   Batch 1: agglomerative 4F, seeds 0/1/2/3
#   Batch 2: K-means 5F seeds 0/1, GMM 5F seeds 0/1
# Four jobs run concurrently in each batch. Each job is pinned to three CPUs.
# When called normally, this script relaunches itself in a detached screen.

THREADS_PER_JOB=3
JOBS_PER_BATCH=4
EPISODES=500

UV_EXE="${UV_EXE:-uv}"
CPU_OFFSET="${CPU_OFFSET:-0}"
USE_GPU="${USE_GPU:-0}"
FORCE="${FORCE:-0}"
SCREEN_NAME="${SCREEN_NAME:-report2_fair_seeds}"

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

if [[ "${1:-}" != "--inside-screen" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "GNU screen is required. Install it first, for example: sudo apt install screen" >&2
    exit 2
  fi
  if screen -list 2>/dev/null | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "A screen session named '$SCREEN_NAME' already exists." >&2
    echo "Attach with: screen -r $SCREEN_NAME" >&2
    exit 2
  fi

  RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  export RUN_STAMP UV_EXE CPU_OFFSET USE_GPU FORCE SCREEN_NAME

  BOOT_LOG_DIR="$REPO_DIR/experiment_queue_logs/report2_fair_seeds_${RUN_STAMP}"
  mkdir -p "$BOOT_LOG_DIR"
  SCREEN_LOG="$BOOT_LOG_DIR/screen.log"
  screen -L -Logfile "$SCREEN_LOG" -dmS "$SCREEN_NAME" \
    bash "$SCRIPT_PATH" --inside-screen
  sleep 2
  if ! screen -list 2>/dev/null | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "The screen session exited during startup." >&2
    echo "Startup log: $SCREEN_LOG" >&2
    if [[ -f "$SCREEN_LOG" ]]; then
      tail -n 80 "$SCREEN_LOG" >&2
    fi
    exit 1
  fi
  echo "Started detached screen session: $SCREEN_NAME"
  echo "Attach: screen -r $SCREEN_NAME"
  echo "List:   screen -ls"
  echo "Logs:   $BOOT_LOG_DIR/"
  exit 0
fi

if ! command -v "$UV_EXE" >/dev/null 2>&1; then
  echo "uv is required but was not found: $UV_EXE" >&2
  exit 2
fi
if ! command -v taskset >/dev/null 2>&1; then
  echo "taskset is required to pin three CPU cores to each experiment." >&2
  exit 2
fi

"$UV_EXE" run --frozen python -c "import torch, sklearn, pandas, numpy, scipy"

read -r -a CPU_IDS <<<"$("$UV_EXE" run --frozen python -c \
  'import os; print(" ".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0))))')"
REQUIRED_CORES=$((CPU_OFFSET + THREADS_PER_JOB * JOBS_PER_BATCH))
if ((${#CPU_IDS[@]} < REQUIRED_CORES)); then
  echo "Need at least $REQUIRED_CORES allowed CPU cores for CPU_OFFSET=$CPU_OFFSET; only ${#CPU_IDS[@]} are available." >&2
  exit 2
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$REPO_DIR/experiment_queue_logs/report2_fair_seeds_${RUN_STAMP}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "Report 2 missing-seed comparison"
echo "Repository: $REPO_DIR"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Batches: 2; concurrent jobs per batch: $JOBS_PER_BATCH"
echo "CPU cores per job: $THREADS_PER_JOB; total active cores: $((THREADS_PER_JOB * JOBS_PER_BATCH))"
echo "Allowed CPUs: ${CPU_IDS[*]}"
echo "GPU enabled: $USE_GPU"
echo "Episodes: $EPISODES"
echo "Seeds: 4F=0,1,2,3; K-means=0,1; GMM=0,1"
echo "Common protocol: VT, train month 1, test month 2, feature month 1"
echo "Communication: TarMAC Hybrid linear, one global round"
echo "Reward weights: NMBE=1.0 CV-RMSE=1.0 comfort=0.8 binary=1.3 degree=0.3"
echo "Logs: $LOG_DIR"

run_job() {
  local method="$1"
  local seed="$2"
  local feature_set="$3"
  local slot="$4"
  local -a feature_columns=()

  case "$feature_set" in
    4f)
      feature_columns=(
        bes_capacity_kwh
        hvac_total_kw
        heating_mean
        nsl_mean
      )
      ;;
    5f)
      feature_columns=(
        bes_capacity_kwh
        hvac_total_kw
        heating_mean
        nsl_mean
        comfort_lower_excess_mean
      )
      ;;
    *)
      echo "Unknown feature set: $feature_set" >&2
      return 2
      ;;
  esac

  local run_name="mappo_grouped_tarmac_hybrid_${method}_capacity_load_${feature_set}_linear_vt_500_seed${seed}"
  local save_dir="$REPO_DIR/results/$run_name"
  local stdout_path="$LOG_DIR/$run_name.stdout.log"
  local stderr_path="$LOG_DIR/$run_name.stderr.log"

  if [[ -f "$save_dir/latest_metrics.json" && "$FORCE" != "1" ]]; then
    echo "[skip] completed method=$method feature=$feature_set seed=$seed: $save_dir/latest_metrics.json"
    return 0
  fi

  local cpu_spec=""
  local offset
  for ((offset = 0; offset < THREADS_PER_JOB; offset++)); do
    cpu_spec+="${cpu_spec:+,}${CPU_IDS[$((CPU_OFFSET + slot * THREADS_PER_JOB + offset))]}"
  done

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
    --n_buildings 25
    --n_episodes "$EPISODES"
    --train_month 1
    --test_month 2
    --grouping_feature_month 1
    --seed "$seed"
    --group_k_candidates 4 5
    --cluster_seed 0
    --cluster_retries 10
    --grouping_method "$method"
    --grouping_feature_set control_profile
    --grouping_feature_columns "${feature_columns[@]}"
    --hidden_size 256
    --layer_N 2
    --lr 3e-4
    --critic_lr 3e-4
    --gamma 0.99
    --gae_lambda 0.95
    --clip_param 0.2
    --ppo_epoch 10
    --num_mini_batch 4
    --value_loss_coef 1.0
    --entropy_coef 0.01
    --max_grad_norm 10.0
    --weight_nmbe 1.0
    --weight_cv_rmse 1.0
    --weight_comfort 0.8
    --comfort_binary_weight 1.3
    --comfort_degree_weight 0.3
    --comm_hidden_dim 64
    --comm_rounds 1
    --comm_key_dim 32
    --comm_value_dim 64
    --comm_fusion_mode linear
    --comm_dropout 0.0
    --wandb_project annex96-ce1
    --wandb_name "$run_name"
    --save_dir "$save_dir"
  )

  echo "[start] method=$method feature=$feature_set seed=$seed cpus=$cpu_spec"
  echo "        run=$run_name"
  echo "        stdout=$stdout_path"
  echo "        stderr=$stderr_path"
  taskset -c "$cpu_spec" env "${env_args[@]}" "${command[@]}" \
    >"$stdout_path" 2>"$stderr_path"
  echo "[done]  method=$method feature=$feature_set seed=$seed at $(date '+%Y-%m-%d %H:%M:%S')"
}

run_batch() {
  local batch_name="$1"
  shift
  local -a specs=("$@")
  if ((${#specs[@]} != JOBS_PER_BATCH)); then
    echo "Batch '$batch_name' must contain exactly $JOBS_PER_BATCH jobs." >&2
    return 2
  fi

  echo ""
  echo "============================================================"
  echo "Starting $batch_name at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  local -a pids=()
  local -a names=()
  local slot=0
  local spec method seed feature_set
  for spec in "${specs[@]}"; do
    IFS=: read -r method seed feature_set <<<"$spec"
    run_job "$method" "$seed" "$feature_set" "$slot" &
    pids+=("$!")
    names+=("$method/$feature_set/seed$seed")
    slot=$((slot + 1))
  done

  local failed=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[ok]    ${names[$i]}"
    else
      echo "[fail]  ${names[$i]} - inspect its stderr log" >&2
      failed=1
    fi
  done

  echo "Finished $batch_name at $(date '+%Y-%m-%d %H:%M:%S')"
  return "$failed"
}

overall_failed=0

run_batch \
  "Batch 1 - agglomerative 4F seeds 0/1/2/3" \
  "agglomerative:0:4f" \
  "agglomerative:1:4f" \
  "agglomerative:2:4f" \
  "agglomerative:3:4f" || overall_failed=1

run_batch \
  "Batch 2 - K-means and GMM 5F seeds 0/1" \
  "kmeans:0:5f" \
  "kmeans:1:5f" \
  "gmm:0:5f" \
  "gmm:1:5f" || overall_failed=1

echo ""
echo "============================================================"
echo "All requested batches finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Master log: $MASTER_LOG"
echo "============================================================"

if ((overall_failed)); then
  echo "At least one experiment failed. Inspect the per-run stderr logs." >&2
  exit 1
fi

echo "All eight experiments completed successfully."
