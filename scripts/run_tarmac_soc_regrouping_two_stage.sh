#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

SOURCE_CHECKPOINT="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
SOC_DIR="results/mappo_grouped_tarmac_soc_regrouping_source_3f_vt_january"
SOC_STATS="${SOC_DIR}/soc_statistics.csv"
SOC_METADATA="${SOC_DIR}/soc_collection_metadata.json"
SEEDS=(42 0 1)

if [[ -f "${SOC_STATS}" && -f "${SOC_METADATA}" ]]; then
  echo "Reusing completed SOC collection in ${SOC_DIR}"
else
  "${PYTHON_EXE}" -m mappo_grouped_tarmac_soc_regrouping.collect_soc \
    --checkpoint "${SOURCE_CHECKPOINT}" \
    --output_dir "${SOC_DIR}" \
    --climate VT \
    --n_buildings 25 \
    --collection_month 1 \
    --seed 42
fi

run_variant() {
  local grouping_mode="$1"
  local result_prefix="$2"
  local seed
  local save_dir

  for seed in "${SEEDS[@]}"; do
    save_dir="results/${result_prefix}_seed${seed}"
    if [[ -f "${save_dir}/checkpoint.pt" && -f "${save_dir}/latest_metrics.json" ]]; then
      echo "Skipping completed ${grouping_mode}, seed=${seed}: ${save_dir}"
      continue
    fi
    echo "Starting ${grouping_mode}, seed=${seed}, save_dir=${save_dir}"
    "${PYTHON_EXE}" -m mappo_grouped_tarmac_soc_regrouping.train \
      --soc_statistics_path "${SOC_STATS}" \
      --soc_grouping_mode "${grouping_mode}" \
      --climate VT \
      --n_buildings 25 \
      --group_k_candidates 4 5 \
      --cluster_seed 0 \
      --cluster_retries 10 \
      --grouping_method agglomerative \
      --grouping_feature_month 1 \
      --n_episodes 500 \
      --train_month 1 \
      --test_month 2 \
      --seed "${seed}" \
      --comm_fusion_mode linear \
      --wandb_name "$(basename "${save_dir}")" \
      --save_dir "${save_dir}"
  done
}

run_variant \
  "soc6f" \
  "mappo_grouped_tarmac_soc6f_agglomerative_linear_vt_500"

run_variant \
  "energy4f" \
  "mappo_grouped_tarmac_energy4f_agglomerative_linear_vt_500"
