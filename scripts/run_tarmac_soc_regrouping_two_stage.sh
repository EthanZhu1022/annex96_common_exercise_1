#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

SOURCE_CHECKPOINT="results/mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final"
SOC_DIR="results/mappo_grouped_tarmac_soc_regrouping_source_3f_vt_january"
SOC_STATS="${SOC_DIR}/soc_statistics.csv"

SOC6F_SAVE_DIR="results/mappo_grouped_tarmac_soc6f_agglomerative_linear_vt_500_seed42"
ENERGY4F_SAVE_DIR="results/mappo_grouped_tarmac_energy4f_agglomerative_linear_vt_500_seed42"

"${PYTHON_EXE}" -m mappo_grouped_tarmac_soc_regrouping.collect_soc \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --output_dir "${SOC_DIR}" \
  --climate VT \
  --n_buildings 25 \
  --collection_month 1 \
  --seed 42

"${PYTHON_EXE}" -m mappo_grouped_tarmac_soc_regrouping.train \
  --soc_statistics_path "${SOC_STATS}" \
  --soc_grouping_mode soc6f \
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
  --seed 42 \
  --comm_fusion_mode linear \
  --wandb_name "$(basename "${SOC6F_SAVE_DIR}")" \
  --save_dir "${SOC6F_SAVE_DIR}"

"${PYTHON_EXE}" -m mappo_grouped_tarmac_soc_regrouping.train \
  --soc_statistics_path "${SOC_STATS}" \
  --soc_grouping_mode energy4f \
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
  --seed 42 \
  --comm_fusion_mode linear \
  --wandb_name "$(basename "${ENERGY4F_SAVE_DIR}")" \
  --save_dir "${ENERGY4F_SAVE_DIR}"
