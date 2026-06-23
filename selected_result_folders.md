# Selected Result Folders for Final Reporting

This file lists the experiment outputs that are worth citing in the final report. Folder names are kept unchanged so paths can be copied directly.

## 1. MAPPO Series Comparison

These runs are relatively complete and use comparable settings for the updated final comparison set.

- results/mappo_standard_vt_500_final3
- results/mappo_grouped_vt_500_final3
- results/mappo_grouped_comm_v2_vt_500_final2
- results/mappo_grouped_tarmac_vt_500_final2
- results/mappo_grouped_gat_vt_500_final2
- results/mappo_grouped_powernet_vt_500_final2
- results/mappo_grouped_powernet_global_vt_500_final2
- results/mappo_grouped_commnet_vt_500_final2
- results/mappo_grouped_dial_vt_500_final2

## 2. TarMAC Grouping and Soft-Router Ablations

These runs extend the TarMAC hybrid line with alternative grouping features/methods and dynamic soft-router settings.

- results/mappo_grouped_tarmac_hybrid_agglomerative_static_operational_vt_500_final
- results/mappo_grouped_tarmac_hybrid_gmm_static_operational_vt_500_final
- results/mappo_grouped_tarmac_hybrid_kmeans_operational_profile_vt_500_final
- results/mappo_grouped_tarmac_hybrid_kmeans_static_extended_vt_500_final
- results/mappo_grouped_tarmac_soft_router_vt_500_final_balanced_temperature0.7_warmup50
- results/mappo_grouped_tarmac_soft_router_vt_500_final_sharp_temperature_0.5_warmup_50
- results/mappo_grouped_tarmac_soft_router_vt_500_final_stable_temperature_1_warmup_100

## 3. Weighted Communication Ablation

These runs are suitable for a separate ablation table because the main difference is the intra-group and inter-group communication weighting, controlled by alpha and beta.

- results/mappo_grouped_comm_weighted_default_vt_500_final2
- results/mappo_grouped_comm_weighted_a090_b010_vt_500_final2
- results/mappo_grouped_comm_weighted_a055_b045_vt_500_final2

## 4. Independent-Agent Baselines

These runs can be used as independent-agent baselines or supplementary comparisons. They do not use exactly the same training budget as the main 500-episode MAPPO runs, so avoid using them in the main conclusion to claim absolute superiority.

- results/rllib_independent_ppo_vt_80_final2
- results/rllib_sac_vt_500_final2

Notes:

- `results/rllib_independent_ppo_vt_80_final2` is the PPO independent-agent baseline to use in the updated final summary tables and comparison figures.
- `results/rllib_sac_vt_500_final2` is the SAC independent-agent baseline to use in the updated final summary tables and comparison figures.

## 5. Early 200-Episode MAPPO Results

These runs are recommended only for discussing early results, training-length effects, or historical midterm-report results. Do not directly mix them with the 500-episode final runs in the main comparison table.

- results/mappo_standard_vt_200
- results/mappo_grouped_vt_200
- results/mappo_grouped_comm_vt_200
- results/mappo_grouped_comm_v2_vt_200
- results/mappo_grouped_comm_weighted_vt_200
- results/mappo_grouped_dial_vt_200
- results/mappo_grouped_powernet_vt_200
- results/mappo_grouped_powernet_global_vt_200
- results/mappo_grouped_tarmac_vt_200

## 5. Files to Prioritize in Each Result Folder

When organizing final-report data, prioritize these files:

- `run_config.json`: experiment configuration
- `latest_metrics.json`: final training and testing summary
- `test_metrics.csv`: main test-month metrics, suitable for tables
- `test_metrics.json`: main test-month metrics, suitable for programmatic reading
- `test_daily_metrics.csv`: daily primary metrics
- `test_daily_metrics.png`: daily primary-metrics figure
- `test_daily_secondary_flexible_metrics.csv`: daily secondary metrics for flexible operation
- `test_daily_secondary_baseline_metrics.csv`: daily secondary metrics for the baseline
- `test_daily_secondary_metrics.png`: secondary-metrics figure
- `test_building_comfort_metrics.csv`: per-building comfort metrics
- `training_curves.png`: training curves

For grouped and GAT methods, also keep:

- `cluster_summary.json`
- `cluster_centers.csv`
- `building_cluster_assignment.csv`
- `gat_graph_summary.json`
- `gat_graph_adjacency.csv`

## 6. Folders Not Recommended for Main Numerical Comparison

These folders either lack complete test metrics, are smoke tests, or use training budgets that differ too much from the main experiments. The files can be kept, but they should not be used as primary evidence for the final main conclusions.

- results/rllib_independent_ppo_vt_200
- results/rllib_independent_ppo_vt_300_final
- results/rllib_sac_vt_200
- results/rllib_sac_vt_smoketest
- results/sb3_independent_ppo_vt_200
- results/sb3_independent_ppo_vt_800_final
- results/sb3_independent_sac_vt_200
- results/sb3_independent_sac_vt_300_final
