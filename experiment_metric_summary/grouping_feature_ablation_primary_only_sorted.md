# Grouping Feature Ablation + Soft-Router: Primary-Only Sorted

Sorted by `primary_rank` from `selected_experiment_metrics_primary_only_table.csv`. Lower rank is better.

Included rows: compact 5-feature fixed grouping, previous larger/full fixed grouping, and dynamic soft-router runs.

| primary_rank | primary_rank_score | overall_rank | architecture | grouping_method_short | grouping_feature_set_short | feature_group | experiment | primary_load_cv_rmse_pct | primary_abs_nmbe_pct | primary_comfort_exceedance_pct | test_reward_sum | train_load_cv_rmse_pct | train_comfort_exceedance_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 10.5 | 22 | PowerNet global | agglomerative | capacity_load_5f | 5f compact | mappo_grouped_powernet_global_agglomerative_capacity_load_5f_vt_500_final | 50.7454 | 0.0322 | 20.0595 | -5220.9771 | 38.5177 | 9.586 |
| 7 | 10.75 | 14 | TarMAC hybrid | gmm | static_operational | previous larger/full | mappo_grouped_tarmac_hybrid_gmm_static_operational_vt_500_final | 49.1514 | 0.7687 | 22.4286 | -5927.6258 | 36.9433 | 11.5538 |
| 8 | 11.3333 | 21 | TarMAC hybrid | agglomerative | capacity_load_5f | 5f compact | mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_5f_linear_vt_500_final | 49.2795 | 2.3189 | 22.381 | -5894.957 | 37.686 | 10.3226 |
| 10 | 13.0833 | 17 | TarMAC hybrid | kmeans | capacity_load_5f | 5f compact | mappo_grouped_tarmac_hybrid_kmeans_capacity_load_5f_linear_vt_500_final | 51.5607 | 2.5278 | 17.3333 | -4714.7458 | 39.3795 | 13.7849 |
| 11 | 13.6667 | 7 | PowerNet global | kmeans | capacity_load_5f | 5f compact | mappo_grouped_powernet_global_kmeans_capacity_load_5f_vt_500_final | 49.6914 | 3.7919 | 19.5833 | -5119.5337 | 38.5751 | 13.957 |
| 13 | 14.9167 | 12 | TarMAC hybrid | agglomerative | static_operational | previous larger/full | mappo_grouped_tarmac_hybrid_agglomerative_static_operational_vt_500_final | 48.9323 | 2.3356 | 25.6131 | -6770.2994 | 35.8126 | 20.7204 |
| 16 | 16.25 | 11 | TarMAC soft-router | soft_router | dynamic_router | dynamic soft-router | mappo_grouped_tarmac_soft_router_vt_500_final_sharp_temperature_0.5_warmup_50 | 47.6043 | 4.3058 | 24.6429 | -6890.7015 | 34.6741 | 18.5 |
| 17 | 16.4167 | 8 | PowerNet global | kmeans | static_extended | previous larger/full | mappo_grouped_powernet_global_kmeans_static_extended_vt_500_final | 53.2467 | 0.9472 | 21.2321 | -5689.6631 | 41.4964 | 17.4194 |
| 20 | 18.0 | 27 | TarMAC hybrid | gmm | capacity_load_5f | 5f compact | mappo_grouped_tarmac_hybrid_gmm_capacity_load_5f_linear_vt_500_final | 53.0838 | 0.9417 | 23.75 | -6114.9309 | 39.1036 | 11.6022 |
| 21 | 19.9167 | 31 | TarMAC hybrid | kmeans | operational_profile | previous larger/full | mappo_grouped_tarmac_hybrid_kmeans_operational_profile_vt_500_final | 55.9836 | 0.5487 | 24.5 | -6589.6976 | 42.2954 | 18.6882 |
| 22 | 21.1667 | 19 | TarMAC hybrid | kmeans | static_extended | previous larger/full | mappo_grouped_tarmac_hybrid_kmeans_static_extended_vt_500_final | 50.6258 | 3.7907 | 28.7381 | -8108.6298 | 39.745 | 22.8387 |
| 23 | 21.3333 | 20 | TarMAC soft-router | soft_router | dynamic_router | dynamic soft-router | mappo_grouped_tarmac_soft_router_vt_500_final_stable_temperature_1_warmup_100 | 52.5414 | 5.7213 | 21.8869 | -6535.0646 | 39.7798 | 19.7419 |
| 26 | 22.6667 | 26 | PowerNet global | gmm | static_operational | previous larger/full | mappo_grouped_powernet_global_gmm_static_operational_vt_500_final | 47.6097 | 12.5494 | 36.3452 | -13682.2484 | 39.1219 | 34.9624 |
| 28 | 24.0833 | 23 | TarMAC soft-router | soft_router | dynamic_router | dynamic soft-router | mappo_grouped_tarmac_soft_router_vt_500_final_balanced_temperature0.7_warmup50 | 49.6939 | 7.8848 | 31.4881 | -10883.3506 | 37.7618 | 17.672 |
| 30 | 24.75 | 25 | PowerNet global | gmm | capacity_load_5f | 5f compact | mappo_grouped_powernet_global_gmm_capacity_load_5f_vt_500_final | 52.6427 | 4.94 | 23.244 | -7807.2103 | 41.5932 | 11.0968 |
| 31 | 26.3333 | 34 | PowerNet global | agglomerative | static_operational | previous larger/full | mappo_grouped_powernet_global_agglomerative_static_operational_vt_500_final | 58.8911 | 2.8959 | 39.4464 | -11970.8511 | 47.9889 | 35.1989 |
| 33 | 27.1667 | 32 | PowerNet global | kmeans | operational_profile | previous larger/full | mappo_grouped_powernet_global_kmeans_operational_profile_vt_500_final | 53.3941 | 3.7482 | 31.3571 | -8419.4369 | 39.36 | 16.8441 |
