# Selected Experiment Metrics Secondary Objective Tables

## Fairness

| fairness_rank | experiment                                          | fairness_rank_score | secondary_fairness_gini | secondary_fairness_entropy | secondary_fairness_max_share_pct |
| ------------- | --------------------------------------------------- | ------------------- | ----------------------- | -------------------------- | -------------------------------- |
| 1             | mappo_grouped_tarmac_hybrid_vt_500_final2           | 1.667               | 0.1207                  | 0.9931                     | 5.658                            |
| 2             | mappo_grouped_vt_500_final3                         | 1.667               | 0.1263                  | 0.9919                     | 5.532                            |
| 3             | mappo_grouped_tarmac_vt_500_final2                  | 3                   | 0.1381                  | 0.9897                     | 5.537                            |
| 4             | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 5.333               | 0.1417                  | 0.9903                     | 6.492                            |
| 5             | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 5.667               | 0.151                   | 0.9891                     | 5.983                            |
| 6             | mappo_standard_vt_500_final3                        | 6.333               | 0.1507                  | 0.9891                     | 6.309                            |
| 7             | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 8                   | 0.1534                  | 0.9886                     | 6.484                            |
| 8             | mappo_grouped_comm_v2_vt_500_final2                 | 8.667               | 0.1498                  | 0.9884                     | 7.052                            |
| 9             | mappo_grouped_powernet_global_vt_500_final2         | 8.667               | 0.166                   | 0.9861                     | 6.098                            |
| 10            | mappo_grouped_powernet_vt_500_final2                | 9.667               | 0.1484                  | 0.9877                     | 8.347                            |
| 11            | mappo_grouped_comm_weighted_default_vt_500_final2   | 11                  | 0.1925                  | 0.9813                     | 6.076                            |
| 12            | mappo_grouped_dial_vt_500_final2                    | 11.67               | 0.1686                  | 0.9855                     | 7.411                            |
| 13            | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 12.33               | 0.1838                  | 0.9835                     | 7.046                            |
| 14            | mappo_grouped_commnet_vt_500_final2                 | 12.67               | 0.1697                  | 0.9851                     | 7.638                            |
| 15            | mappo_grouped_gat_vt_500_final2                     | 13.33               | 0.2012                  | 0.9803                     | 6.587                            |
| 16            | rllib_independent_ppo_vt_80_final2                  | 16.33               | 0.2888                  | 0.954                      | 13.62                            |
| 17            | rllib_sac_vt_500_final2                             | 16.67               | 0.2983                  | 0.9529                     | 13.02                            |
| 18            | rbc_baseline_vt_february                            | 18                  |                         |                            |                                  |

## Site Energy

| site_energy_rank | experiment                                          | site_energy_rank_score | secondary_site_energy_change_pct | secondary_site_total_energy_kwh | secondary_site_total_energy_baseline_kwh |
| ---------------- | --------------------------------------------------- | ---------------------- | -------------------------------- | ------------------------------- | ---------------------------------------- |
| 1                | rllib_sac_vt_500_final2                             | 1                      | 13.75                            | 6.218e+04                       | 6.827e+04                                |
| 2                | mappo_grouped_commnet_vt_500_final2                 | 2                      | 22.19                            | 6.372e+04                       | 6.934e+04                                |
| 3                | mappo_grouped_dial_vt_500_final2                    | 3                      | 23.76                            | 6.479e+04                       | 6.985e+04                                |
| 4                | mappo_grouped_tarmac_vt_500_final2                  | 4                      | 25.72                            | 6.802e+04                       | 7.035e+04                                |
| 5                | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 5                      | 26.19                            | 6.832e+04                       | 7.048e+04                                |
| 6                | mappo_grouped_tarmac_hybrid_vt_500_final2           | 6                      | 26.57                            | 6.79e+04                        | 7.037e+04                                |
| 7                | mappo_grouped_gat_vt_500_final2                     | 7                      | 26.63                            | 6.819e+04                       | 7.024e+04                                |
| 8                | mappo_grouped_comm_weighted_default_vt_500_final2   | 8                      | 26.67                            | 6.895e+04                       | 7.058e+04                                |
| 9                | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 9                      | 26.72                            | 6.912e+04                       | 7.049e+04                                |
| 10               | rllib_independent_ppo_vt_80_final2                  | 10                     | 26.81                            | 7.259e+04                       | 7.17e+04                                 |
| 11               | mappo_grouped_comm_v2_vt_500_final2                 | 11                     | 27.37                            | 6.751e+04                       | 7.018e+04                                |
| 12               | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 12                     | 27.79                            | 6.854e+04                       | 7.075e+04                                |
| 13               | mappo_grouped_vt_500_final3                         | 13                     | 28.73                            | 6.852e+04                       | 7.046e+04                                |
| 14               | rbc_baseline_vt_february                            | 14                     | 29.47                            | 9.44e+04                        | 7.291e+04                                |
| 15               | mappo_grouped_powernet_global_vt_500_final2         | 15                     | 30.21                            | 7.156e+04                       | 7.125e+04                                |
| 16               | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 16                     | 30.47                            | 7.106e+04                       | 7.087e+04                                |
| 17               | mappo_standard_vt_500_final3                        | 17                     | 34.08                            | 7.576e+04                       | 7.195e+04                                |
| 18               | mappo_grouped_powernet_vt_500_final2                | 18                     | 43.31                            | 7.91e+04                        | 7.247e+04                                |

## Peak Demand

| peak_demand_rank | experiment                                          | peak_demand_rank_score | secondary_peak_demand_kw | secondary_peak_demand_baseline_kw | secondary_peak_demand_change_pct | secondary_peak_demand_time | secondary_peak_demand_baseline_time |
| ---------------- | --------------------------------------------------- | ---------------------- | ------------------------ | --------------------------------- | -------------------------------- | -------------------------- | ----------------------------------- |
| 1                | rllib_sac_vt_500_final2                             | 1                      | 197.4                    | 243.1                             | -18.8                            | D02 20:00                  | D03 09:00                           |
| 2                | rbc_baseline_vt_february                            | 2                      | 218.8                    | 239                               | -8.442                           |                            |                                     |
| 3                | mappo_grouped_commnet_vt_500_final2                 | 3                      | 243.1                    | 254.4                             | -4.448                           | D03 04:00                  | D03 09:00                           |
| 4                | mappo_grouped_powernet_global_vt_500_final2         | 4                      | 251.7                    | 262.7                             | -4.189                           | D02 19:00                  | D03 09:00                           |
| 5                | rllib_independent_ppo_vt_80_final2                  | 5.5                    | 256.3                    | 263.4                             | -2.688                           | D02 20:00                  | D03 09:00                           |
| 6                | mappo_grouped_dial_vt_500_final2                    | 6                      | 254.5                    | 259.8                             | -2.044                           | D07 22:00                  | D03 09:00                           |
| 7                | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 7                      | 258.1                    | 263.9                             | -2.2                             | D03 03:00                  | D03 09:00                           |
| 8                | mappo_grouped_vt_500_final3                         | 8                      | 257                      | 260.1                             | -1.191                           | D03 06:00                  | D03 09:00                           |
| 9                | mappo_grouped_tarmac_vt_500_final2                  | 9                      | 260.5                    | 264.1                             | -1.375                           | D03 06:00                  | D03 09:00                           |
| 10               | mappo_grouped_powernet_vt_500_final2                | 9.5                    | 259.8                    | 261.4                             | -0.6307                          | D03 08:00                  | D03 09:00                           |
| 11               | mappo_grouped_gat_vt_500_final2                     | 11                     | 263.7                    | 264.1                             | -0.1488                          | D03 04:00                  | D03 09:00                           |
| 12               | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 12.5                   | 264.9                    | 264.6                             | 0.1026                           | D02 19:00                  | D03 09:00                           |
| 13               | mappo_grouped_comm_v2_vt_500_final2                 | 13                     | 270.3                    | 270.4                             | -0.0513                          | D03 08:00                  | D03 09:00                           |
| 14               | mappo_grouped_tarmac_hybrid_vt_500_final2           | 14.5                   | 269.5                    | 259.7                             | 3.777                            | D03 04:00                  | D03 09:00                           |
| 15               | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 15                     | 275.3                    | 273.8                             | 0.535                            | D03 08:00                  | D03 09:00                           |
| 16               | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 15                     | 271.4                    | 265                               | 2.421                            | D02 20:00                  | D03 09:00                           |
| 17               | mappo_grouped_comm_weighted_default_vt_500_final2   | 17                     | 275.4                    | 262                               | 5.129                            | D03 05:00                  | D03 09:00                           |
| 18               | mappo_standard_vt_500_final3                        | 18                     | 299.3                    | 273.9                             | 9.277                            | D03 07:00                  | D03 09:00                           |

## Peak To Valley Ratio

| peak_to_valley_ratio_rank | experiment                                          | peak_to_valley_ratio_rank_score | secondary_peak_to_valley_ratio_pct | secondary_peak_to_valley_ratio_baseline_pct |
| ------------------------- | --------------------------------------------------- | ------------------------------- | ---------------------------------- | ------------------------------------------- |
| 1                         | mappo_grouped_powernet_global_vt_500_final2         | 1                               | 565.2                              | 213.3                                       |
| 2                         | mappo_grouped_tarmac_hybrid_vt_500_final2           | 2                               | 606.3                              | 214.5                                       |
| 3                         | mappo_grouped_tarmac_vt_500_final2                  | 3                               | 635.7                              | 214.6                                       |
| 4                         | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 4                               | 727.5                              | 213.2                                       |
| 5                         | mappo_grouped_comm_v2_vt_500_final2                 | 5                               | 818.9                              | 212.7                                       |
| 6                         | mappo_grouped_commnet_vt_500_final2                 | 6                               | 824.8                              | 212.5                                       |
| 7                         | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 7                               | 897.6                              | 214.9                                       |
| 8                         | rllib_sac_vt_500_final2                             | 8                               | 965.4                              | 215.9                                       |
| 9                         | mappo_grouped_dial_vt_500_final2                    | 9                               | 1026                               | 209.2                                       |
| 10                        | mappo_standard_vt_500_final3                        | 10                              | 1249                               | 216.8                                       |
| 11                        | mappo_grouped_vt_500_final3                         | 11                              | 1347                               | 213.5                                       |
| 12                        | rllib_independent_ppo_vt_80_final2                  | 12                              | 1375                               | 214.6                                       |
| 13                        | mappo_grouped_powernet_vt_500_final2                | 13                              | 1520                               | 215.3                                       |
| 14                        | mappo_grouped_comm_weighted_default_vt_500_final2   | 14                              | 2138                               | 213.7                                       |
| 15                        | mappo_grouped_gat_vt_500_final2                     | 15                              | 3333                               | 212.4                                       |
| 16                        | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 16                              | 3542                               | 214.7                                       |
| 17                        | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 17                              | 5209                               | 212.7                                       |
| 18                        | rbc_baseline_vt_february                            | 18                              |                                    |                                             |

## Load Factor

| load_factor_rank | experiment                                          | load_factor_rank_score | secondary_load_factor_pct | secondary_load_factor_baseline_pct |
| ---------------- | --------------------------------------------------- | ---------------------- | ------------------------- | ---------------------------------- |
| 1                | rllib_sac_vt_500_final2                             | 1                      | 66.06                     | 73.23                              |
| 2                | mappo_grouped_powernet_vt_500_final2                | 2                      | 66.04                     | 72.81                              |
| 3                | rllib_independent_ppo_vt_80_final2                  | 3                      | 64.11                     | 72.89                              |
| 4                | mappo_grouped_powernet_global_vt_500_final2         | 4                      | 63.97                     | 73.48                              |
| 5                | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 5                      | 62.94                     | 73.17                              |
| 6                | mappo_grouped_comm_weighted_default_vt_500_final2   | 6                      | 62.49                     | 73.15                              |
| 7                | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 7                      | 61.72                     | 72.65                              |
| 8                | mappo_grouped_gat_vt_500_final2                     | 8                      | 61                        | 73.08                              |
| 9                | mappo_grouped_tarmac_vt_500_final2                  | 9                      | 60.99                     | 72.76                              |
| 10               | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 10                     | 60.77                     | 73.02                              |
| 11               | mappo_standard_vt_500_final3                        | 11                     | 60.75                     | 72.86                              |
| 12               | mappo_grouped_tarmac_hybrid_vt_500_final2           | 12                     | 60.46                     | 72.75                              |
| 13               | mappo_grouped_comm_v2_vt_500_final2                 | 13                     | 60.17                     | 72.86                              |
| 14               | mappo_grouped_commnet_vt_500_final2                 | 14                     | 59.81                     | 73.14                              |
| 15               | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 15                     | 59.31                     | 73.04                              |
| 16               | mappo_grouped_vt_500_final3                         | 16                     | 58.8                      | 73.18                              |
| 17               | mappo_grouped_dial_vt_500_final2                    | 17                     | 55.01                     | 73.29                              |
| 18               | rbc_baseline_vt_february                            | 18                     |                           |                                    |

## System Ramping

| system_ramping_rank | experiment                                          | system_ramping_rank_score | secondary_system_ramping_kw | secondary_system_ramping_baseline_kw |
| ------------------- | --------------------------------------------------- | ------------------------- | --------------------------- | ------------------------------------ |
| 1                   | rllib_sac_vt_500_final2                             | 1                         | 353.5                       | 195.1                                |
| 2                   | rllib_independent_ppo_vt_80_final2                  | 2                         | 372.1                       | 199.6                                |
| 3                   | mappo_grouped_comm_weighted_default_vt_500_final2   | 3                         | 429.1                       | 199.8                                |
| 4                   | mappo_grouped_commnet_vt_500_final2                 | 4                         | 441.8                       | 196.3                                |
| 5                   | mappo_grouped_comm_weighted_a090_b010_vt_500_final2 | 5                         | 443.1                       | 201.2                                |
| 6                   | mappo_grouped_tarmac_hybrid_vt_500_final2           | 6                         | 457.8                       | 200.8                                |
| 7                   | mappo_grouped_powernet_vt_500_final2                | 7                         | 459.2                       | 203.4                                |
| 8                   | mappo_grouped_comm_weighted_a055_b045_vt_500_final2 | 8                         | 463.4                       | 200.7                                |
| 9                   | mappo_grouped_powernet_global_vt_500_final2         | 9                         | 478.2                       | 197.5                                |
| 10                  | mappo_grouped_tarmac_vt_500_final2                  | 10                        | 486.3                       | 204.1                                |
| 11                  | mappo_grouped_tarmac_hybrid_gated_vt_500_final2     | 11                        | 489.7                       | 202.2                                |
| 12                  | mappo_grouped_comm_v2_vt_500_final2                 | 12                        | 498                         | 202.7                                |
| 13                  | mappo_grouped_dial_vt_500_final2                    | 13                        | 499                         | 203                                  |
| 14                  | mappo_standard_vt_500_final3                        | 14                        | 506.4                       | 204                                  |
| 15                  | mappo_grouped_vt_500_final3                         | 15                        | 519.1                       | 200.6                                |
| 16                  | mappo_grouped_tarmac_hybrid_linear_vt_500_final2    | 16                        | 526                         | 200.8                                |
| 17                  | mappo_grouped_gat_vt_500_final2                     | 17                        | 537.4                       | 198.5                                |
| 18                  | rbc_baseline_vt_february                            | 18                        |                             |                                      |
