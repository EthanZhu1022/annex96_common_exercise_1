# Selected Experiment Metrics Secondary Objective Tables

## Fairness

| fairness_rank | experiment                                         | fairness_rank_score | secondary_fairness_gini | secondary_fairness_entropy | secondary_fairness_max_share_pct |
| ------------- | -------------------------------------------------- | ------------------- | ----------------------- | -------------------------- | -------------------------------- |
| 1             | mappo_grouped_dial_vt_500_final                    | 1.667               | 0.145                   | 0.9899                     | 5.662                            |
| 2             | mappo_grouped_gat_vt_500_final                     | 2                   | 0.1276                  | 0.991                      | 6.679                            |
| 3             | mappo_grouped_powernet_global_vt_500_final         | 4.667               | 0.1612                  | 0.9869                     | 5.973                            |
| 4             | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 5.667               | 0.1784                  | 0.9847                     | 6.358                            |
| 5             | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 6                   | 0.1527                  | 0.9881                     | 7.635                            |
| 6             | mappo_grouped_tarmac_vt_500_final                  | 6                   | 0.1564                  | 0.988                      | 6.935                            |
| 7             | mappo_grouped_powernet_vt_500_final                | 6.333               | 0.1784                  | 0.9847                     | 6.699                            |
| 8             | mappo_grouped_vt_500_final                         | 6.333               | 0.1558                  | 0.9875                     | 7.304                            |
| 9             | sb3_independent_sac_vt_50_final                    | 8.667               | 0.1794                  | 0.9841                     | 6.876                            |
| 10            | mappo_grouped_comm_v2_vt_500_final                 | 9                   | 0.1796                  | 0.9844                     | 6.911                            |
| 11            | mappo_grouped_comm_weighted_default_vt_500_final   | 10                  | 0.1906                  | 0.9821                     | 6.805                            |
| 12            | mappo_standard_vt_500_final                        | 11.67               | 0.1809                  | 0.9839                     | 7.741                            |
| 13            | mappo_grouped_commnet_vt_500_final                 | 12.33               | 0.1913                  | 0.9818                     | 7.463                            |
| 14            | rllib_independent_ppo_vt_42_final                  | 14                  | 0.2841                  | 0.9574                     | 11.46                            |
| 15            | rbc_baseline_vt_february                           | 15                  |                         |                            |                                  |

## Site Energy

| site_energy_rank | experiment                                         | site_energy_rank_score | secondary_site_energy_change_pct | secondary_site_total_energy_kwh | secondary_site_total_energy_baseline_kwh |
| ---------------- | -------------------------------------------------- | ---------------------- | -------------------------------- | ------------------------------- | ---------------------------------------- |
| 1                | rllib_independent_ppo_vt_42_final                  | 1                      | 3.667                            | 5.248e+04                       | 6.714e+04                                |
| 2                | mappo_grouped_dial_vt_500_final                    | 2                      | 4.831                            | 4.675e+04                       | 6.593e+04                                |
| 3                | sb3_independent_sac_vt_50_final                    | 3                      | 6.176                            | 5.487e+04                       | 6.756e+04                                |
| 4                | mappo_standard_vt_500_final                        | 4                      | 6.455                            | 5.356e+04                       | 6.69e+04                                 |
| 5                | mappo_grouped_commnet_vt_500_final                 | 5                      | 6.459                            | 5.293e+04                       | 6.701e+04                                |
| 6                | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 6                      | 8.318                            | 5.439e+04                       | 6.709e+04                                |
| 7                | mappo_grouped_powernet_global_vt_500_final         | 7                      | 9.185                            | 5.404e+04                       | 6.721e+04                                |
| 8                | mappo_grouped_vt_500_final                         | 8                      | 9.484                            | 5.356e+04                       | 6.713e+04                                |
| 9                | mappo_grouped_tarmac_vt_500_final                  | 9                      | 9.832                            | 5.493e+04                       | 6.744e+04                                |
| 10               | mappo_grouped_powernet_vt_500_final                | 10                     | 9.893                            | 5.435e+04                       | 6.722e+04                                |
| 11               | mappo_grouped_comm_v2_vt_500_final                 | 11                     | 10.21                            | 5.541e+04                       | 6.736e+04                                |
| 12               | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 12                     | 10.23                            | 5.375e+04                       | 6.692e+04                                |
| 13               | mappo_grouped_comm_weighted_default_vt_500_final   | 13                     | 10.97                            | 5.52e+04                        | 6.737e+04                                |
| 14               | mappo_grouped_gat_vt_500_final                     | 14                     | 13.82                            | 5.638e+04                       | 6.762e+04                                |
| 15               | rbc_baseline_vt_february                           | 15                     | 29.47                            | 9.44e+04                        | 7.291e+04                                |

## Peak Demand

| peak_demand_rank | experiment                                         | peak_demand_rank_score | secondary_peak_demand_kw | secondary_peak_demand_baseline_kw | secondary_peak_demand_change_pct | secondary_peak_demand_time | secondary_peak_demand_baseline_time |
| ---------------- | -------------------------------------------------- | ---------------------- | ------------------------ | --------------------------------- | -------------------------------- | -------------------------- | ----------------------------------- |
| 1                | rllib_independent_ppo_vt_42_final                  | 1.5                    | 222.4                    | 251.3                             | -11.49                           | D02 20:00                  | D03 09:00                           |
| 2                | mappo_grouped_dial_vt_500_final                    | 3                      | 224.3                    | 251.7                             | -10.88                           | D02 20:00                  | D03 09:00                           |
| 3                | sb3_independent_sac_vt_50_final                    | 3                      | 227.2                    | 255.2                             | -10.97                           | D03 08:00                  | D03 09:00                           |
| 4                | rbc_baseline_vt_february                           | 3.5                    | 218.8                    | 239                               | -8.442                           |                            |                                     |
| 5                | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 4.5                    | 233.8                    | 259.2                             | -9.788                           | D03 08:00                  | D03 09:00                           |
| 6                | mappo_grouped_powernet_vt_500_final                | 5.5                    | 235.9                    | 258.9                             | -8.9                             | D03 08:00                  | D03 09:00                           |
| 7                | mappo_grouped_comm_weighted_default_vt_500_final   | 7                      | 237.8                    | 257.8                             | -7.743                           | D02 20:00                  | D03 09:00                           |
| 8                | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 9.5                    | 245.5                    | 258.8                             | -5.142                           | D03 08:00                  | D03 09:00                           |
| 9                | mappo_grouped_tarmac_vt_500_final                  | 9.5                    | 243                      | 254.3                             | -4.468                           | D03 06:00                  | D03 09:00                           |
| 10               | mappo_grouped_vt_500_final                         | 9.5                    | 240.9                    | 251.5                             | -4.245                           | D02 20:00                  | D03 09:00                           |
| 11               | mappo_standard_vt_500_final                        | 11                     | 251.3                    | 263.8                             | -4.743                           | D03 08:00                  | D03 09:00                           |
| 12               | mappo_grouped_powernet_global_vt_500_final         | 11.5                   | 243.7                    | 252.5                             | -3.469                           | D02 20:00                  | D03 09:00                           |
| 13               | mappo_grouped_commnet_vt_500_final                 | 12                     | 247.2                    | 257.4                             | -3.958                           | D03 07:00                  | D03 09:00                           |
| 14               | mappo_grouped_comm_v2_vt_500_final                 | 14.5                   | 252                      | 257.4                             | -2.104                           | D03 06:00                  | D03 09:00                           |
| 15               | mappo_grouped_gat_vt_500_final                     | 14.5                   | 253.7                    | 260.8                             | -2.72                            | D02 20:00                  | D03 09:00                           |

## Peak To Valley Ratio

| peak_to_valley_ratio_rank | experiment                                         | peak_to_valley_ratio_rank_score | secondary_peak_to_valley_ratio_pct | secondary_peak_to_valley_ratio_baseline_pct |
| ------------------------- | -------------------------------------------------- | ------------------------------- | ---------------------------------- | ------------------------------------------- |
| 1                         | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 1                               | 565.6                              | 213.8                                       |
| 2                         | mappo_grouped_powernet_global_vt_500_final         | 2                               | 695.7                              | 213.7                                       |
| 3                         | mappo_standard_vt_500_final                        | 3                               | 698.1                              | 215.5                                       |
| 4                         | mappo_grouped_comm_v2_vt_500_final                 | 4                               | 701.4                              | 213.8                                       |
| 5                         | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 5                               | 705.4                              | 214.4                                       |
| 6                         | mappo_grouped_dial_vt_500_final                    | 6                               | 708.6                              | 218.1                                       |
| 7                         | mappo_grouped_vt_500_final                         | 7                               | 852.3                              | 215.4                                       |
| 8                         | mappo_grouped_commnet_vt_500_final                 | 8                               | 1099                               | 213.2                                       |
| 9                         | mappo_grouped_gat_vt_500_final                     | 9                               | 1367                               | 214.4                                       |
| 10                        | rllib_independent_ppo_vt_42_final                  | 10                              | 1557                               | 212.3                                       |
| 11                        | mappo_grouped_powernet_vt_500_final                | 11                              | 2127                               | 215                                         |
| 12                        | mappo_grouped_comm_weighted_default_vt_500_final   | 12                              | 2419                               | 217.9                                       |
| 13                        | mappo_grouped_tarmac_vt_500_final                  | 13                              | 4118                               | 214.2                                       |
| 14                        | sb3_independent_sac_vt_50_final                    | 14                              | 4838                               | 213.5                                       |
| 15                        | rbc_baseline_vt_february                           | 15                              |                                    |                                             |

## Load Factor

| load_factor_rank | experiment                                         | load_factor_rank_score | secondary_load_factor_pct | secondary_load_factor_baseline_pct |
| ---------------- | -------------------------------------------------- | ---------------------- | ------------------------- | ---------------------------------- |
| 1                | sb3_independent_sac_vt_50_final                    | 1                      | 58.62                     | 72.29                              |
| 2                | mappo_grouped_comm_v2_vt_500_final                 | 2                      | 57.57                     | 72.11                              |
| 3                | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 2                      | 57.57                     | 72.1                               |
| 4                | mappo_grouped_tarmac_vt_500_final                  | 4                      | 56.57                     | 72.18                              |
| 5                | rllib_independent_ppo_vt_42_final                  | 5                      | 56.52                     | 72.48                              |
| 6                | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 6                      | 55.82                     | 72.18                              |
| 7                | mappo_grouped_commnet_vt_500_final                 | 7                      | 55.4                      | 72.18                              |
| 8                | mappo_grouped_powernet_vt_500_final                | 8                      | 55.33                     | 71.98                              |
| 9                | mappo_standard_vt_500_final                        | 9                      | 55.21                     | 71.97                              |
| 10               | mappo_grouped_gat_vt_500_final                     | 10                     | 54.8                      | 71.96                              |
| 11               | mappo_grouped_vt_500_final                         | 11                     | 53.69                     | 72.08                              |
| 12               | mappo_grouped_comm_weighted_default_vt_500_final   | 12                     | 53.62                     | 71.81                              |
| 13               | mappo_grouped_powernet_global_vt_500_final         | 13                     | 53.41                     | 71.89                              |
| 14               | mappo_grouped_dial_vt_500_final                    | 14                     | 47.63                     | 71.21                              |
| 15               | rbc_baseline_vt_february                           | 15                     |                           |                                    |

## System Ramping

| system_ramping_rank | experiment                                         | system_ramping_rank_score | secondary_system_ramping_kw | secondary_system_ramping_baseline_kw |
| ------------------- | -------------------------------------------------- | ------------------------- | --------------------------- | ------------------------------------ |
| 1                   | sb3_independent_sac_vt_50_final                    | 1                         | 375.8                       | 200.5                                |
| 2                   | rllib_independent_ppo_vt_42_final                  | 2                         | 380.3                       | 198.8                                |
| 3                   | mappo_standard_vt_500_final                        | 3                         | 418.5                       | 203.7                                |
| 4                   | mappo_grouped_comm_weighted_a090_b010_vt_500_final | 4                         | 430.1                       | 202.1                                |
| 5                   | mappo_grouped_comm_v2_vt_500_final                 | 5                         | 438.8                       | 202.9                                |
| 6                   | mappo_grouped_comm_weighted_a055_b045_vt_500_final | 6                         | 439.4                       | 207.1                                |
| 7                   | mappo_grouped_commnet_vt_500_final                 | 7                         | 440.5                       | 203.1                                |
| 8                   | mappo_grouped_tarmac_vt_500_final                  | 8                         | 457.2                       | 202                                  |
| 9                   | mappo_grouped_comm_weighted_default_vt_500_final   | 9                         | 460.7                       | 206.2                                |
| 10                  | mappo_grouped_powernet_global_vt_500_final         | 10                        | 471.9                       | 204.3                                |
| 11                  | mappo_grouped_powernet_vt_500_final                | 11                        | 484.7                       | 202.9                                |
| 12                  | mappo_grouped_gat_vt_500_final                     | 12                        | 515.1                       | 203                                  |
| 13                  | mappo_grouped_vt_500_final                         | 13                        | 517.5                       | 203.9                                |
| 14                  | mappo_grouped_dial_vt_500_final                    | 14                        | 523.4                       | 215.1                                |
| 15                  | rbc_baseline_vt_february                           | 15                        |                             |                                      |
