# Report Table Summary

This document collects the 6 tables used in `report/evaluation.tex` in a readable text format.

## Table 1. Annex 96 train and test split and current evaluation scope

Source label: `tab:split`

| Climate | Training month | Testing month | Current status |
|---|---|---|---|
| Vermont | January | February | fully summarized in this report |
| Texas | August | September | RL comparison remains future work |

## Table 2. Primary-only ranking for the selected Vermont experiments

Source label: `tab:vt-rank-summary`

| Controller | Primary rank | Primary score | CV-RMSE (%) | \|NMBE\| (%) | Comfort exceedance hour |
|---|---:|---:|---:|---:|---:|
| Grouped MAPPO + Weighted Default | 1 | 3.33 | 48.68 | 2.31 | 3166 |
| Grouped MAPPO + PowerNet Global | 2 | 3.75 | 46.45 | 0.43 | 4359 |
| RLlib Independent PPO | 3 | 4.00 | 49.76 | 1.24 | 2309 |
| Grouped MAPPO + Weighted (0.90,0.10) | 4 | 4.83 | 49.75 | 1.95 | 3308 |
| Grouped MAPPO + Weighted (0.55,0.45) | 5 | 5.33 | 49.28 | 0.27 | 5376 |
| Grouped MAPPO | 6 | 6.92 | 51.50 | 2.75 | 3520 |
| Grouped MAPPO + TarMAC | 7 | 7.00 | 49.65 | 3.32 | 3713 |
| Grouped MAPPO + GAT | 8 | 7.75 | 51.35 | 2.92 | 3866 |
| Grouped MAPPO + CommNet | 9 | 8.75 | 48.94 | 8.11 | 4873 |
| RLlib SAC | 10 | 9.25 | 46.53 | 8.93 | 7470 |
| Grouped MAPPO + Comm v2 | 11 | 10.08 | 52.92 | 3.80 | 4031 |
| Standard MAPPO | 12 | 10.42 | 53.30 | 5.29 | 3542 |
| Grouped MAPPO + DIAL | 13 | 11.33 | 52.39 | 7.25 | 6755 |
| Grouped MAPPO + PowerNet | 14 | 12.42 | 50.16 | 9.15 | 7685 |
| RBC baseline | 15 | 14.83 | 138.27 | 109.37 | 11465 |

## Table 3. Overall composite ranking for the selected Vermont experiments

Source label: `tab:vt-overall-composite`

| Controller | Overall rank | Overall score | Primary rank | Recommended rank | Peak change (%) | Ramping (kW) |
|---|---:|---:|---:|---:|---:|---:|
| Grouped MAPPO + PowerNet Global | 1 | 5.14 | 2 | 3 | -4.19 | 478.22 |
| RLlib SAC | 2 | 5.60 | 10 | 7 | -18.80 | 353.48 |
| RLlib Independent PPO | 3 | 5.87 | 3 | 1 | -2.69 | 372.06 |
| Grouped MAPPO + TarMAC | 4 | 6.22 | 7 | 6 | -1.38 | 486.31 |
| Grouped MAPPO + Weighted Default | 5 | 6.48 | 1 | 2 | 5.13 | 429.05 |
| Grouped MAPPO + CommNet | 6 | 6.77 | 9 | 8 | -4.45 | 441.78 |
| Grouped MAPPO + Weighted (0.90,0.10) | 7 | 6.98 | 4 | 4 | -2.20 | 443.12 |
| Grouped MAPPO + Weighted (0.55,0.45) | 8 | 8.15 | 5 | 5 | 0.54 | 463.45 |
| Grouped MAPPO | 9 | 8.19 | 6 | 9 | -1.19 | 519.14 |
| Grouped MAPPO + Comm v2 | 10 | 9.03 | 11 | 12 | -0.05 | 498.03 |
| Grouped MAPPO + DIAL | 11 | 9.19 | 13 | 11 | -2.04 | 498.95 |
| Grouped MAPPO + GAT | 12 | 9.29 | 8 | 10 | -0.15 | 537.38 |
| Grouped MAPPO + PowerNet | 13 | 9.64 | 14 | 14 | -0.63 | 459.18 |
| Standard MAPPO | 14 | 10.40 | 12 | 13 | 9.28 | 506.42 |
| RBC baseline | 15 | 13.06 | 15 | 15 | -8.44 | -- |

## Table 4. Vermont comparison for grouped communication variants

Source label: `tab:vt-comm-results`

| Method | Overall | Primary | Recommended | CV-RMSE (%) | Comfort hours | Peak change (%) |
|---|---:|---:|---:|---:|---:|---:|
| PowerNet Global | 1 | 2 | 3 | 46.45 | 4359 | -4.19 |
| TarMAC | 4 | 7 | 6 | 49.65 | 3713 | -1.38 |
| Weighted default | 5 | 1 | 2 | 48.68 | 3166 | 5.13 |
| CommNet | 6 | 9 | 8 | 48.94 | 4873 | -4.45 |
| Weighted (0.90,0.10) | 7 | 4 | 4 | 49.75 | 3308 | -2.20 |
| Weighted (0.55,0.45) | 8 | 5 | 5 | 49.28 | 5376 | 0.54 |
| Comm v2 | 10 | 11 | 12 | 52.92 | 4031 | -0.05 |
| DIAL | 11 | 13 | 11 | 52.39 | 6755 | -2.04 |
| GAT | 12 | 8 | 10 | 51.35 | 3866 | -0.15 |
| PowerNet | 13 | 14 | 14 | 50.16 | 7685 | -0.63 |

## Table 5. Updated weighted communication ablation for Vermont

Source label: `tab:weighted-ablation`

| Weighted setting | Overall | Primary | CV-RMSE (%) | Comfort hours | Peak change (%) |
|---|---:|---:|---:|---:|---:|
| Default (0.75,0.25) | 5 | 1 | 48.68 | 3166 | 5.13 |
| (0.90,0.10) | 7 | 4 | 49.75 | 3308 | -2.20 |
| (0.55,0.45) | 8 | 5 | 49.28 | 5376 | 0.54 |

## Table 6. RBC notebook baseline comparison against the CE1 target profile

Source label: `tab:rbc-baseline-results`

| Climate | NMBE RBC (%) | NMBE baseline (%) | CV-RMSE RBC (%) | CV-RMSE baseline (%) |
|---|---:|---:|---:|---:|
| TX | 89.57 | 79.12 | 207.63 | 163.74 |
| VT | 103.93 | 101.11 | 129.09 | 144.56 |
