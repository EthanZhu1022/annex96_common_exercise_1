"""Build report2 figures from recorded experiment summary CSV files only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = ROOT / "experiment_metric_summary"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY = pd.read_csv(SUMMARY / "grouping_feature_ablation_primary_only_sorted.csv")
FULL = pd.read_csv(SUMMARY / "selected_experiment_metrics_full.csv")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 130,
    }
)


def exact_row(frame: pd.DataFrame, experiment: str) -> pd.Series:
    rows = frame.loc[frame["experiment"] == experiment]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {experiment!r}, found {len(rows)}")
    return rows.iloc[0]


def save_hybrid_ablation() -> None:
    experiments = [
        "mappo_grouped_tarmac_vt_500_final2",
        "mappo_grouped_tarmac_hybrid_vt_500_final2",
        "mappo_grouped_tarmac_hybrid_linear_vt_500_final2",
        "mappo_grouped_tarmac_hybrid_gated_vt_500_final2",
    ]
    labels = ["Original", "Hybrid ReLU", "Hybrid linear", "Hybrid gated"]
    rows = pd.DataFrame([exact_row(FULL, name) for name in experiments])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    colors = ["#7f8c8d", "#2878b5", "#9ecae1", "#f28e2b"]
    axes[0].bar(labels, rows["primary_load_cv_rmse_pct"], color=colors)
    axes[0].set_ylabel("CV-RMSE (%)")
    axes[0].set_ylim(44, 52)
    axes[0].set_title("Load tracking")
    axes[1].bar(labels, rows["primary_comfort_exceedance_pct"], color=colors)
    axes[1].set_ylabel("Comfort exceedance (%)")
    axes[1].set_ylim(15, 27)
    axes[1].set_title("Thermal comfort")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_ablation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_feature_ablation() -> None:
    count_names = {
        "3fA": "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_3f_linear_vt_500_final",
        "4F": "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_4f_linear_vt_500_final",
        "5F": "mappo_grouped_tarmac_hybrid_agglomerative_capacity_load_5f_linear_vt_500_final",
    }
    count_values = [
        float(exact_row(PRIMARY, name)["primary_load_cv_rmse_pct"])
        for name in count_names.values()
    ]

    patterns = {
        "A": r"agglomerative_capacity_load_3f_linear_vt_500_seed[0-2]$",
        "B": r"agglomerative_3f_B_.*seed[0-2]$",
        "C": r"agglomerative_3f_C_.*seed[0-2]$",
        "D": r"agglomerative_3f_D_.*seed[0-2]$",
        "E": r"agglomerative_3f_E_.*seed[0-2]$",
        "F": r"agglomerative_3f_F_.*seed[0-2]$",
        "G": r"agglomerative_3f_G_.*seed[0-2]$",
        "H": r"agglomerative_3f_H_.*seed[0-2]$",
        "I": r"agglomerative_3f_I_.*seed[0-2]$",
    }
    means: list[float] = []
    errors: list[float] = []
    for pattern in patterns.values():
        values = pd.to_numeric(
            PRIMARY.loc[
                PRIMARY["experiment"].str.contains(pattern, regex=True, na=False),
                "primary_load_cv_rmse_pct",
            ]
        )
        if len(values) != 3:
            raise ValueError(f"Expected three feature-combination rows for {pattern!r}")
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1)))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].bar(list(count_names), count_values, color=["#2878b5", "#f28e2b", "#59a14f"])
    axes[0].set_ylim(40, 57)
    axes[0].set_ylabel("CV-RMSE (%)")
    axes[0].set_title("Feature count (seed 42)")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(patterns))
    axes[1].bar(x, means, yerr=errors, capsize=3, color="#4e79a7")
    axes[1].set_xticks(x, list(patterns))
    axes[1].set_ylim(44, 59)
    axes[1].set_ylabel("CV-RMSE (%)")
    axes[1].set_title("Three-feature combinations (mean ± SD, 3 seeds)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "feature_ablation.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def save_router_progression() -> None:
    single_names = [
        "mappo_grouped_tarmac_soft_router_vt_500_final_sharp_temperature_0.5_warmup_50",
        "mappo_grouped_tarmac_soft_router_twostage_3f_expert_freeze200_temp05_prior1_vt_500_final",
        "mappo_grouped_tarmac_soft_router_three_stage_shared_3f_vt_1500_seed42",
        "mappo_grouped_tarmac_soft_router_three_stage_full_expert_3f_vt_seed42",
        "mappo_grouped_tarmac_soft_router_twostage_3f_expert_routeronly500_temp05_prior1_vt_500_final",
    ]
    labels = ["Legacy", "Two-stage", "Shared 3-stage", "Full 3-stage", "Router-only", "Stable heads"]
    singles = pd.DataFrame([exact_row(PRIMARY, name) for name in single_names])
    stable = PRIMARY.loc[
        PRIMARY["experiment"].str.contains(
            r"soft_router_full_expert_stable_heads_3f_vt_seed(?:0|1|42)$",
            regex=True,
            na=False,
        )
    ]
    if len(stable) != 3:
        raise ValueError(f"Expected three stable-head rows, found {len(stable)}")

    cv = list(pd.to_numeric(singles["primary_load_cv_rmse_pct"])) + [
        float(pd.to_numeric(stable["primary_load_cv_rmse_pct"]).mean())
    ]
    degree = list(pd.to_numeric(singles["primary_comfort_degree_hours_per_building_day"])) + [
        float(pd.to_numeric(stable["primary_comfort_degree_hours_per_building_day"]).mean())
    ]
    cv_err = [0.0] * 5 + [float(pd.to_numeric(stable["primary_load_cv_rmse_pct"]).std(ddof=1))]
    degree_err = [0.0] * 5 + [
        float(pd.to_numeric(stable["primary_comfort_degree_hours_per_building_day"]).std(ddof=1))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    colors = ["#bab0ab", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#2878b5"]
    x = np.arange(len(labels))
    axes[0].bar(x, cv, yerr=cv_err, capsize=3, color=colors)
    axes[0].set_ylabel("CV-RMSE (%)")
    axes[0].set_ylim(42, 55)
    axes[0].set_title("Load tracking")
    axes[1].bar(x, degree, yerr=degree_err, capsize=3, color=colors)
    axes[1].set_ylabel("Degree-hours per building-day")
    axes[1].set_title("Thermal comfort")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "soft_router_progression.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    save_hybrid_ablation()
    save_feature_ablation()
    save_router_progression()
    print(f"Wrote report figures to {OUT}")
