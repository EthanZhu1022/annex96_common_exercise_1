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
    metrics = [
        "primary_load_cv_rmse_pct",
        "primary_abs_nmbe_pct",
        "primary_comfort_degree_hours_per_building_day",
        "primary_comfort_exceedance_pct",
    ]
    count_specs = {
        "3fA": ("capacity_load_3f", 5),
        "4F": ("capacity_load_4f", 4),
        "5F": ("capacity_load_5f", 5),
    }
    count_means: list[pd.Series] = []
    for feature_set, expected_count in count_specs.values():
        rows = PRIMARY.loc[
            (PRIMARY["architecture"] == "TarMAC hybrid")
            & (PRIMARY["grouping_method_short"] == "agglomerative")
            & (PRIMARY["grouping_feature_set_short"] == feature_set)
        ]
        if len(rows) != expected_count:
            raise ValueError(
                f"Expected {expected_count} feature-count rows for {feature_set!r}, found {len(rows)}"
            )
        count_means.append(rows[metrics].apply(pd.to_numeric).mean())

    count_frame = pd.DataFrame(count_means, index=count_specs)
    count_relative = count_frame.min(axis=0).div(count_frame).mul(100.0)
    count_scores = count_relative.mean(axis=1).sort_values(ascending=False)

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
    combination_means: list[pd.Series] = []
    for pattern in patterns.values():
        rows = PRIMARY.loc[
            PRIMARY["experiment"].str.contains(pattern, regex=True, na=False)
        ]
        if len(rows) != 3:
            raise ValueError(f"Expected three feature-combination rows for {pattern!r}")
        combination_means.append(rows[metrics].apply(pd.to_numeric).mean())

    combination_frame = pd.DataFrame(combination_means, index=patterns)
    combination_relative = (
        combination_frame.min(axis=0).div(combination_frame).mul(100.0)
    )
    combination_scores = combination_relative.mean(axis=1).sort_values(
        ascending=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    count_colors = [
        "#2878b5" if label == "3fA" else "#b8c2cc" for label in count_scores.index
    ]
    count_bars = axes[0].bar(
        count_scores.index,
        count_scores.values,
        color=count_colors,
    )
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Four-metric score - higher is better")
    axes[0].set_title("Feature-count ablation")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(count_bars, fmt="%.1f", padding=3)

    combination_colors = [
        "#2878b5" if label == "A" else "#b8c2cc"
        for label in combination_scores.index
    ]
    combination_bars = axes[1].barh(
        [f"3f{label}" for label in combination_scores.index],
        combination_scores.values,
        color=combination_colors,
    )
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 105)
    axes[1].set_xlabel("Four-metric score - higher is better")
    axes[1].set_title("Three-feature combinations - 3 seeds")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].bar_label(combination_bars, fmt="%.1f", padding=3)

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
