"""Build the focused controller comparison used in the workshop paper.

All values are read from the repository's recorded experiment summary.  Exact
experiment names are used deliberately so the figure cannot silently switch to
a different seed or training variant.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = ROOT / "experiment_metric_summary" / "selected_experiment_metrics_full.csv"
OUT = HERE / "figures" / "controller_comparison.png"

EXPERIMENTS = [
    ("Ind.\nPPO", "rllib_independent_ppo_vt_80_final2"),
    ("Ind.\nSAC", "rllib_sac_vt_500_final2"),
    ("MAPPO\n(no comm.)", "mappo_grouped_vt_500_final3"),
    ("TarMAC\nHybrid", "mappo_grouped_tarmac_hybrid_vt_500_final2"),
    (
        "Soft Router",
        "mappo_grouped_tarmac_soft_router_full_expert_stable_heads_3f_vt_seed0",
    ),
]

METRICS = [
    ("primary_load_cv_rmse_pct", "CV-RMSE (%)", "Portfolio tracking"),
    ("primary_load_nmbe_pct", "|NMBE| (%)", "Portfolio bias"),
    ("primary_comfort_exceedance_pct", "Exceedance (%)", "Mean comfort"),
]


def main() -> None:
    frame = pd.read_csv(SUMMARY)
    rows = []
    for label, experiment in EXPERIMENTS:
        selected = frame.loc[frame["experiment"] == experiment]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one row for {experiment!r}, found {len(selected)}"
            )
        row = selected.iloc[0].copy()
        row["label"] = label
        rows.append(row)
    data = pd.DataFrame(rows)

    colors = ["#9aa0a6", "#d98c4a", "#7f8c8d", "#4e79a7", "#2a9d8f"]
    x = np.arange(len(data))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))

    for ax, (column, ylabel, title) in zip(axes, METRICS):
        values = pd.to_numeric(data[column]).abs().to_numpy()
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, data["label"])
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(values) * 1.20)
        fmt = "%.2f" if max(values) < 100 else "%.0f"
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=7.5)

    fig.suptitle(
        "Vermont February: independent control, communication, and routing",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
