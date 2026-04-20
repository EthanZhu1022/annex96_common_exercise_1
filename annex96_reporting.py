from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mappo.utils import compute_daily_power_metrics


def _safe_label(value: Any) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{value_f:.2f}" if np.isfinite(value_f) else "nan"


def compute_secondary_daily_tables(
    flexible_loads: Sequence[float],
    baseline_loads: Sequence[float],
    steps_per_day: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flexible_df = compute_daily_power_metrics(flexible_loads, steps_per_day)
    baseline_df = compute_daily_power_metrics(baseline_loads, steps_per_day)
    return flexible_df, baseline_df


def build_readme_secondary_daily_log(row: Any, prefix: str) -> Dict[str, Optional[float]]:
    """Return README-aligned W&B aliases for one daily secondary row."""

    def _value(column: str) -> Optional[float]:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _pct(column: str) -> Optional[float]:
        value = _value(column)
        return None if value is None else float(value * 100.0)

    return {
        f"{prefix}/system_ramping_kw": _value("ramping"),
        f"{prefix}/peak_demand_kw": _value("daily_peak"),
        f"{prefix}/load_factor_pct": _pct("load_factor"),
        f"{prefix}/peak_to_valley_ratio_pct": _pct("pvr"),
        f"{prefix}/site_total_energy_kwh": _value("energy"),
    }


def save_secondary_daily_metrics_plot(
    flexible_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    save_dir: Path,
    climate: str,
    month_name: str,
    algorithm_label: str,
) -> Optional[Path]:
    if flexible_df.empty and baseline_df.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"{algorithm_label} - Secondary Metrics | {climate} | {month_name}", fontsize=13)
    days = flexible_df["day"].tolist() if not flexible_df.empty else baseline_df["day"].tolist()

    def _plot(ax: plt.Axes, column: str, title: str, ylabel: str) -> None:
        if not flexible_df.empty:
            flex_vals = flexible_df[column].tolist()
            ax.plot(days, flex_vals, marker="o", markersize=3, color="#4e79a7", linewidth=1.2, label="Flexible")
            ax.fill_between(days, flex_vals, alpha=0.12, color="#4e79a7")
            ax.axhline(
                float(np.nanmean(flex_vals)),
                color="#4e79a7",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                label=f"Flexible mean={_safe_label(np.nanmean(flex_vals))}",
            )
        if not baseline_df.empty:
            base_vals = baseline_df[column].tolist()
            ax.plot(days, base_vals, marker="s", markersize=3, color="#e15759", linewidth=1.2, label="Baseline")
            ax.fill_between(days, base_vals, alpha=0.10, color="#e15759")
            ax.axhline(
                float(np.nanmean(base_vals)),
                color="#e15759",
                linestyle=":",
                linewidth=0.8,
                alpha=0.7,
                label=f"Baseline mean={_safe_label(np.nanmean(base_vals))}",
            )
        ax.set_title(title)
        ax.set_xlabel("Day of test month")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    _plot(axes[0, 0], "ramping", "Daily Ramping", "kW")
    _plot(axes[0, 1], "daily_peak", "Daily Peak Demand", "kW")
    _plot(axes[0, 2], "load_factor", "Daily Load Factor", "-")
    _plot(axes[1, 0], "pvr", "Daily Peak-to-Valley Ratio", "-")
    _plot(axes[1, 1], "energy", "Daily Site Energy", "kWh")

    ax_summary = axes[1, 2]
    ax_summary.axis("off")
    summary_rows = []
    for label, frame in [("Flexible", flexible_df), ("Baseline", baseline_df)]:
        if frame.empty:
            continue
        for metric in ["ramping", "daily_peak", "load_factor", "pvr", "energy"]:
            values = frame[metric].dropna()
            summary_rows.append(
                [
                    f"{label}:{metric}",
                    f"{values.mean():.3f}" if not values.empty else "nan",
                    f"{values.std():.3f}" if not values.empty else "nan",
                ]
            )
    if summary_rows:
        tbl = ax_summary.table(
            cellText=summary_rows,
            colLabels=["Series:Metric", "Mean", "Std"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.35)
    ax_summary.set_title("Summary Statistics")

    plt.tight_layout()
    out = Path(save_dir) / "test_daily_secondary_metrics.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return out


def export_secondary_daily_metrics(
    flexible_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    save_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    flexible_path: Optional[Path] = None
    baseline_path: Optional[Path] = None
    if not flexible_df.empty:
        flexible_path = Path(save_dir) / "test_daily_secondary_flexible_metrics.csv"
        flexible_df.to_csv(flexible_path, index=False)
    if not baseline_df.empty:
        baseline_path = Path(save_dir) / "test_daily_secondary_baseline_metrics.csv"
        baseline_df.to_csv(baseline_path, index=False)
    return flexible_path, baseline_path
