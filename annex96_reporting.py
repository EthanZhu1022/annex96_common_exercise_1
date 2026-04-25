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


def collect_building_temperature_timeseries(base_env: Any) -> pd.DataFrame:
    records = []
    buildings = list(getattr(base_env, "buildings", []))

    for building_id, building in enumerate(buildings):
        building_name = getattr(building, "name", f"building_{building_id}")

        indoor = np.asarray(getattr(building, "indoor_dry_bulb_temperature", []), dtype=float)
        cooling_sp = np.asarray(
            getattr(building, "indoor_dry_bulb_temperature_cooling_set_point", []),
            dtype=float,
        )
        heating_sp = np.asarray(
            getattr(building, "indoor_dry_bulb_temperature_heating_set_point", []),
            dtype=float,
        )
        comfort_band = np.asarray(getattr(building, "comfort_band", []), dtype=float)

        length = max(len(indoor), len(cooling_sp), len(heating_sp), len(comfort_band))
        if length <= 0:
            continue

        def _pad(series: np.ndarray) -> np.ndarray:
            output = np.full(length, np.nan, dtype=float)
            if len(series) > 0:
                copy_len = min(length, len(series))
                output[:copy_len] = series[:copy_len]
            return output

        indoor = _pad(indoor)
        cooling_sp = _pad(cooling_sp)
        heating_sp = _pad(heating_sp)
        comfort_band = _pad(comfort_band)
        comfort_low = heating_sp - comfort_band
        comfort_high = cooling_sp + comfort_band

        for hour in range(length):
            indoor_value = indoor[hour]
            lower_value = comfort_low[hour]
            upper_value = comfort_high[hour]
            exceeds_comfort = False
            if np.isfinite(indoor_value):
                if np.isfinite(lower_value) and indoor_value < lower_value:
                    exceeds_comfort = True
                if np.isfinite(upper_value) and indoor_value > upper_value:
                    exceeds_comfort = True

            records.append(
                {
                    "hour": hour,
                    "day": hour // 24 + 1,
                    "hour_of_day": hour % 24,
                    "building_id": building_id,
                    "building_name": building_name,
                    "indoor_temperature": indoor_value,
                    "cooling_set_point": cooling_sp[hour],
                    "heating_set_point": heating_sp[hour],
                    "comfort_band": comfort_band[hour],
                    "comfort_lower_bound": lower_value,
                    "comfort_upper_bound": upper_value,
                    "exceeds_comfort_band": int(exceeds_comfort),
                }
            )

    return pd.DataFrame.from_records(records)


def export_building_temperature_artifacts(
    temperature_df: pd.DataFrame,
    save_dir: Path,
    climate: str,
    month_name: str,
    algorithm_label: Optional[str] = None,
    prefix: str = "test",
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if temperature_df.empty:
        return None, None, None

    save_dir = Path(save_dir)
    csv_path = save_dir / f"{prefix}_building_temperature_timeseries.csv"
    temperature_df.sort_values(["building_id", "hour"]).to_csv(csv_path, index=False)

    label = algorithm_label or "Temperature Curves"
    full_path = save_dir / f"{prefix}_building_temperatures_full.png"
    week1_path = save_dir / f"{prefix}_building_temperatures_week1.png"

    def _plot(frame: pd.DataFrame, out_path: Path, suffix: str) -> Optional[Path]:
        if frame.empty:
            return None

        building_ids = sorted(frame["building_id"].dropna().astype(int).unique().tolist())
        if not building_ids:
            return None

        ncols = min(5, max(1, len(building_ids)))
        nrows = int(np.ceil(len(building_ids) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * 3.6, nrows * 2.7),
            sharex=False,
            sharey=False,
        )
        axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
        legend_handles = None
        legend_labels = None

        for ax, building_id in zip(axes_arr.flat, building_ids):
            building_frame = frame[frame["building_id"] == building_id].sort_values("hour")
            building_name = str(building_frame["building_name"].iloc[0])
            hours = building_frame["hour"].to_numpy(dtype=float)
            indoor = building_frame["indoor_temperature"].to_numpy(dtype=float)
            cooling_sp = building_frame["cooling_set_point"].to_numpy(dtype=float)
            heating_sp = building_frame["heating_set_point"].to_numpy(dtype=float)
            lower = building_frame["comfort_lower_bound"].to_numpy(dtype=float)
            upper = building_frame["comfort_upper_bound"].to_numpy(dtype=float)

            ax.plot(hours, indoor, color="#4e79a7", linewidth=1.2, label="Indoor")
            if np.isfinite(cooling_sp).any():
                ax.plot(hours, cooling_sp, color="#e15759", linewidth=0.9, linestyle="--", label="Cooling SP")
            if np.isfinite(heating_sp).any():
                ax.plot(hours, heating_sp, color="#59a14f", linewidth=0.9, linestyle="--", label="Heating SP")
            if np.isfinite(lower).any() and np.isfinite(upper).any():
                ax.fill_between(hours, lower, upper, color="#9c755f", alpha=0.10, label="Comfort band")

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            ax.set_title(f"B{building_id}: {building_name}", fontsize=9)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Temp (C)")
            ax.grid(True, alpha=0.25)

        for ax in axes_arr.flat[len(building_ids):]:
            ax.axis("off")

        fig.suptitle(f"{label} | {climate} | {month_name} | {suffix}", fontsize=13)
        if legend_handles and legend_labels:
            fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(4, len(legend_labels)))
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(out_path, dpi=130)
        plt.close()
        return out_path

    _plot(temperature_df, full_path, "Full Test Window")
    _plot(temperature_df[temperature_df["hour"] < 24 * 7], week1_path, "Week 1")

    return csv_path, full_path, week1_path
