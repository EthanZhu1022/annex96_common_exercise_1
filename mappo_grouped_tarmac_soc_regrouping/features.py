"""Pure feature definitions and statistics for policy-induced SOC regrouping."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


SOC_6F_FEATURES = (
    "soc_mean",
    "soc_std",
    "soc_q10",
    "soc_low_fraction",
    "soc_high_fraction",
    "soc_daily_range_mean",
)

ENERGY_4F_FEATURES = (
    "bes_capacity_kwh",
    "soc_q10",
    "heating_mean",
    "nsl_mean",
)

GROUPING_MODES = {
    "soc6f": SOC_6F_FEATURES,
    "energy4f": ENERGY_4F_FEATURES,
}

REQUIRED_TRAJECTORY_COLUMNS = {
    "building_idx",
    "sample_index",
    "electrical_storage_soc",
}


def compute_soc_statistics(trajectory: pd.DataFrame) -> pd.DataFrame:
    """Compute per-building SOC statistics from normalized hourly SOC samples."""

    missing = sorted(REQUIRED_TRAJECTORY_COLUMNS.difference(trajectory.columns))
    if missing:
        raise ValueError(f"SOC trajectory is missing required columns: {missing}")

    frame = trajectory.copy()
    frame["building_idx"] = pd.to_numeric(frame["building_idx"], errors="raise").astype(int)
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    frame["electrical_storage_soc"] = pd.to_numeric(
        frame["electrical_storage_soc"], errors="raise"
    ).astype(float)
    if not np.isfinite(frame["electrical_storage_soc"]).all():
        raise ValueError("SOC trajectory contains non-finite values.")
    if (
        (frame["electrical_storage_soc"] < -1e-6)
        | (frame["electrical_storage_soc"] > 1.0 + 1e-6)
    ).any():
        raise ValueError("electrical_storage_soc must be normalized to [0, 1].")

    frame["electrical_storage_soc"] = frame["electrical_storage_soc"].clip(0.0, 1.0)
    frame["day_index"] = frame["sample_index"] // 24

    rows: List[Dict[str, object]] = []
    for building_idx, building_frame in frame.groupby("building_idx", sort=True):
        building_frame = building_frame.sort_values("sample_index")
        values = building_frame["electrical_storage_soc"].to_numpy(dtype=np.float64)
        daily_ranges = (
            building_frame.groupby("day_index")["electrical_storage_soc"]
            .agg(lambda x: float(x.max() - x.min()))
            .to_numpy(dtype=np.float64)
        )
        name = (
            str(building_frame["building_name"].iloc[0])
            if "building_name" in building_frame.columns
            else f"building_{int(building_idx)}"
        )
        rows.append(
            {
                "building_idx": int(building_idx),
                "building_name": name,
                "n_soc_samples": int(values.size),
                "soc_mean": float(np.mean(values)),
                "soc_std": float(np.std(values, ddof=0)),
                "soc_q10": float(np.quantile(values, 0.10)),
                "soc_low_fraction": float(np.mean(values < 0.1)),
                "soc_high_fraction": float(np.mean(values > 0.9)),
                "soc_daily_range_mean": float(np.mean(daily_ranges)),
            }
        )

    return pd.DataFrame(rows).sort_values("building_idx").reset_index(drop=True)
