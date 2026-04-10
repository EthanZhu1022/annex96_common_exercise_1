"""
Baseline Comparison Utility
============================

Reads test_metrics.csv files produced by independent_sac/train.py,
mappo/train.py, or any other baseline and merges them into a single
comparison table.

Usage:
    # Compare SAC vs MAPPO:
    python -m independent_sac.compare \\
        --sac   results/independent_sac/test_metrics.csv \\
        --mappo results/mappo/test_metrics.csv

    # Add an extra baseline with a custom label:
    python -m independent_sac.compare \\
        --sac   results/independent_sac/test_metrics.csv \\
        --mappo results/mappo/test_metrics.csv \\
        --extra RBC=notebooks/rbc_test_metrics.csv \\
        --output results/comparison_table.csv

    # Programmatic use (from another script):
    from independent_sac.compare import compare
    table = compare({
        "Independent-SAC": "results/independent_sac/test_metrics.csv",
        "MAPPO":           "results/mappo/test_metrics.csv",
    }, output="results/comparison_table.csv")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Metrics to include in the comparison table, in display order.
# These match the column names written by _export_test_metrics in both
# independent_sac/train.py and mappo/train.py.
METRIC_COLS = [
    "climate",
    "seed",
    "test_month_name",
    # Portfolio reward
    "test/portfolio/reward_sum",
    "test/portfolio/reward_mean",
    # District-level CityLearn KPIs (all normalized w.r.t. no-control baseline)
    "test/kpi/electricity_consumption",
    "test/kpi/carbon_emissions",
    "test/kpi/cost",
    "test/kpi/ramping",
    "test/kpi/daily_peak",
    "test/kpi/all_time_peak",
    "test/kpi/load_factor",
]


def load_metrics(path: str, label: str) -> pd.DataFrame:
    """Load a test_metrics.csv and tag it with a baseline label."""
    p = Path(path)
    if not p.exists():
        print(f"[warn] File not found: {p}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["baseline"] = label
    # MAPPO exports KPIs without the test/ prefix in older versions;
    # normalise to the test/ prefix used by this script.
    for col in list(df.columns):
        if col.startswith("kpi/") and f"test/{col}" not in df.columns:
            df[f"test/{col}"] = df[col]
    return df


def compare(
    files:  Dict[str, str],
    output: Optional[str] = None,
    round_digits: int = 4,
) -> pd.DataFrame:
    """Merge multiple baseline test_metrics.csv files into one table.

    Args:
        files:        Mapping of {label: csv_path}.
        output:       Optional path to save the merged CSV.
        round_digits: Decimal places for float columns.

    Returns:
        DataFrame indexed by baseline label.
    """
    dfs = []
    for label, path in files.items():
        df = load_metrics(path, label)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No valid metric files found.", file=sys.stderr)
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Select only the columns that are present
    cols = ["baseline"] + [c for c in METRIC_COLS if c in combined.columns]
    result = combined[cols].set_index("baseline")

    # Round numeric columns for readability
    numeric_cols = result.select_dtypes(include="number").columns
    result[numeric_cols] = result[numeric_cols].round(round_digits)

    print("\n" + "=" * 80)
    print("BASELINE COMPARISON  (test metrics, KPIs normalized to no-control = 1.0)")
    print("=" * 80)
    print(result.to_string())
    print("=" * 80 + "\n")
    print("Interpretation: KPI values < 1.0 indicate improvement over the no-control")
    print("baseline.  reward_sum is episode-cumulative; higher is better.\n")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path)
        print(f"Comparison table saved → {out_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline test_metrics.csv files for Annex96-CE1"
    )
    parser.add_argument(
        "--sac", default=None,
        metavar="CSV",
        help="Path to Independent SAC test_metrics.csv",
    )
    parser.add_argument(
        "--mappo", default=None,
        metavar="CSV",
        help="Path to MAPPO test_metrics.csv",
    )
    parser.add_argument(
        "--rbc", default=None,
        metavar="CSV",
        help="Path to RBC test_metrics.csv (if available)",
    )
    parser.add_argument(
        "--extra", nargs="+", metavar="LABEL=PATH",
        help="Additional baselines as LABEL=PATH pairs (e.g. MyModel=results/my/test_metrics.csv)",
    )
    parser.add_argument(
        "--output", default=None, metavar="CSV",
        help="Save merged comparison table to this CSV path",
    )
    args = parser.parse_args()

    files: Dict[str, str] = {}
    if args.sac:
        files["Independent-SAC"] = args.sac
    if args.mappo:
        files["MAPPO"] = args.mappo
    if args.rbc:
        files["RBC"] = args.rbc
    if args.extra:
        for item in args.extra:
            if "=" not in item:
                print(f"[error] --extra items must be LABEL=PATH, got: {item}", file=sys.stderr)
                sys.exit(1)
            label, path = item.split("=", 1)
            files[label] = path

    if not files:
        parser.print_help()
        sys.exit(1)

    compare(files, output=args.output)


if __name__ == "__main__":
    main()
