from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_ROOT = "target_folder"
DEFAULT_OUTPUT_ROOT = "combined_target_figures"
TIMESERIES_FILE = "test_load_tracking_timeseries.csv"


GROUPS: Dict[str, List[Tuple[str, str]]] = {
    "main_mappo_comparison": [
        ("mappo_standard_vt_500_final3", "MAPPO Standard"),
        ("mappo_grouped_vt_500_final3", "Grouped MAPPO"),
        ("mappo_grouped_comm_v2_vt_500_final2", "Grouped Comm v2"),
    ],
    "weighted_comm_ablation": [
        ("mappo_grouped_comm_weighted_default_vt_500_final2", "Weighted Comm default"),
        ("mappo_grouped_comm_weighted_a090_b010_vt_500_final2", "Weighted Comm alpha=0.90 beta=0.10"),
        ("mappo_grouped_comm_weighted_a055_b045_vt_500_final2", "Weighted Comm alpha=0.55 beta=0.45"),
    ],
    "communication_method_comparison": [
        ("mappo_grouped_comm_v2_vt_500_final2", "Grouped Comm v2"),
        ("mappo_grouped_tarmac_vt_500_final2", "TarMAC"),
        ("mappo_grouped_gat_vt_500_final2", "GAT"),
        ("mappo_grouped_powernet_global_vt_500_final2", "PowerNet Global"),
        ("mappo_grouped_dial_vt_500_final2", "DIAL"),
        ("mappo_grouped_comm_weighted_default_vt_500_final2", "Weighted Comm default"),
    ],
    "independent_agent_baselines": [
        ("rllib_independent_ppo_vt_80_final2", "RLlib Independent PPO"),
        ("rllib_sac_vt_500_final2", "RLlib SAC"),
    ],
    "selected_independent_vs_mappo": [
        ("rllib_independent_ppo_vt_80_final2", "RLlib Independent PPO"),
        ("rllib_sac_vt_500_final2", "RLlib SAC"),
        ("mappo_standard_vt_500_final3", "MAPPO Standard"),
        ("mappo_grouped_vt_500_final3", "Grouped MAPPO"),
    ],
}


def _resolve_root(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def _load_experiment_series(input_root: Path, experiment: str) -> pd.DataFrame:
    csv_path = input_root / experiment / TIMESERIES_FILE
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing load-tracking CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"hour", "controlled_load", "baseline_load", "plotted_target_load"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    return df


def _plot_group(
    *,
    output_path: Path,
    title: str,
    series: Sequence[Tuple[str, pd.DataFrame]],
    max_rows: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_df = series[0][1]
    if max_rows is None:
        x = first_df["hour"].to_numpy()
        target = first_df["plotted_target_load"].to_numpy()
        baseline = first_df["baseline_load"].to_numpy()
    else:
        x = first_df["hour"].to_numpy()[:max_rows]
        target = first_df["plotted_target_load"].to_numpy()[:max_rows]
        baseline = first_df["baseline_load"].to_numpy()[:max_rows]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(x, target, color="black", linestyle="--", linewidth=2.2, label="District Target")
    ax.plot(x, baseline, color="#7a7a7a", linestyle=":", linewidth=1.8, alpha=0.9, label="Baseline")

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (label, df) in enumerate(series):
        y = df["controlled_load"].to_numpy()
        if max_rows is not None:
            y = y[:max_rows]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        ax.plot(x, y, linewidth=1.8, alpha=0.9, label=label, color=color)

    ax.set_title(title)
    ax.set_xlabel("Hour in February Test Period")
    ax.set_ylabel("Portfolio Net Electricity Consumption")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_group_csv(output_path: Path, series: Sequence[Tuple[str, pd.DataFrame]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = series[0][1][["hour", "day", "hour_of_day", "baseline_load", "plotted_target_load"]].copy()

    for label, df in series:
        column = (
            label.lower()
            .replace(" ", "_")
            .replace("=", "")
            .replace(".", "")
            .replace("-", "_")
        )
        base[column] = df["controlled_load"].to_numpy()

    base.to_csv(output_path, index=False)


def _write_manifest(output_root: Path, rows: Sequence[Dict[str, str]]) -> None:
    path = output_root / "combined_load_tracking_manifest.csv"
    fieldnames = ["group", "experiment", "label", "status", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine generated CE1 load-tracking CSVs into multi-experiment target comparison figures."
    )
    parser.add_argument(
        "--input_root",
        default=DEFAULT_INPUT_ROOT,
        help=f"Folder created by generate_load_tracking_figures.py. Default: {DEFAULT_INPUT_ROOT}.",
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Folder for combined figures. Default: {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any requested experiment CSV is missing.",
    )
    parser.add_argument("--steps_per_day", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = _resolve_root(args.input_root)
    output_root = _resolve_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []

    for group_name, experiments in GROUPS.items():
        loaded: List[Tuple[str, pd.DataFrame]] = []

        for experiment, label in experiments:
            try:
                df = _load_experiment_series(input_root, experiment)
            except Exception as exc:
                message = str(exc)
                manifest_rows.append(
                    {
                        "group": group_name,
                        "experiment": experiment,
                        "label": label,
                        "status": "missing_or_invalid",
                        "message": message,
                    }
                )
                if args.strict:
                    raise
                print(f"[WARN] {group_name}: skipped {experiment} - {message}")
                continue

            loaded.append((label, df))
            manifest_rows.append(
                {
                    "group": group_name,
                    "experiment": experiment,
                    "label": label,
                    "status": "loaded",
                    "message": "",
                }
            )

        if not loaded:
            print(f"[WARN] {group_name}: no valid experiments found.")
            continue

        group_dir = output_root / group_name
        _write_group_csv(group_dir / f"{group_name}_combined_timeseries.csv", loaded)

        _plot_group(
            output_path=group_dir / f"{group_name}_full.png",
            title=f"{group_name.replace('_', ' ').title()} - February Load Tracking",
            series=loaded,
            max_rows=None,
        )

        _plot_group(
            output_path=group_dir / f"{group_name}_week1.png",
            title=f"{group_name.replace('_', ' ').title()} - First Week",
            series=loaded,
            max_rows=7 * args.steps_per_day,
        )

        print(f"[OK] {group_name} -> {group_dir}")

    _write_manifest(output_root, manifest_rows)
    print(f"Combined figures -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
