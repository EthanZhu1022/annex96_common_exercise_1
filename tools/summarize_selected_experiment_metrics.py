from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = "experiment_metric_summary"
DEFAULT_TRACKING_ROOT = "target_folder"


SELECTED_EXPERIMENTS: List[str] = [
    "rbc_baseline_vt_february",
    "mappo_standard_vt_500_final",
    "mappo_grouped_vt_500_final",
    "mappo_grouped_comm_v2_vt_500_final",
    "mappo_grouped_tarmac_vt_500_final",
    "mappo_grouped_gat_vt_500_final",
    "mappo_grouped_powernet_vt_500_final",
    "mappo_grouped_powernet_global_vt_500_final",
    "mappo_grouped_commnet_vt_500_final",
    "mappo_grouped_dial_vt_500_final",
    "rllib_independent_ppo_vt_42_final",
    "sb3_independent_sac_vt_50_final",
    "mappo_grouped_comm_weighted_default_vt_500_final",
    "mappo_grouped_comm_weighted_a090_b010_vt_500_final",
    "mappo_grouped_comm_weighted_a055_b045_vt_500_final",
]


COMFORT_RANK_COLUMNS: List[str] = [
    "primary_comfort_exceedance_pct",
    "primary_comfort_exceedance_hours_total",
    "primary_comfort_exceedance_hours_mean",
    "primary_comfort_exceedance_hours_max",
]


SECONDARY_RANK_COLUMNS: List[str] = [
    "kpi_electricity_consumption",
    "kpi_ramping",
    "kpi_daily_peak",
    "secondary_peak_demand_change_pct",
    "secondary_site_energy_change_pct",
]


SUMMARY_COLUMNS: List[str] = [
    "primary_rank",
    "primary_rank_score",
    "experiment",
    "method",
    "algorithm_family",
    "n_episodes",
    "seed",
    "test_month",
    "primary_load_cv_rmse_pct",
    "primary_load_nmbe_pct",
    "primary_abs_nmbe_pct",
    "primary_comfort_exceedance_pct",
    "primary_comfort_exceedance_hours_total",
    "primary_comfort_exceedance_hours_mean",
    "primary_comfort_exceedance_hours_max",
    "test_reward_sum",
    "test_step_reward_mean",
    "kpi_electricity_consumption",
    "kpi_ramping",
    "kpi_daily_peak",
    "kpi_all_time_peak",
    "kpi_load_factor",
    "secondary_site_energy_change_pct",
    "secondary_site_total_energy_kwh",
    "secondary_peak_demand_kw",
    "secondary_peak_demand_change_pct",
    "secondary_system_ramping_kw",
    "secondary_peak_to_valley_ratio_pct",
    "secondary_load_factor_pct",
    "secondary_fairness_gini",
    "secondary_fairness_entropy",
    "secondary_fairness_max_share_pct",
    "train_load_cv_rmse_pct",
    "train_load_nmbe_pct",
    "train_comfort_exceedance_pct",
    "train_reward_sum",
    "wandb_run_id",
    "wandb_started_at",
    "wandb_runtime_seconds",
    "wandb_program",
    "result_dir",
    "wandb_run_dir",
]


MARKDOWN_COLUMNS: List[str] = [
    "primary_rank",
    "experiment",
    "primary_rank_score",
    "primary_load_cv_rmse_pct",
    "primary_load_nmbe_pct",
    "primary_comfort_exceedance_pct",
    "primary_comfort_exceedance_hours_total",
    "primary_comfort_exceedance_hours_mean",
    "primary_comfort_exceedance_hours_max",
    "test_reward_sum",
    "kpi_electricity_consumption",
    "kpi_ramping",
    "kpi_daily_peak",
    "secondary_peak_demand_change_pct",
    "secondary_site_energy_change_pct",
    "wandb_run_id",
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _value(mapping: Dict[str, Any], key: str) -> Any:
    return mapping.get(key)


def _float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 4) -> Optional[float]:
    number = _float(value)
    return None if number is None else round(number, digits)


def _method_label(experiment: str) -> str:
    if experiment == "rbc_baseline_vt_february":
        return "BasicBatteryRBC"
    label = experiment
    label = label.replace("_vt_500_final", "")
    label = label.replace("_vt_42_final", "")
    label = label.replace("_vt_50_final", "")
    label = label.replace("mappo_", "MAPPO ")
    label = label.replace("rllib_", "RLlib ")
    label = label.replace("sb3_", "SB3 ")
    label = label.replace("_", " ")
    return " ".join(part.upper() if part in {"ppo", "sac", "gat"} else part for part in label.split())


def _algorithm_family(experiment: str) -> str:
    if experiment.startswith("rbc_baseline"):
        return "RBC"
    if experiment.startswith("mappo"):
        return "MAPPO"
    if experiment.startswith("rllib_independent_ppo") or experiment.startswith("sb3_independent_ppo"):
        return "Independent PPO"
    if experiment.startswith("rllib_sac") or experiment.startswith("sb3_independent_sac"):
        return "Independent SAC"
    return "Other"


def _flatten_latest_metrics(latest: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    train = latest.get("train")
    if isinstance(train, dict):
        row["train_reward_sum"] = train.get("last_ep_reward_sum")
        row["train_load_cv_rmse_pct"] = train.get("primary/load_tracking/cv_rmse_pct")
        row["train_load_nmbe_pct"] = train.get("primary/load_tracking/nmbe_pct")
        row["train_comfort_exceedance_pct"] = train.get("primary/thermal_comfort/portfolio_exceedance_pct")
    else:
        row["train_reward_sum"] = latest.get("train/portfolio/reward_sum")
        row["train_load_cv_rmse_pct"] = latest.get("train/primary/load_tracking/cv_rmse_pct")
        row["train_load_nmbe_pct"] = latest.get("train/primary/load_tracking/nmbe_pct")
        row["train_comfort_exceedance_pct"] = latest.get("train/primary/thermal_comfort/portfolio_exceedance_pct")
    return row


def _extract_yaml_value(text: str, key: str) -> Optional[str]:
    pattern = rf"(?m)^{re.escape(key)}:\s*\r?\n\s+value:\s*(.+?)\s*$"
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).strip()
    if value in {"null", "None", "~"}:
        return None
    return value.strip("\"'")


def _find_wandb_run(experiment: str, wandb_root: Path) -> Dict[str, Any]:
    if experiment.startswith("rbc_baseline"):
        return {}
    if not wandb_root.exists():
        return {}

    candidates: List[Tuple[str, Path, Dict[str, Any], Dict[str, Any], str]] = []
    experiment_path = f"results/{experiment}"

    for run_dir in wandb_root.glob("run-*"):
        files_dir = run_dir / "files"
        config_path = files_dir / "config.yaml"
        metadata_path = files_dir / "wandb-metadata.json"
        summary_path = files_dir / "wandb-summary.json"
        if not config_path.exists():
            continue

        config_text = config_path.read_text(encoding="utf-8", errors="replace")
        metadata = _load_json(metadata_path)
        args_text = " ".join(str(item) for item in metadata.get("args", []))

        save_dir = _extract_yaml_value(config_text, "save_dir")
        wandb_name = _extract_yaml_value(config_text, "wandb_name")
        matches = {
            save_dir == experiment_path,
            save_dir == experiment,
            wandb_name == experiment,
            experiment_path in config_text,
            experiment in config_text,
            experiment_path in args_text,
            experiment in args_text,
        }
        if not any(matches):
            continue

        summary = _load_json(summary_path)
        started_at = str(metadata.get("startedAt", ""))
        candidates.append((started_at, run_dir, metadata, summary, config_text))

    if not candidates:
        return {}

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, run_dir, metadata, summary, config_text = candidates[0]
    run_id = run_dir.name.split("-")[-1]
    return {
        "wandb_run_id": run_id,
        "wandb_run_dir": str(run_dir.relative_to(REPO_DIR)),
        "wandb_started_at": metadata.get("startedAt"),
        "wandb_program": metadata.get("program"),
        "wandb_runtime_seconds": _round(summary.get("_runtime"), 2),
        "wandb_name": _extract_yaml_value(config_text, "wandb_name"),
        "wandb_summary": summary,
    }


def _compute_rbc_metrics_from_tracking(tracking_dir: Path) -> Dict[str, Any]:
    csv_path = tracking_dir / "test_load_tracking_timeseries.csv"
    summary_path = tracking_dir / "test_load_tracking_summary.json"
    df = pd.read_csv(csv_path)
    summary = _load_json(summary_path)

    controlled = df["controlled_load"].to_numpy(dtype=float)
    baseline = df["baseline_load"].to_numpy(dtype=float)
    target = df["plotted_target_load"].to_numpy(dtype=float)
    valid = pd.notna(target)
    target_valid = target[valid]
    controlled_valid = controlled[valid]

    reference_mean = float(target_valid.mean()) if len(target_valid) else math.nan
    actual_mean = float(controlled_valid.mean()) if len(controlled_valid) else math.nan
    rmse = float(((controlled_valid - target_valid) ** 2).mean() ** 0.5) if len(target_valid) else math.nan
    nmbe = float((controlled_valid - target_valid).mean() / reference_mean * 100.0) if reference_mean else math.nan
    cv_rmse = float(rmse / reference_mean * 100.0) if reference_mean else math.nan

    baseline_energy = float(baseline.sum()) if len(baseline) else math.nan
    controlled_energy = float(controlled.sum()) if len(controlled) else math.nan
    baseline_peak = float(baseline.max()) if len(baseline) else math.nan
    controlled_peak = float(controlled.max()) if len(controlled) else math.nan
    peak_change_pct = (
        float((controlled_peak - baseline_peak) / baseline_peak * 100.0)
        if baseline_peak and math.isfinite(baseline_peak)
        else math.nan
    )
    site_energy_change_pct = (
        float((controlled_energy - baseline_energy) / baseline_energy * 100.0)
        if baseline_energy and math.isfinite(baseline_energy)
        else math.nan
    )

    return {
        "climate": summary.get("climate", "VT"),
        "test_month": summary.get("test_month", 2),
        "test_month_name": summary.get("test_month_name", "February"),
        "test_reward_sum": None,
        "test_reward_mean": None,
        "test_step_reward_mean": None,
        "primary_load_cv_rmse_pct": round(cv_rmse, 4) if math.isfinite(cv_rmse) else None,
        "primary_load_nmbe_pct": round(nmbe, 4) if math.isfinite(nmbe) else None,
        "primary_load_reference_mean": round(reference_mean, 4) if math.isfinite(reference_mean) else None,
        "primary_load_actual_mean": round(actual_mean, 4) if math.isfinite(actual_mean) else None,
        "primary_load_reference_peak": round(float(target_valid.max()), 4) if len(target_valid) else None,
        "primary_load_actual_peak": round(controlled_peak, 4) if math.isfinite(controlled_peak) else None,
        "primary_comfort_exceedance_hours_total": None,
        "primary_comfort_exceedance_hours_mean": None,
        "primary_comfort_exceedance_hours_max": None,
        "primary_comfort_exceedance_pct": None,
        "kpi_electricity_consumption": None,
        "kpi_carbon_emissions": None,
        "kpi_cost": None,
        "kpi_ramping": None,
        "kpi_daily_peak": None,
        "kpi_all_time_peak": None,
        "kpi_load_factor": None,
        "secondary_site_energy_change_pct": round(site_energy_change_pct, 4) if math.isfinite(site_energy_change_pct) else None,
        "secondary_site_total_energy_kwh": round(controlled_energy, 4) if math.isfinite(controlled_energy) else None,
        "secondary_site_total_energy_baseline_kwh": round(baseline_energy, 4) if math.isfinite(baseline_energy) else None,
        "secondary_peak_demand_kw": round(controlled_peak, 4) if math.isfinite(controlled_peak) else None,
        "secondary_peak_demand_baseline_kw": round(baseline_peak, 4) if math.isfinite(baseline_peak) else None,
        "secondary_peak_demand_change_pct": round(peak_change_pct, 4) if math.isfinite(peak_change_pct) else None,
        "secondary_peak_demand_time": None,
        "secondary_peak_demand_baseline_time": None,
        "secondary_peak_to_valley_ratio_pct": None,
        "secondary_peak_to_valley_ratio_baseline_pct": None,
        "secondary_load_factor_pct": None,
        "secondary_load_factor_baseline_pct": None,
        "secondary_system_ramping_kw": None,
        "secondary_system_ramping_baseline_kw": None,
        "secondary_fairness_gini": None,
        "secondary_fairness_entropy": None,
        "secondary_fairness_max_share_pct": None,
    }


def _build_row(experiment: str, wandb_root: Path) -> Dict[str, Any]:
    if experiment.startswith("rbc_baseline"):
        tracking_dir = REPO_DIR / DEFAULT_TRACKING_ROOT / experiment
        tracking_metrics = _compute_rbc_metrics_from_tracking(tracking_dir)
        row = {
            "experiment": experiment,
            "method": _method_label(experiment),
            "algorithm_family": _algorithm_family(experiment),
            "result_dir": str(tracking_dir.relative_to(REPO_DIR)),
            "n_episodes": None,
            "seed": None,
            "climate": tracking_metrics.get("climate"),
            "train_month": None,
            "test_month": tracking_metrics.get("test_month"),
            "test_month_name": tracking_metrics.get("test_month_name"),
            "test_start_step": None,
            "test_end_step": None,
            "train_reward_sum": None,
            "train_load_cv_rmse_pct": None,
            "train_load_nmbe_pct": None,
            "train_comfort_exceedance_pct": None,
            "wandb_test_reward_sum": None,
            "wandb_test_cv_rmse_pct": None,
            "wandb_test_comfort_exceedance_pct": None,
            "wandb_train_reward_sum": None,
            "wandb_train_cv_rmse_pct": None,
            "wandb_train_comfort_exceedance_pct": None,
            "wandb_run_id": None,
            "wandb_run_dir": None,
            "wandb_started_at": None,
            "wandb_program": None,
            "wandb_runtime_seconds": None,
        }
        row.update(tracking_metrics)
        row["primary_abs_nmbe_pct"] = _round(abs(row["primary_load_nmbe_pct"]), 4)
        return row

    result_dir = REPO_DIR / "results" / experiment
    test_metrics = _load_json(result_dir / "test_metrics.json")
    latest_metrics = _load_json(result_dir / "latest_metrics.json")
    run_config = _load_json(result_dir / "run_config.json")
    wandb = _find_wandb_run(experiment, wandb_root)
    wandb_summary = wandb.pop("wandb_summary", {})

    row: Dict[str, Any] = {
        "experiment": experiment,
        "method": _method_label(experiment),
        "algorithm_family": _algorithm_family(experiment),
        "result_dir": str(result_dir.relative_to(REPO_DIR)),
        "n_episodes": run_config.get("n_episodes") or test_metrics.get("episode"),
        "seed": run_config.get("seed") or test_metrics.get("seed"),
        "climate": run_config.get("climate") or test_metrics.get("climate"),
        "train_month": run_config.get("train_month"),
        "test_month": run_config.get("test_month") or test_metrics.get("test_month"),
        "test_month_name": test_metrics.get("test_month_name"),
        "test_start_step": test_metrics.get("test_start_step"),
        "test_end_step": test_metrics.get("test_end_step"),
        "test_reward_sum": _round(_value(test_metrics, "test/portfolio/reward_sum"), 4),
        "test_reward_mean": _round(_value(test_metrics, "test/portfolio/reward_mean"), 4),
        "test_step_reward_mean": _round(_value(test_metrics, "test/step_reward_mean"), 4),
        "primary_load_cv_rmse_pct": _round(_value(test_metrics, "test/primary/load_tracking/cv_rmse_pct"), 4),
        "primary_load_nmbe_pct": _round(_value(test_metrics, "test/primary/load_tracking/nmbe_pct"), 4),
        "primary_load_reference_mean": _round(_value(test_metrics, "test/primary/load_tracking/reference_mean"), 4),
        "primary_load_actual_mean": _round(_value(test_metrics, "test/primary/load_tracking/actual_mean"), 4),
        "primary_load_reference_peak": _round(_value(test_metrics, "test/primary/load_tracking/reference_peak"), 4),
        "primary_load_actual_peak": _round(_value(test_metrics, "test/primary/load_tracking/actual_peak"), 4),
        "primary_comfort_exceedance_hours_total": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_total"), 4
        ),
        "primary_comfort_exceedance_hours_mean": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_mean"), 4
        ),
        "primary_comfort_exceedance_hours_max": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_max"), 4
        ),
        "primary_comfort_exceedance_pct": _round(
            _value(test_metrics, "test/primary/thermal_comfort/portfolio_exceedance_pct"), 4
        ),
        "kpi_electricity_consumption": _round(_value(test_metrics, "test/kpi/electricity_consumption"), 4),
        "kpi_carbon_emissions": _round(_value(test_metrics, "test/kpi/carbon_emissions"), 4),
        "kpi_cost": _round(_value(test_metrics, "test/kpi/cost"), 4),
        "kpi_ramping": _round(_value(test_metrics, "test/kpi/ramping"), 4),
        "kpi_daily_peak": _round(_value(test_metrics, "test/kpi/daily_peak"), 4),
        "kpi_all_time_peak": _round(_value(test_metrics, "test/kpi/all_time_peak"), 4),
        "kpi_load_factor": _round(_value(test_metrics, "test/kpi/load_factor"), 4),
        "secondary_site_energy_change_pct": _round(_value(test_metrics, "test/secondary/site_total_energy_change_pct"), 4),
        "secondary_site_total_energy_kwh": _round(_value(test_metrics, "test/secondary/site_total_energy_kwh"), 4),
        "secondary_site_total_energy_baseline_kwh": _round(
            _value(test_metrics, "test/secondary/site_total_energy_baseline_kwh"), 4
        ),
        "secondary_peak_demand_kw": _round(_value(test_metrics, "test/secondary/peak_demand_kw"), 4),
        "secondary_peak_demand_baseline_kw": _round(
            _value(test_metrics, "test/secondary/peak_demand_baseline_kw"), 4
        ),
        "secondary_peak_demand_change_pct": _round(_value(test_metrics, "test/secondary/peak_demand_change_pct"), 4),
        "secondary_peak_demand_time": _value(test_metrics, "test/secondary/peak_demand_time"),
        "secondary_peak_demand_baseline_time": _value(test_metrics, "test/secondary/peak_demand_baseline_time"),
        "secondary_peak_to_valley_ratio_pct": _round(_value(test_metrics, "test/secondary/peak_to_valley_ratio_pct"), 4),
        "secondary_peak_to_valley_ratio_baseline_pct": _round(
            _value(test_metrics, "test/secondary/peak_to_valley_ratio_baseline_pct"), 4
        ),
        "secondary_load_factor_pct": _round(_value(test_metrics, "test/secondary/load_factor_pct"), 4),
        "secondary_load_factor_baseline_pct": _round(
            _value(test_metrics, "test/secondary/load_factor_baseline_pct"), 4
        ),
        "secondary_system_ramping_kw": _round(_value(test_metrics, "test/secondary/system_ramping_kw"), 4),
        "secondary_system_ramping_baseline_kw": _round(
            _value(test_metrics, "test/secondary/system_ramping_baseline_kw"), 4
        ),
        "secondary_fairness_gini": _round(_value(test_metrics, "test/secondary/fairness_flexibility_gini"), 4),
        "secondary_fairness_entropy": _round(_value(test_metrics, "test/secondary/fairness_flexibility_entropy"), 4),
        "secondary_fairness_max_share_pct": _round(_value(test_metrics, "test/secondary/fairness_max_share_pct"), 4),
        "wandb_test_reward_sum": _round(wandb_summary.get("test/portfolio/reward_sum"), 4),
        "wandb_test_cv_rmse_pct": _round(wandb_summary.get("test/primary/load_tracking/cv_rmse_pct"), 4),
        "wandb_test_comfort_exceedance_pct": _round(
            wandb_summary.get("test/primary/thermal_comfort/portfolio_exceedance_pct"), 4
        ),
        "wandb_train_reward_sum": _round(wandb_summary.get("train/portfolio/reward_sum"), 4),
        "wandb_train_cv_rmse_pct": _round(wandb_summary.get("train/primary/load_tracking/cv_rmse_pct"), 4),
        "wandb_train_comfort_exceedance_pct": _round(
            wandb_summary.get("train/primary/thermal_comfort/portfolio_exceedance_pct"), 4
        ),
    }

    row["primary_abs_nmbe_pct"] = _round(abs(row["primary_load_nmbe_pct"]), 4)
    row.update(_flatten_latest_metrics(latest_metrics))
    row.update(wandb)
    return row


def _add_primary_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    base_rank_columns = ["primary_load_cv_rmse_pct", "primary_abs_nmbe_pct"] + COMFORT_RANK_COLUMNS + SECONDARY_RANK_COLUMNS

    for column in base_rank_columns:
        rank_column = f"rank_{column}"
        ranked[rank_column] = ranked[column].rank(method="min", ascending=True, na_option="bottom")

    comfort_rank_columns = [f"rank_{column}" for column in COMFORT_RANK_COLUMNS]
    secondary_rank_columns = [f"rank_{column}" for column in SECONDARY_RANK_COLUMNS]
    ranked["comfort_rank_score"] = ranked[comfort_rank_columns].mean(axis=1)
    ranked["secondary_rank_score"] = ranked[secondary_rank_columns].mean(axis=1)

    ranked["primary_rank_score"] = (
        0.30 * ranked["rank_primary_load_cv_rmse_pct"]
        + 0.30 * ranked["rank_primary_abs_nmbe_pct"]
        + 0.30 * ranked["comfort_rank_score"]
        + 0.10 * ranked["secondary_rank_score"]
    )
    ranked = ranked.sort_values(
        by=["primary_rank_score", "rank_primary_load_cv_rmse_pct", "rank_primary_abs_nmbe_pct", "comfort_rank_score"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    ranked["primary_rank"] = ranked.index + 1
    ranked["primary_rank_score"] = ranked["primary_rank_score"].round(4)
    ranked["comfort_rank_score"] = ranked["comfort_rank_score"].round(4)
    ranked["secondary_rank_score"] = ranked["secondary_rank_score"].round(4)
    return ranked


def _write_markdown_table(path: Path, df: pd.DataFrame, columns: Sequence[str]) -> None:
    view = df.loc[:, [column for column in columns if column in df.columns]].copy()
    headers = list(view.columns)

    def format_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    rows = [[format_value(value) for value in row] for row in view.to_numpy()]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]

    lines = []
    lines.append("| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")

    note = [
        "# Selected Experiment Metrics Summary",
        "",
        "Sorted by `primary_rank_score` ascending.",
        "",
        "Weighted rank formula:",
        "`0.30 * rank(primary_load_cv_rmse_pct)`",
        "+ `0.30 * rank(|primary_load_nmbe_pct|)`",
        "+ `0.30 * average_rank(thermal comfort metrics)`",
        "+ `0.10 * average_rank(selected secondary metrics)`",
        "",
        "Thermal comfort metrics: `primary_comfort_exceedance_pct`, "
        "`primary_comfort_exceedance_hours_total`, "
        "`primary_comfort_exceedance_hours_mean`, "
        "`primary_comfort_exceedance_hours_max`.",
        "",
        "Selected secondary metrics: `kpi_electricity_consumption`, `kpi_ramping`, "
        "`kpi_daily_peak`, `secondary_peak_demand_change_pct`, `secondary_site_energy_change_pct`.",
        "",
    ]
    path.write_text("\n".join(note + lines) + "\n", encoding="utf-8")


def _resolve_root(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_DIR / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize selected CE1 experiment result and local W&B metrics into sorted tables."
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output folder relative to repo root by default. Default: {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--wandb_root",
        default="wandb",
        help="Local W&B folder relative to repo root by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = _resolve_root(args.output_root)
    wandb_root = _resolve_root(args.wandb_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [_build_row(experiment, wandb_root) for experiment in SELECTED_EXPERIMENTS]
    df = _add_primary_ranking(pd.DataFrame(rows))

    all_columns = SUMMARY_COLUMNS + [
        column for column in df.columns if column not in SUMMARY_COLUMNS
    ]
    df = df.loc[:, [column for column in all_columns if column in df.columns]]

    full_csv = output_root / "selected_experiment_metrics_full.csv"
    report_csv = output_root / "selected_experiment_metrics_report_table.csv"
    markdown_path = output_root / "selected_experiment_metrics_report_table.md"

    df.to_csv(full_csv, index=False)
    df.loc[:, [column for column in SUMMARY_COLUMNS if column in df.columns]].to_csv(report_csv, index=False)
    _write_markdown_table(markdown_path, df, MARKDOWN_COLUMNS)

    print(f"Full metrics table -> {full_csv}")
    print(f"Report table -> {report_csv}")
    print(f"Markdown table -> {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
