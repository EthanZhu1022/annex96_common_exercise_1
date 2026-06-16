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
MONTH_STARTS: Dict[int, int] = {
    1: 0,
    2: 744,
    3: 1416,
    4: 2160,
    5: 2880,
    6: 3624,
    7: 4344,
    8: 5088,
    9: 5832,
    10: 6552,
    11: 7296,
    12: 8016,
}
MONTH_ENDS: Dict[int, int] = {
    1: 743,
    2: 1415,
    3: 2159,
    4: 2879,
    5: 3623,
    6: 4343,
    7: 5087,
    8: 5831,
    9: 6551,
    10: 7295,
    11: 8015,
    12: 8759,
}


SELECTED_EXPERIMENTS: List[str] = [
    "rbc_baseline_vt_february",
    "mappo_grouped_comm_v2_vt_500_final2",
    "mappo_grouped_comm_weighted_a055_b045_vt_500_final2",
    "mappo_grouped_comm_weighted_a090_b010_vt_500_final2",
    "mappo_grouped_comm_weighted_default_vt_500_final2",
    "mappo_grouped_commnet_vt_500_final2",
    "mappo_grouped_dial_vt_500_final2",
    "mappo_grouped_gat_vt_500_final2",
    "mappo_grouped_powernet_global_vt_500_final2",
    "mappo_grouped_powernet_vt_500_final2",
    "mappo_grouped_tarmac_vt_500_final2",
    "mappo_grouped_tarmac_hybrid_vt_500_final2",
    "mappo_grouped_tarmac_hybrid_linear_vt_500_final2",
    "mappo_grouped_tarmac_hybrid_gated_vt_500_final2",
    "mappo_grouped_vt_500_final3",
    "mappo_standard_vt_500_final3",
    "rllib_independent_ppo_vt_80_final2",
    "rllib_sac_vt_500_final2",
]


COMFORT_RANK_COLUMNS: List[str] = [
    "primary_comfort_exceedance_pct",
    "primary_comfort_exceedance_hours_total",
    "primary_comfort_exceedance_hours_mean",
    "primary_comfort_exceedance_hours_max",
]

CATEGORY_SPECS: Dict[str, List[Tuple[str, bool]]] = {
    "primary_load_cv_rmse": [("primary_load_cv_rmse_pct", True)],
    "primary_load_nmbe": [("primary_abs_nmbe_pct", True)],
    "thermal_comfort": [(column, True) for column in COMFORT_RANK_COLUMNS],
    "fairness": [
        ("secondary_fairness_gini", True),
        ("secondary_fairness_entropy", False),
        ("secondary_fairness_max_share_pct", True),
    ],
    "site_energy": [("secondary_site_energy_change_pct", True)],
    "peak_demand": [
        ("secondary_peak_demand_change_pct", True),
        ("secondary_peak_demand_kw", True),
    ],
    "peak_to_valley_ratio": [("secondary_peak_to_valley_ratio_pct", True)],
    "load_factor": [("secondary_load_factor_pct", False)],
    "system_ramping": [("secondary_system_ramping_kw", True)],
}


ALL_THINGS_CATEGORIES: List[str] = [
    "primary_load_cv_rmse",
    "primary_load_nmbe",
    "thermal_comfort",
    "fairness",
    "site_energy",
    "peak_demand",
    "peak_to_valley_ratio",
    "load_factor",
    "system_ramping",
]


PRIMARY_ONLY_CATEGORIES: List[str] = [
    "primary_load_cv_rmse",
    "primary_load_nmbe",
    "thermal_comfort",
]


RECOMMENDED_WEIGHT_BY_CATEGORY: Dict[str, float] = {
    "primary_load_cv_rmse": 0.25,
    "primary_load_nmbe": 0.25,
    "thermal_comfort": 0.25,
    "site_energy": 0.10,
    "peak_demand": 0.10,
    "system_ramping": 0.05,
}


SECONDARY_OBJECTIVE_CATEGORIES: List[str] = [
    "fairness",
    "site_energy",
    "peak_demand",
    "peak_to_valley_ratio",
    "load_factor",
    "system_ramping",
]


SUMMARY_COLUMNS: List[str] = [
    "overall_rank",
    "overall_rank_score",
    "primary_rank",
    "primary_rank_score",
    "recommended_rank",
    "recommended_rank_score",
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
    "kpi_carbon_emissions",
    "kpi_cost",
    "kpi_ramping",
    "kpi_daily_peak",
    "kpi_all_time_peak",
    "kpi_load_factor",
    "secondary_cost_change_pct",
    "secondary_cost_absolute",
    "secondary_cost_baseline",
    "secondary_carbon_emissions_change_pct",
    "secondary_carbon_emissions_kgco2e",
    "secondary_carbon_emissions_baseline_kgco2e",
    "secondary_site_energy_change_pct",
    "secondary_site_total_energy_kwh",
    "secondary_site_total_energy_baseline_kwh",
    "secondary_peak_demand_kw",
    "secondary_peak_demand_baseline_kw",
    "secondary_peak_demand_change_pct",
    "secondary_peak_demand_time",
    "secondary_peak_demand_baseline_time",
    "secondary_system_ramping_kw",
    "secondary_system_ramping_baseline_kw",
    "secondary_peak_to_valley_ratio_pct",
    "secondary_peak_to_valley_ratio_baseline_pct",
    "secondary_load_factor_pct",
    "secondary_load_factor_baseline_pct",
    "secondary_fairness_gini",
    "secondary_fairness_entropy",
    "secondary_fairness_max_share_pct",
    "fairness_rank_score",
    "site_energy_rank_score",
    "peak_demand_rank_score",
    "peak_to_valley_ratio_rank_score",
    "load_factor_rank_score",
    "system_ramping_rank_score",
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
    "overall_rank",
    "overall_rank_score",
    "primary_rank",
    "experiment",
    "primary_rank_score",
    "recommended_rank",
    "recommended_rank_score",
    "primary_load_cv_rmse_pct",
    "primary_load_nmbe_pct",
    "primary_comfort_exceedance_pct",
    "primary_comfort_exceedance_hours_total",
    "primary_comfort_exceedance_hours_mean",
    "primary_comfort_exceedance_hours_max",
    "test_reward_sum",
    "kpi_electricity_consumption",
    "kpi_carbon_emissions",
    "kpi_cost",
    "kpi_ramping",
    "kpi_daily_peak",
    "kpi_all_time_peak",
    "kpi_load_factor",
    "secondary_cost_change_pct",
    "secondary_cost_absolute",
    "secondary_cost_baseline",
    "secondary_carbon_emissions_change_pct",
    "secondary_carbon_emissions_kgco2e",
    "secondary_carbon_emissions_baseline_kgco2e",
    "secondary_site_energy_change_pct",
    "secondary_site_total_energy_kwh",
    "secondary_site_total_energy_baseline_kwh",
    "secondary_peak_demand_kw",
    "secondary_peak_demand_baseline_kw",
    "secondary_peak_demand_change_pct",
    "secondary_peak_demand_time",
    "secondary_peak_demand_baseline_time",
    "secondary_peak_to_valley_ratio_pct",
    "secondary_peak_to_valley_ratio_baseline_pct",
    "secondary_load_factor_pct",
    "secondary_load_factor_baseline_pct",
    "secondary_system_ramping_kw",
    "secondary_system_ramping_baseline_kw",
    "secondary_fairness_gini",
    "secondary_fairness_entropy",
    "secondary_fairness_max_share_pct",
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
    label = label.replace("_vt_500_final3", "")
    label = label.replace("_vt_500_final2", "")
    label = label.replace("_vt_500_final", "")
    label = label.replace("_vt_80_final2", "")
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


def _get_season_comfort_bounds(month: int) -> Tuple[str, float, float]:
    if month in {12, 1, 2, 3}:
        return "heating", 20.0, 24.0
    return "cooling", 22.0, 26.0


def _compute_rbc_comfort_from_dataset(climate: str, test_month: int, n_buildings: int = 25) -> Dict[str, Optional[float]]:
    dataset_dir = REPO_DIR / "data" / "datasets" / f"annex96_ce1_{climate.lower()}_neighborhood"
    schema = _load_json(dataset_dir / "schema.json")
    buildings = schema.get("buildings", {})
    if not isinstance(buildings, dict) or not buildings:
        return {
            "primary_comfort_exceedance_hours_total": None,
            "primary_comfort_exceedance_hours_mean": None,
            "primary_comfort_exceedance_hours_max": None,
            "primary_comfort_exceedance_pct": None,
        }

    start = MONTH_STARTS[test_month]
    end = MONTH_ENDS[test_month]
    _, comfort_low_c, comfort_high_c = _get_season_comfort_bounds(test_month)

    hours_by_building: List[float] = []
    for _, building_cfg in list(buildings.items())[:n_buildings]:
        csv_name = building_cfg.get("energy_simulation")
        if not csv_name:
            continue
        csv_path = dataset_dir / str(csv_name)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, usecols=["indoor_dry_bulb_temperature"])
        indoor = df["indoor_dry_bulb_temperature"].iloc[start : end + 1].to_numpy(dtype=float)
        exceed_mask = (indoor < comfort_low_c) | (indoor > comfort_high_c)
        hours_by_building.append(float(exceed_mask.sum()))

    if not hours_by_building:
        return {
            "primary_comfort_exceedance_hours_total": None,
            "primary_comfort_exceedance_hours_mean": None,
            "primary_comfort_exceedance_hours_max": None,
            "primary_comfort_exceedance_pct": None,
        }

    total = float(sum(hours_by_building))
    mean = float(total / len(hours_by_building))
    max_hours = float(max(hours_by_building))
    total_portfolio_hours = float(len(hours_by_building) * (end - start + 1))
    return {
        "primary_comfort_exceedance_hours_total": round(total, 4),
        "primary_comfort_exceedance_hours_mean": round(mean, 4),
        "primary_comfort_exceedance_hours_max": round(max_hours, 4),
        "primary_comfort_exceedance_pct": round(total / total_portfolio_hours * 100.0, 4),
    }


def _compute_rbc_metrics_from_tracking(tracking_dir: Path) -> Dict[str, Any]:
    csv_path = tracking_dir / "test_load_tracking_timeseries.csv"
    summary_path = tracking_dir / "test_load_tracking_summary.json"
    test_metrics_path = tracking_dir / "test_metrics.json"
    df = pd.read_csv(csv_path)
    summary = _load_json(summary_path)
    test_metrics = _load_json(test_metrics_path)

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
    climate = str(summary.get("climate", "VT"))
    test_month = int(summary.get("test_month", 2))
    comfort_metrics = _compute_rbc_comfort_from_dataset(climate, test_month)

    return {
        "climate": climate,
        "test_month": test_month,
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
        "primary_comfort_exceedance_hours_total": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_total"), 4
        ) if test_metrics else comfort_metrics["primary_comfort_exceedance_hours_total"],
        "primary_comfort_exceedance_hours_mean": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_mean"), 4
        ) if test_metrics else comfort_metrics["primary_comfort_exceedance_hours_mean"],
        "primary_comfort_exceedance_hours_max": _round(
            _value(test_metrics, "test/primary/thermal_comfort/temperature_exceedance_hours_max"), 4
        ) if test_metrics else comfort_metrics["primary_comfort_exceedance_hours_max"],
        "primary_comfort_exceedance_pct": _round(
            _value(test_metrics, "test/primary/thermal_comfort/portfolio_exceedance_pct"), 4
        ) if test_metrics else comfort_metrics["primary_comfort_exceedance_pct"],
        "kpi_electricity_consumption": None,
        "kpi_carbon_emissions": None,
        "kpi_cost": None,
        "kpi_ramping": None,
        "kpi_daily_peak": None,
        "kpi_all_time_peak": None,
        "kpi_load_factor": None,
        "secondary_cost_change_pct": None,
        "secondary_cost_absolute": None,
        "secondary_cost_baseline": None,
        "secondary_carbon_emissions_change_pct": None,
        "secondary_carbon_emissions_kgco2e": None,
        "secondary_carbon_emissions_baseline_kgco2e": None,
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
        "secondary_cost_change_pct": _round(_value(test_metrics, "test/secondary/cost_change_pct"), 4),
        "secondary_cost_absolute": _round(_value(test_metrics, "test/secondary/cost_flexible"), 4),
        "secondary_cost_baseline": _round(_value(test_metrics, "test/secondary/cost_baseline"), 4),
        "secondary_carbon_emissions_change_pct": _round(
            _value(test_metrics, "test/secondary/carbon_emissions_change_pct"), 4
        ),
        "secondary_carbon_emissions_kgco2e": _round(
            _value(test_metrics, "test/secondary/carbon_emissions_kgco2e"), 4
        ),
        "secondary_carbon_emissions_baseline_kgco2e": _round(
            _value(test_metrics, "test/secondary/carbon_emissions_baseline_kgco2e"), 4
        ),
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


def _add_category_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()

    for category, specs in CATEGORY_SPECS.items():
        metric_rank_columns: List[str] = []
        for column, ascending in specs:
            rank_column = f"rank_{column}"
            ranked[rank_column] = ranked[column].rank(method="min", ascending=ascending, na_option="bottom")
            metric_rank_columns.append(rank_column)
        ranked[f"{category}_rank_score"] = ranked[metric_rank_columns].mean(axis=1)

    return ranked


def _apply_score_ranking(
    df: pd.DataFrame,
    *,
    score_name: str,
    rank_name: str,
    sort_tiebreakers: Sequence[str],
) -> pd.DataFrame:
    ranked = df.sort_values(
        by=[score_name, *sort_tiebreakers],
        ascending=[True] * (1 + len(sort_tiebreakers)),
    ).reset_index(drop=True)
    ranked[rank_name] = ranked.index + 1
    ranked[score_name] = ranked[score_name].round(4)
    return ranked


def _build_ranked_tables(base_df: pd.DataFrame) -> Dict[str, Any]:
    ranked = _add_category_rank_scores(base_df.copy())

    overall_score_columns = [f"{category}_rank_score" for category in ALL_THINGS_CATEGORIES]
    ranked["overall_rank_score"] = ranked[overall_score_columns].mean(axis=1)

    primary_score_columns = [f"{category}_rank_score" for category in PRIMARY_ONLY_CATEGORIES]
    ranked["primary_rank_score"] = ranked[primary_score_columns].mean(axis=1)

    ranked["recommended_rank_score"] = sum(
        weight * ranked[f"{category}_rank_score"] for category, weight in RECOMMENDED_WEIGHT_BY_CATEGORY.items()
    )

    overall_df = _apply_score_ranking(
        ranked.copy(),
        score_name="overall_rank_score",
        rank_name="overall_rank",
        sort_tiebreakers=["primary_load_cv_rmse_rank_score", "primary_load_nmbe_rank_score", "thermal_comfort_rank_score"],
    )
    primary_df = _apply_score_ranking(
        overall_df.copy(),
        score_name="primary_rank_score",
        rank_name="primary_rank",
        sort_tiebreakers=["primary_load_cv_rmse_rank_score", "primary_load_nmbe_rank_score", "thermal_comfort_rank_score"],
    )
    recommended_df = _apply_score_ranking(
        primary_df.copy(),
        score_name="recommended_rank_score",
        rank_name="recommended_rank",
        sort_tiebreakers=["primary_load_cv_rmse_rank_score", "primary_load_nmbe_rank_score", "thermal_comfort_rank_score"],
    )

    secondary_tables: Dict[str, pd.DataFrame] = {}
    for category in SECONDARY_OBJECTIVE_CATEGORIES:
        score_column = f"{category}_rank_score"
        rank_column = f"{category}_rank"
        secondary_tables[category] = _apply_score_ranking(
            recommended_df.copy(),
            score_name=score_column,
            rank_name=rank_column,
            sort_tiebreakers=["experiment"],
        )

    full_df = recommended_df.copy()
    for score_column in [*overall_score_columns, "overall_rank_score", "primary_rank_score", "recommended_rank_score"]:
        if score_column in full_df.columns:
            full_df[score_column] = full_df[score_column].round(4)

    return {
        "full": full_df,
        "overall": full_df,
        "primary_only": primary_df,
        "recommended": recommended_df,
        "secondary": secondary_tables,
    }


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

    note = ["# Selected Experiment Metrics Summary", ""]
    path.write_text("\n".join(note + lines) + "\n", encoding="utf-8")


def _table_view(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.loc[:, [column for column in columns if column in df.columns]].copy()


def _write_secondary_markdown(path: Path, secondary_tables: Dict[str, pd.DataFrame]) -> None:
    sections = ["# Selected Experiment Metrics Secondary Objective Tables", ""]
    column_map: Dict[str, List[str]] = {
        "fairness": [
            "fairness_rank",
            "experiment",
            "fairness_rank_score",
            "secondary_fairness_gini",
            "secondary_fairness_entropy",
            "secondary_fairness_max_share_pct",
        ],
        "site_energy": [
            "site_energy_rank",
            "experiment",
            "site_energy_rank_score",
            "secondary_site_energy_change_pct",
            "secondary_site_total_energy_kwh",
            "secondary_site_total_energy_baseline_kwh",
        ],
        "peak_demand": [
            "peak_demand_rank",
            "experiment",
            "peak_demand_rank_score",
            "secondary_peak_demand_kw",
            "secondary_peak_demand_baseline_kw",
            "secondary_peak_demand_change_pct",
            "secondary_peak_demand_time",
            "secondary_peak_demand_baseline_time",
        ],
        "peak_to_valley_ratio": [
            "peak_to_valley_ratio_rank",
            "experiment",
            "peak_to_valley_ratio_rank_score",
            "secondary_peak_to_valley_ratio_pct",
            "secondary_peak_to_valley_ratio_baseline_pct",
        ],
        "load_factor": [
            "load_factor_rank",
            "experiment",
            "load_factor_rank_score",
            "secondary_load_factor_pct",
            "secondary_load_factor_baseline_pct",
        ],
        "system_ramping": [
            "system_ramping_rank",
            "experiment",
            "system_ramping_rank_score",
            "secondary_system_ramping_kw",
            "secondary_system_ramping_baseline_kw",
        ],
    }

    for category, columns in column_map.items():
        sections.append(f"## {category.replace('_', ' ').title()}")
        sections.append("")
        view = _table_view(secondary_tables[category], columns)
        headers = list(view.columns)
        rows = []
        for row in view.to_numpy():
            formatted = []
            for value in row:
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    formatted.append("")
                elif isinstance(value, float):
                    formatted.append(f"{value:.4g}")
                else:
                    formatted.append(str(value))
            rows.append(formatted)
        widths = [max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header) for idx, header in enumerate(headers)]
        sections.append("| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
        sections.append("| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |")
        for row in rows:
            sections.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
        sections.append("")

    path.write_text("\n".join(sections), encoding="utf-8")


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
    tables = _build_ranked_tables(pd.DataFrame(rows))
    df = tables["full"]

    all_columns = SUMMARY_COLUMNS + [
        column for column in df.columns if column not in SUMMARY_COLUMNS
    ]
    df = df.loc[:, [column for column in all_columns if column in df.columns]]
    tables["full"] = df

    full_csv = output_root / "selected_experiment_metrics_full.csv"
    report_csv = output_root / "selected_experiment_metrics_report_table.csv"
    markdown_path = output_root / "selected_experiment_metrics_report_table.md"
    primary_csv = output_root / "selected_experiment_metrics_primary_only_table.csv"
    primary_md = output_root / "selected_experiment_metrics_primary_only_table.md"
    recommended_csv = output_root / "selected_experiment_metrics_recommended_table.csv"
    recommended_md = output_root / "selected_experiment_metrics_recommended_table.md"
    secondary_csv = output_root / "selected_experiment_metrics_secondary_objective_ranks.csv"
    secondary_md = output_root / "selected_experiment_metrics_secondary_objective_tables.md"

    df.to_csv(full_csv, index=False)
    overall_columns = [column for column in SUMMARY_COLUMNS if column in df.columns]
    _table_view(tables["overall"], overall_columns).to_csv(report_csv, index=False)
    _write_markdown_table(markdown_path, tables["overall"], MARKDOWN_COLUMNS)

    primary_columns = [
        "primary_rank",
        "experiment",
        "primary_rank_score",
        "primary_load_cv_rmse_pct",
        "primary_load_nmbe_pct",
        "primary_abs_nmbe_pct",
        "primary_comfort_exceedance_pct",
        "primary_comfort_exceedance_hours_total",
        "primary_comfort_exceedance_hours_mean",
        "primary_comfort_exceedance_hours_max",
        "test_reward_sum",
        "wandb_run_id",
    ]
    _table_view(tables["primary_only"], primary_columns).to_csv(primary_csv, index=False)
    _write_markdown_table(primary_md, tables["primary_only"], primary_columns)

    recommended_columns = [
        "recommended_rank",
        "experiment",
        "recommended_rank_score",
        "primary_rank",
        "primary_rank_score",
        "primary_load_cv_rmse_pct",
        "primary_load_nmbe_pct",
        "primary_comfort_exceedance_pct",
        "secondary_site_energy_change_pct",
        "secondary_peak_demand_change_pct",
        "secondary_system_ramping_kw",
        "test_reward_sum",
        "wandb_run_id",
    ]
    _table_view(tables["recommended"], recommended_columns).to_csv(recommended_csv, index=False)
    _write_markdown_table(recommended_md, tables["recommended"], recommended_columns)

    secondary_frames = []
    for category, table in tables["secondary"].items():
        rank_column = f"{category}_rank"
        score_column = f"{category}_rank_score"
        secondary_frames.append(
            _table_view(
                table,
                [
                    rank_column,
                    "experiment",
                    score_column,
                ],
            ).assign(objective=category)
        )
    pd.concat(secondary_frames, ignore_index=True).to_csv(secondary_csv, index=False)
    _write_secondary_markdown(secondary_md, tables["secondary"])

    print(f"Full metrics table -> {full_csv}")
    print(f"Overall report table -> {report_csv}")
    print(f"Overall report markdown -> {markdown_path}")
    print(f"Primary-only table -> {primary_csv}")
    print(f"Recommended table -> {recommended_csv}")
    print(f"Secondary objective tables -> {secondary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
