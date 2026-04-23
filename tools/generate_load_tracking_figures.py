from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

DEFAULT_OUTPUT_ROOT_NAME = "target_folder"

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
MONTH_NAMES: Dict[int, str] = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}


MODULE_BY_PREFIX: List[Tuple[str, str]] = [
    ("mappo_grouped_powernet_global", "mappo_grouped_powernet_global.train"),
    ("mappo_grouped_comm_weighted", "mappo_grouped_comm_weighted.train"),
    ("mappo_grouped_comm_v2", "mappo_grouped_comm_v2.train"),
    ("mappo_grouped_commnet", "mappo_grouped_comm.train"),
    ("mappo_grouped_comm", "mappo_grouped_comm.train"),
    ("mappo_grouped_powernet", "mappo_grouped_powernet.train"),
    ("mappo_grouped_tarmac", "mappo_grouped_tarmac.train"),
    ("mappo_grouped_dial", "mappo_grouped_dial.train"),
    ("mappo_grouped_gat", "mappo_grouped_gat.train"),
    ("mappo_standard", "mappo_standard.train"),
    ("mappo_grouped", "mappo_grouped.train"),
    ("rllib_independent_ppo", "rllib_independent_ppo.train"),
    ("rllib_sac", "rllib_sac.train"),
    ("sb3_independent_ppo", "sb3_independent_ppo.train"),
    ("sb3_independent_sac", "sb3_independent_sac.train"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_selected_result_dirs(path: Path, include_not_recommended: bool) -> List[Path]:
    """Read result folders from selected_result_folders.md.

    The selected-result file is expected to use English headings.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    selected: List[Path] = []
    in_not_recommended = False

    not_recommended_markers = (
        "folders not recommended for main numerical comparison",
        "not recommended",
    )

    for line in text.splitlines():
        lower_line = line.lower()
        if line.lstrip().startswith("##") and any(marker in lower_line for marker in not_recommended_markers):
            in_not_recommended = True

        match = re.match(r"\s*-\s+(results/[^\s`]+)", line)
        if not match:
            continue
        if in_not_recommended and not include_not_recommended:
            continue

        result_dir = (REPO_DIR / match.group(1)).resolve()
        if result_dir not in selected:
            selected.append(result_dir)

    return selected


def _resolve_module_name(result_dir: Path) -> str:
    name = result_dir.name
    for prefix, module_name in MODULE_BY_PREFIX:
        if name.startswith(prefix):
            return module_name
    raise ValueError(f"Cannot infer experiment module from result directory name: {name}")


def _disable_wandb(module: Any) -> None:
    for candidate in [module, getattr(module, "base_train", None)]:
        if candidate is None:
            continue
        if hasattr(candidate, "_WANDB_OK"):
            setattr(candidate, "_WANDB_OK", False)


def _make_config(module: Any, result_dir: Path, output_dir: Path, overrides: Dict[str, Any]) -> Any:
    if not hasattr(module, "Config"):
        raise AttributeError(f"{module.__name__} has no Config dataclass.")

    cfg = module.Config()
    run_cfg = _load_json(result_dir / "run_config.json")
    for key, value in run_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)

    if hasattr(cfg, "save_dir"):
        setattr(cfg, "save_dir", str(result_dir))
    if hasattr(cfg, "test_only"):
        setattr(cfg, "test_only", True)
    if hasattr(cfg, "do_test"):
        setattr(cfg, "do_test", True)
    if hasattr(cfg, "test_save_dir"):
        setattr(cfg, "test_save_dir", str(output_dir))
    if hasattr(cfg, "wandb_name"):
        setattr(cfg, "wandb_name", f"{result_dir.name}-load-tracking")

    if hasattr(cfg, "checkpoint_dir"):
        checkpoint_dir = _checkpoint_arg_for(result_dir)
        setattr(cfg, "checkpoint_dir", str(checkpoint_dir))

    return cfg


def _checkpoint_arg_for(result_dir: Path) -> Path:
    if result_dir.name.startswith(("rllib_independent_ppo", "rllib_sac")):
        return result_dir / "checkpoint"
    return result_dir


def _run_module_test_only(module_name: str, result_dir: Path, output_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    module = importlib.import_module(module_name)
    _disable_wandb(module)
    cfg = _make_config(module, result_dir, output_dir, overrides)

    run_module = module
    if not hasattr(run_module, "run_test_only") and hasattr(module, "base_train"):
        run_module = module.base_train
        _disable_wandb(run_module)

    if not hasattr(run_module, "run_test_only"):
        raise AttributeError(f"{module_name} does not expose run_test_only().")

    result = run_module.run_test_only(cfg)
    if not isinstance(result, dict):
        raise TypeError(f"{module_name}.run_test_only() returned {type(result).__name__}, expected dict.")
    return result


def _run_rllib_sac_test(result_dir: Path, output_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    import ray
    from rllib_sac.env import CityLearnMultiAgentEnv
    from rllib_sac.train import Config, _MONTH_ENDS, _MONTH_STARTS, build_sac_config, evaluate_on_test, seed_everything

    module = importlib.import_module("rllib_sac.train")
    _disable_wandb(module)

    cfg = Config()
    run_cfg = _load_json(result_dir / "run_config.json")
    for key, value in run_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    for key, value in overrides.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    cfg.save_dir = str(result_dir)
    cfg.do_test = True

    test_month = cfg.test_month or {"VT": 2, "TX": 9}.get(cfg.climate)
    train_month = cfg.train_month or {"VT": 1, "TX": 8}.get(cfg.climate)
    if test_month is None or train_month is None:
        raise ValueError(f"Cannot resolve train/test months for {result_dir}")
    cfg.test_month = test_month
    cfg.train_month = train_month

    train_start = _MONTH_STARTS[train_month]
    train_end = _MONTH_ENDS[train_month]
    test_start = _MONTH_STARTS[test_month]
    test_end = _MONTH_ENDS[test_month]

    seed_everything(cfg.seed)
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    probe_env = CityLearnMultiAgentEnv(
        {
            "climate": cfg.climate,
            "n_buildings": cfg.n_buildings,
            "start_step": train_start,
            "end_step": train_end,
            "seed": cfg.seed,
        }
    )
    probe_env.reset()
    agent_ids = sorted(probe_env.get_agent_ids(), key=lambda s: int(s.split("_")[1]))
    sample_aid = agent_ids[0]
    obs_space = probe_env.observation_space[sample_aid]
    act_space = probe_env.action_space[sample_aid]
    action_spaces = {aid: probe_env.action_space[aid] for aid in agent_ids}
    probe_env.close()

    sac_config = build_sac_config(
        cfg=cfg,
        env_config={
            "climate": cfg.climate,
            "n_buildings": cfg.n_buildings,
            "start_step": train_start,
            "end_step": train_end,
            "seed": cfg.seed,
        },
        agent_ids=agent_ids,
        obs_space=obs_space,
        act_space=act_space,
    )
    algorithm = sac_config.build()
    try:
        checkpoint_dir = result_dir / "checkpoint"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"RLlib SAC checkpoint directory not found: {checkpoint_dir}")
        algorithm.restore(str(checkpoint_dir))
        return evaluate_on_test(
            algorithm=algorithm,
            cfg=cfg,
            test_start=test_start,
            test_end=test_end,
            agent_ids=agent_ids,
            action_spaces=action_spaces,
            use_wandb=False,
            iteration=None,
        )
    finally:
        algorithm.stop()


def _run_sb3_test(module_name: str, result_dir: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    module = importlib.import_module(module_name)
    common = importlib.import_module("sb3_independent_common")
    cfg = _make_config(module, result_dir, result_dir, overrides)

    if result_dir.name.startswith("sb3_independent_ppo"):
        from stable_baselines3 import PPO as Algorithm
    elif result_dir.name.startswith("sb3_independent_sac"):
        from stable_baselines3 import SAC as Algorithm
    else:
        raise ValueError(f"Unsupported SB3 result directory: {result_dir.name}")

    models = []
    for i in range(int(getattr(cfg, "n_buildings", 25))):
        model_path = result_dir / f"building_{i}_model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"SB3 model file not found: {model_path}")
        models.append(Algorithm.load(str(model_path), device="auto"))

    test_month = getattr(cfg, "test_month", None) or {"VT": 2, "TX": 9}.get(getattr(cfg, "climate", "VT"))
    if test_month is None:
        raise ValueError(f"Cannot resolve test month for {result_dir}")
    setattr(cfg, "test_month", test_month)

    return common.evaluate_portfolio_models(
        models=models,
        cfg=cfg,
        start_step=MONTH_STARTS[test_month],
        end_step=MONTH_ENDS[test_month],
        month=test_month,
        deterministic=True,
    )


def _extract_series(test_result: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    loads = np.asarray(test_result.get("_step_portfolio_loads", []), dtype=float)
    baseline = np.asarray(test_result.get("_step_portfolio_loads_baseline", []), dtype=float)
    if loads.size == 0:
        raise ValueError("Test result did not include _step_portfolio_loads.")
    if baseline.size == 0:
        baseline = np.full_like(loads, np.nan, dtype=float)
    n = min(loads.size, baseline.size)
    return loads[:n], baseline[:n]


def _computed_daily_reference(baseline: np.ndarray, steps_per_day: int) -> np.ndarray:
    ref = np.full_like(baseline, np.nan, dtype=float)
    n_days = len(baseline) // steps_per_day
    for day in range(n_days):
        start = day * steps_per_day
        end = start + steps_per_day
        ref[start:end] = float(np.nanmean(baseline[start:end]))
    if n_days * steps_per_day < len(baseline):
        start = n_days * steps_per_day
        ref[start:] = float(np.nanmean(baseline[start:]))
    return ref


def _load_district_target(climate: str, start_step: int, n_steps: int) -> np.ndarray:
    target_path = REPO_DIR / "data" / "datasets" / f"annex96_ce1_{climate.lower()}_neighborhood" / "district_target.csv"
    if not target_path.exists():
        return np.full(n_steps, np.nan, dtype=float)
    df = pd.read_csv(target_path)
    column = df.columns[0]
    arr = df[column].to_numpy(dtype=float)
    target = arr[start_step : start_step + n_steps]
    if target.size < n_steps:
        target = np.pad(target, (0, n_steps - target.size), constant_values=np.nan)
    return target


def _write_tracking_outputs(
    *,
    result_dir: Path,
    output_dir: Path,
    label: str,
    climate: str,
    test_month: int,
    controlled_load: np.ndarray,
    baseline_load: np.ndarray,
    district_target: np.ndarray,
    computed_reference: np.ndarray,
    target_source: str,
    steps_per_day: int,
    controlled_label: str = "Controlled load",
    baseline_label: str = "Baseline (no storage)",
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(controlled_load)
    hours = np.arange(n, dtype=int)
    days = hours // steps_per_day + 1
    hour_of_day = hours % steps_per_day

    if target_source == "district":
        plotted_target = district_target
        target_label = "District Target"
    elif target_source == "computed":
        plotted_target = computed_reference
        target_label = "Computed Daily Reference"
    else:
        plotted_target = district_target
        target_label = "District Target"

    df = pd.DataFrame(
        {
            "hour": hours,
            "day": days,
            "hour_of_day": hour_of_day,
            "controlled_load": controlled_load,
            "baseline_load": baseline_load,
            "district_target_load": district_target,
            "computed_daily_reference_load": computed_reference,
            "plotted_target_load": plotted_target,
            "tracking_error_to_plotted_target": controlled_load - plotted_target,
            "tracking_error_to_computed_reference": controlled_load - computed_reference,
        }
    )
    csv_path = output_dir / "test_load_tracking_timeseries.csv"
    df.to_csv(csv_path, index=False)

    month_name = MONTH_NAMES.get(test_month, str(test_month))
    full_path = output_dir / "test_load_tracking_full.png"
    week_path = output_dir / "test_load_tracking_week1.png"

    _plot_tracking(
        path=full_path,
        title=f"{label} | {climate} {month_name} Test - Full Simulation Period",
        x=hours,
        controlled=controlled_load,
        baseline=baseline_load,
        target=plotted_target,
        target_label=target_label,
        controlled_label=controlled_label,
        baseline_label=baseline_label,
    )

    week_n = min(7 * steps_per_day, n)
    _plot_tracking(
        path=week_path,
        title=f"{label} | {climate} {month_name} Test - First Week",
        x=hours[:week_n],
        controlled=controlled_load[:week_n],
        baseline=baseline_load[:week_n],
        target=plotted_target[:week_n],
        target_label=target_label,
        controlled_label=controlled_label,
        baseline_label=baseline_label,
    )

    summary_path = output_dir / "test_load_tracking_summary.json"
    summary = {
        "result_dir": str(result_dir),
        "label": label,
        "climate": climate,
        "test_month": test_month,
        "test_month_name": month_name,
        "n_steps": int(n),
        "target_source": target_source,
        "controlled_mean": _safe_float(np.nanmean(controlled_load)),
        "baseline_mean": _safe_float(np.nanmean(baseline_load)),
        "district_target_mean": _safe_float(np.nanmean(district_target)),
        "computed_reference_mean": _safe_float(np.nanmean(computed_reference)),
        "tracking_mae_to_plotted_target": _safe_float(np.nanmean(np.abs(controlled_load - plotted_target))),
        "tracking_rmse_to_plotted_target": _safe_float(
            np.sqrt(np.nanmean(np.square(controlled_load - plotted_target)))
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "timeseries_csv": str(csv_path),
        "full_png": str(full_path),
        "week1_png": str(week_path),
        "summary_json": str(summary_path),
    }


def _plot_tracking(
    *,
    path: Path,
    title: str,
    x: np.ndarray,
    controlled: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    target_label: str,
    controlled_label: str,
    baseline_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.plot(x, baseline, color="#4e79a7", linewidth=1.0, alpha=0.75, label=baseline_label)
    ax.plot(x, controlled, color="#f28e2b", linewidth=1.1, label=controlled_label)
    ax.plot(x, target, color="#ff5c5c", linewidth=1.5, linestyle="--", label=target_label)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time step [hour]")
    ax.set_ylabel("District Load [kWh]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _safe_float(value: Any) -> Optional[float]:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _run_one(
    result_dir: Path,
    output_root: Optional[Path],
    target_source: str,
    steps_per_day: int,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    result_dir = result_dir.resolve()
    module_name = _resolve_module_name(result_dir)
    output_dir = (
        (output_root / result_dir.name).resolve()
        if output_root is not None
        else (result_dir / "load_tracking_eval").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    if result_dir.name.startswith("rllib_sac"):
        test_result = _run_rllib_sac_test(result_dir, output_dir, overrides)
    elif result_dir.name.startswith("sb3_independent"):
        test_result = _run_sb3_test(module_name, result_dir, overrides)
    else:
        test_result = _run_module_test_only(module_name, result_dir, output_dir, overrides)

    controlled_load, baseline_load = _extract_series(test_result)

    run_cfg = _load_json(result_dir / "run_config.json")
    climate = str(overrides.get("climate") or run_cfg.get("climate") or "VT")
    test_month = int(overrides.get("test_month") or run_cfg.get("test_month") or (2 if climate == "VT" else 9))
    start_step = MONTH_STARTS[test_month]
    district_target = _load_district_target(climate, start_step, len(controlled_load))
    computed_reference = _computed_daily_reference(baseline_load, steps_per_day)

    paths = _write_tracking_outputs(
        result_dir=result_dir,
        output_dir=output_dir,
        label=result_dir.name,
        climate=climate,
        test_month=test_month,
        controlled_load=controlled_load,
        baseline_load=baseline_load,
        district_target=district_target,
        computed_reference=computed_reference,
        target_source=target_source,
        steps_per_day=steps_per_day,
        controlled_label=result_dir.name,
    )
    return {
        "result_dir": str(result_dir),
        "module": module_name,
        "status": "ok",
        **paths,
    }


def _run_rbc_baseline(
    *,
    output_root: Path,
    climate: str,
    test_month: int,
    n_buildings: int,
    target_source: str,
    steps_per_day: int,
) -> Dict[str, Any]:
    from citylearn.agents.rbc import BasicBatteryRBC
    from citylearn.citylearn import CityLearnEnv
    from mappo.utils import resolve_reference_baseline_series

    dataset_name = f"annex96_ce1_{climate.lower()}_neighborhood"
    dataset_dir = REPO_DIR / "data" / "datasets" / dataset_name
    schema_path = dataset_dir / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    test_start = MONTH_STARTS[test_month]
    test_end = MONTH_ENDS[test_month]
    env = CityLearnEnv(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=False,
        buildings=list(range(n_buildings)),
        simulation_start_time_step=test_start,
        simulation_end_time_step=test_end,
        episode_time_steps=test_end - test_start + 1,
    )

    rbc_loads: List[float] = []
    try:
        agent = BasicBatteryRBC(env)
        observations, _ = env.reset()
        terminated = False

        while not terminated:
            actions = agent.predict(observations, deterministic=True)
            observations, _rewards, terminated, truncated, _info = env.step(actions)

            try:
                rbc_loads.append(
                    float(sum(float(b.net_electricity_consumption[-1]) for b in env.buildings))
                )
            except Exception:
                rbc_loads.append(float("nan"))

            if truncated:
                break

        rbc_load = np.asarray(rbc_loads, dtype=float)
        baseline_load = np.asarray(resolve_reference_baseline_series(env), dtype=float)[: len(rbc_load)]
        if baseline_load.size == 0:
            baseline_load = np.full_like(rbc_load, np.nan, dtype=float)

        district_target = _load_district_target(climate, test_start, len(rbc_load))
        computed_reference = _computed_daily_reference(baseline_load, steps_per_day)

        # This is a fresh evaluation produced by this script, not an existing
        # training-result folder.
        month_slug = MONTH_NAMES.get(test_month, str(test_month)).lower()
        label = f"rbc_baseline_{climate.lower()}_{month_slug}"
        output_dir = output_root / label
        paths = _write_tracking_outputs(
            result_dir=output_dir,
            output_dir=output_dir,
            label=label,
            climate=climate,
            test_month=test_month,
            controlled_load=rbc_load,
            baseline_load=baseline_load[: len(rbc_load)],
            district_target=district_target,
            computed_reference=computed_reference,
            target_source=target_source,
            steps_per_day=steps_per_day,
            controlled_label="BasicBatteryRBC",
            baseline_label="Baseline (no storage)",
        )
        return {
            "result_dir": str(output_dir),
            "module": "citylearn.agents.rbc.BasicBatteryRBC",
            "status": "ok",
            **paths,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def _write_batch_summary(rows: Sequence[Dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "load_tracking_eval_summary.csv"
    fieldnames = [
        "status",
        "result_dir",
        "module",
        "timeseries_csv",
        "full_png",
        "week1_png",
        "summary_json",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run trained CE1 controllers on the test month and export RBC-style "
            "portfolio load tracking figures."
        )
    )
    parser.add_argument(
        "--selected",
        default="selected_result_folders.md",
        help="Markdown file containing bullet-list result directories.",
    )
    parser.add_argument(
        "--result_dir",
        action="append",
        default=None,
        help="Specific result directory to process. Can be passed multiple times. Overrides --selected.",
    )
    parser.add_argument(
        "--include_not_recommended",
        action="store_true",
        help="Also process result directories under the not-recommended section.",
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT_NAME,
        help=f"Common output directory, relative to repo root by default. Default: {DEFAULT_OUTPUT_ROOT_NAME}.",
    )
    parser.add_argument("--climate", choices=["VT", "TX"], default=None)
    parser.add_argument("--n_buildings", type=int, default=25)
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month", type=int, default=2, help="Default is February for VT.")
    parser.add_argument("--steps_per_day", type=int, default=24)
    parser.add_argument(
        "--target_source",
        choices=["district", "computed"],
        default="district",
        help="Target line in the PNG. 'district' matches rbc_baseline_comparison_full style.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue processing remaining result directories if one run fails.",
    )
    parser.add_argument(
        "--skip_rbc",
        action="store_true",
        help="Do not generate the BasicBatteryRBC baseline tracking figure.",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("WANDB_MODE", "disabled")
    args = parse_args()

    if args.result_dir:
        result_dirs = [(REPO_DIR / path).resolve() for path in args.result_dir]
    else:
        result_dirs = _read_selected_result_dirs(
            (REPO_DIR / args.selected).resolve(),
            include_not_recommended=args.include_not_recommended,
        )

    if not result_dirs:
        print("No result directories found.", file=sys.stderr)
        return 2

    output_root_arg = Path(args.output_root)
    output_root = output_root_arg if output_root_arg.is_absolute() else REPO_DIR / output_root_arg
    output_root = output_root.resolve()
    batch_summary_root = output_root
    overrides = {
        "climate": args.climate,
        "train_month": args.train_month,
        "test_month": args.test_month,
    }

    rows: List[Dict[str, Any]] = []
    if not args.skip_rbc:
        rbc_climate = args.climate or "VT"
        print(f"[RBC] BasicBatteryRBC | {rbc_climate} | {MONTH_NAMES.get(args.test_month, args.test_month)}")
        try:
            row = _run_rbc_baseline(
                output_root=output_root,
                climate=rbc_climate,
                test_month=args.test_month,
                n_buildings=args.n_buildings,
                target_source=args.target_source,
                steps_per_day=args.steps_per_day,
            )
            print(f"  ok -> {row['full_png']}")
        except Exception as exc:
            row = {
                "status": "error",
                "result_dir": str(
                    output_root
                    / f"rbc_baseline_{rbc_climate.lower()}_{MONTH_NAMES.get(args.test_month, str(args.test_month)).lower()}"
                ),
                "module": "citylearn.agents.rbc.BasicBatteryRBC",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            print(f"  error: {row['error']}", file=sys.stderr)
            if not args.continue_on_error:
                _write_batch_summary(rows, batch_summary_root)
                return 1
        else:
            rows.append(row)

    for idx, result_dir in enumerate(result_dirs, start=1):
        print(f"[{idx}/{len(result_dirs)}] {result_dir}")
        try:
            row = _run_one(
                result_dir=result_dir,
                output_root=output_root,
                target_source=args.target_source,
                steps_per_day=args.steps_per_day,
                overrides=overrides,
            )
            print(f"  ok -> {row['full_png']}")
        except Exception as exc:
            row = {
                "status": "error",
                "result_dir": str(result_dir),
                "module": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            print(f"  error: {row['error']}", file=sys.stderr)
            if not args.continue_on_error:
                _write_batch_summary(rows, batch_summary_root)
                return 1
        else:
            rows.append(row)

    _write_batch_summary(rows, batch_summary_root)
    print(f"Batch summary -> {batch_summary_root / 'load_tracking_eval_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
