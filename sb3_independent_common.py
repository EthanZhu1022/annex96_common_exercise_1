from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from annex96_reporting import (
    compute_secondary_daily_tables,
    export_secondary_daily_metrics,
    save_secondary_daily_metrics_plot,
)
from stable_baselines3.common.monitor import Monitor

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from independent_sac.train import (
    REPO_DIR,
    _CLIMATE_DEFAULTS,
    _MONTH_ENDS,
    _MONTH_NAMES,
    _MONTH_STARTS,
    _export_test_metrics,
    _filter_log_values,
    _make_backup_dir,
    _run_daily_pipeline,
    build_env,
    compute_primary_metric_tables,
    extract_episode_kpis,
    get_soc_stats,
    print_repro_metadata,
    seed_everything,
)
from mappo.utils import resolve_reference_baseline_series

try:
    import wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    wandb = None


def _safe_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def save_plots(
    rewards: List[float],
    primary_metrics: List[Dict[str, Any]],
    save_dir: Path,
    algorithm_label: str,
) -> None:
    if not rewards:
        return

    episodes = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{algorithm_label} - Primary Metrics", fontsize=14)

    axes[0, 0].plot(episodes, rewards, color="#4e79a7")
    axes[0, 0].set_title("Portfolio Reward Sum")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].grid(True, alpha=0.3)

    metric_specs = [
        ("primary/load_tracking/nmbe_pct", "Portfolio NMBE", "%", "#e15759"),
        ("primary/load_tracking/cv_rmse_pct", "Portfolio CV-RMSE", "%", "#f28e2b"),
        ("primary/thermal_comfort/portfolio_exceedance_pct", "Portfolio Exceedance Hours", "%", "#59a14f"),
    ]
    for ax, (key, title, ylabel, color) in zip(axes.flat[1:], metric_specs):
        values = [_safe_scalar(metrics.get(key)) for metrics in primary_metrics]
        valid = [(ep, value) for ep, value in zip(episodes, values) if value is not None]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, color=color, linewidth=1.5)
        if key.endswith("nmbe_pct"):
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(str(out), dpi=110)
    plt.close()


def _save_daily_metrics_plot(
    daily_df: pd.DataFrame,
    comfort_df: pd.DataFrame,
    save_dir: Path,
    climate: str,
    month_name: str,
    algorithm_label: str,
) -> Path:
    days = daily_df["day"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{algorithm_label} - Primary Metrics | {climate} | {month_name}", fontsize=13)

    def _line(ax: plt.Axes, column: str, title: str, ylabel: str, color: str) -> None:
        values = daily_df[column].tolist()
        ax.plot(days, values, marker="o", markersize=3, color=color, linewidth=1.2)
        ax.fill_between(days, values, alpha=0.15, color=color)
        mean_value = float(np.nanmean(values))
        ax.axhline(mean_value, color=color, linestyle="--", linewidth=0.8, alpha=0.7, label=f"mean={mean_value:.2f}")
        ax.set_title(title)
        ax.set_xlabel("Day of test month")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    _line(axes[0, 0], "nmbe_pct", "Daily NMBE", "%", "#e15759")
    axes[0, 0].axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    _line(axes[0, 1], "cv_rmse_pct", "Daily CV-RMSE", "%", "#f28e2b")
    _line(
        axes[1, 0],
        "temperature_exceedance_pct_portfolio",
        "Daily Portfolio Exceedance Hours",
        "%",
        "#59a14f",
    )

    ax_box = axes[1, 1]
    if comfort_df.empty:
        ax_box.text(0.5, 0.5, "No comfort data", ha="center", va="center")
        ax_box.axis("off")
    else:
        values = comfort_df["temperature_exceedance_hours"].tolist()
        ax_box.boxplot(
            values,
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "#4e79a7", "alpha": 0.45},
        )
        ax_box.set_title("Per-Building Exceedance Hours")
        ax_box.set_ylabel("Hours")
        ax_box.set_xticks([1])
        ax_box.set_xticklabels(["Buildings"])
        ax_box.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(save_dir) / "test_daily_metrics.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    return out


def _run_daily_pipeline(
    test_result: Dict[str, Any],
    cfg: Any,
    save_dir: Path,
    use_wandb: bool,
    algorithm_label: str,
) -> Optional[pd.DataFrame]:
    daily_records = test_result.get("_daily_primary_metrics")
    comfort_records = test_result.get("_building_comfort_metrics")
    step_loads = test_result.get("_step_portfolio_loads", [])
    baseline_loads = test_result.get("_step_portfolio_loads_baseline", [])
    if not daily_records and not step_loads:
        return None

    daily_df = pd.DataFrame(daily_records or [])
    comfort_df = pd.DataFrame(comfort_records or [])
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    if not daily_df.empty:
        _save_daily_metrics_plot(daily_df, comfort_df, save_dir, cfg.climate, month_name, algorithm_label)

    daily_csv = Path(save_dir) / "test_daily_metrics.csv"
    daily_df.to_csv(daily_csv, index=False)

    comfort_csv = Path(save_dir) / "test_building_comfort_metrics.csv"
    comfort_df.to_csv(comfort_csv, index=False)

    secondary_flexible_df, secondary_baseline_df = compute_secondary_daily_tables(
        step_loads,
        baseline_loads,
        steps_per_day=24,
    ) if step_loads and baseline_loads else (
        compute_secondary_daily_tables(step_loads, [], steps_per_day=24)[0] if step_loads else pd.DataFrame(),
        pd.DataFrame(),
    )

    if not secondary_flexible_df.empty or not secondary_baseline_df.empty:
        save_secondary_daily_metrics_plot(
            secondary_flexible_df,
            secondary_baseline_df,
            save_dir,
            cfg.climate,
            month_name,
            algorithm_label,
        )
        export_secondary_daily_metrics(secondary_flexible_df, secondary_baseline_df, save_dir)

    if use_wandb and _WANDB_OK:
        wandb.define_metric("test_day")
        wandb.define_metric("test/daily_primary/*", step_metric="test_day")
        for _, row in daily_df.iterrows():
            wandb.log(
                _filter_log_values(
                    {
                        "test_day": int(row["day"]),
                        "test/daily_primary/nmbe_pct": row["nmbe_pct"],
                        "test/daily_primary/cv_rmse_pct": row["cv_rmse_pct"],
                        "test/daily_primary/temperature_exceedance_pct_portfolio": row["temperature_exceedance_pct_portfolio"],
                        "test/daily_primary/temperature_exceedance_hours_total": row["temperature_exceedance_hours_total"],
                    }
                )
            )
        if not secondary_flexible_df.empty:
            wandb.define_metric("test/daily/*", step_metric="test_day")
            for _, row in secondary_flexible_df.iterrows():
                wandb.log(
                    _filter_log_values(
                        {
                            "test_day": int(row["day"]),
                            "test/daily/ramping": row["ramping"],
                            "test/daily/peak": row["daily_peak"],
                            "test/daily/load_factor": row["load_factor"],
                            "test/daily/pvr": row["pvr"],
                            "test/daily/energy": row["energy"],
                        }
                    )
                )
        if not secondary_baseline_df.empty:
            wandb.define_metric("test/daily_baseline/*", step_metric="test_day")
            for _, row in secondary_baseline_df.iterrows():
                wandb.log(
                    _filter_log_values(
                        {
                            "test_day": int(row["day"]),
                            "test/daily_baseline/ramping": row["ramping"],
                            "test/daily_baseline/peak": row["daily_peak"],
                            "test/daily_baseline/load_factor": row["load_factor"],
                            "test/daily_baseline/pvr": row["pvr"],
                            "test/daily_baseline/energy": row["energy"],
                        }
                    )
                )

    return daily_df


def _config_to_dict(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    return dict(vars(cfg))


def build_sb3_single_env(
    cfg: Any,
    building_idx: int,
    start_step: int,
    end_step: int,
    seed: int,
):
    dataset_name = f"annex96_ce1_{cfg.climate.lower()}_neighborhood"
    dataset_dir = REPO_DIR / "data" / "datasets" / dataset_name
    schema_path = dataset_dir / "schema.json"

    env_kwargs: Dict[str, Any] = {
        "schema": str(schema_path),
        "root_directory": str(dataset_dir),
        "central_agent": True,
        "buildings": [building_idx],
        "simulation_start_time_step": start_step,
        "simulation_end_time_step": end_step,
        "episode_time_steps": (
            cfg.episode_time_steps
            if getattr(cfg, "episode_time_steps", None) is not None
            else end_step - start_step + 1
        ),
    }

    base_env = CityLearnEnv(**env_kwargs)
    env = NormalizedObservationWrapper(base_env)
    env = StableBaselines3Wrapper(env)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def _extract_logger_means(models: Sequence[Any], key_map: Dict[str, str]) -> Dict[str, float]:
    values: Dict[str, List[float]] = {target: [] for target in key_map.values()}

    for model in models:
        logger_values = getattr(getattr(model, "logger", None), "name_to_value", {}) or {}
        for sb3_key, target in key_map.items():
            value = logger_values.get(sb3_key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_f):
                values[target].append(value_f)

    return {
        key: float(np.mean(val_list))
        for key, val_list in values.items()
        if val_list
    }


def evaluate_portfolio_models(
    models: Sequence[Any],
    cfg: Any,
    start_step: int,
    end_step: int,
    month: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    env, base_env = build_env(cfg, start_step=start_step, end_step=end_step)
    try:
        n_buildings = len(base_env.buildings)
        obs_list, _ = env.reset(seed=cfg.seed)

        per_building_rewards: List[float] = [0.0] * n_buildings
        step_portfolio_rewards: List[float] = []
        step_portfolio_loads: List[float] = []

        while not base_env.terminated:
            env_actions: List[List[float]] = []
            for i, model in enumerate(models):
                action, _ = model.predict(np.asarray(obs_list[i], dtype=np.float32), deterministic=deterministic)
                env_actions.append(np.asarray(action, dtype=np.float32).tolist())

            next_obs_list, rewards, terminated, truncated, _ = env.step(env_actions)

            step_reward = float(np.sum(rewards))
            for i in range(n_buildings):
                per_building_rewards[i] += float(rewards[i])
            step_portfolio_rewards.append(step_reward)

            try:
                net_load = sum(float(b.net_electricity_consumption[-1]) for b in base_env.buildings)
                step_portfolio_loads.append(net_load)
            except Exception:
                step_portfolio_loads.append(float("nan"))

            obs_list = next_obs_list
            if terminated or truncated:
                break

        kpis = extract_episode_kpis(base_env)
        soc_stats = get_soc_stats(base_env)
        primary_metrics, daily_primary_df, comfort_building_df = compute_primary_metric_tables(base_env, month)

        result: Dict[str, Any] = {
            "portfolio/reward_sum": float(np.sum(per_building_rewards)),
            "portfolio/reward_mean": float(np.mean(per_building_rewards)),
            "step_reward_mean": float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0,
            **primary_metrics,
            **kpis,
            **soc_stats,
            "_step_portfolio_loads": step_portfolio_loads,
            "_step_portfolio_loads_baseline": resolve_reference_baseline_series(base_env)[: len(step_portfolio_loads)].tolist(),
            "_daily_primary_metrics": daily_primary_df.to_dict(orient="records"),
            "_building_comfort_metrics": comfort_building_df.to_dict(orient="records"),
        }

        for i, reward in enumerate(per_building_rewards):
            result[f"building_{i}/reward"] = reward

        return result
    finally:
        try:
            env.close()
        except Exception:
            pass


def _prefix_public_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    public: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("_"):
            continue
        public[f"{prefix}/{key}"] = value
    return public


def save_sb3_checkpoint(
    models: Sequence[Any],
    save_dir: Path,
    cfg: Any,
    metrics: Optional[Dict[str, Any]] = None,
    backup_dir: Optional[Path] = None,
) -> None:
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for i, model in enumerate(models):
        model_path = save_dir / f"building_{i}_model"
        model.save(str(model_path))
        saved_files.append(Path(str(model_path) + ".zip"))

    run_config_path = save_dir / "run_config.json"
    run_config_path.write_text(json.dumps(_config_to_dict(cfg), indent=2))
    saved_files.append(run_config_path)

    if metrics is not None:
        metrics_path = save_dir / "latest_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        saved_files.append(metrics_path)

    if backup_dir is not None:
        backup_dir = Path(backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        for path in saved_files:
            if path.exists():
                (backup_dir / path.name).write_bytes(path.read_bytes())


def train_sb3_independent(
    cfg: Any,
    algorithm_label: str,
    model_builder: Callable[[Any, Any, int, int], Any],
    loss_key_map: Dict[str, str],
) -> Sequence[Any]:
    defaults = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    train_month = cfg.train_month or defaults.get("train_month")
    test_month = cfg.test_month or defaults.get("test_month")

    if train_month is None or train_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid train_month={train_month}. Must be 1-12.")
    if test_month is not None and test_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid test_month={test_month}. Must be 1-12.")

    train_start = _MONTH_STARTS[train_month]
    train_end = _MONTH_ENDS[train_month]
    test_start = _MONTH_STARTS[test_month] if test_month is not None else None
    test_end = _MONTH_ENDS[test_month] if test_month is not None else None
    episode_steps = train_end - train_start + 1

    cfg.train_month = train_month
    cfg.test_month = test_month

    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    repro_meta = print_repro_metadata(cfg, torch.device(device), train_start, train_end, test_start, test_end)

    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = _config_to_dict(cfg)
        cfg_dict.update(
            {
                "algorithm": algorithm_label,
                "train_start_step": train_start,
                "train_end_step": train_end,
                "test_start_step": test_start,
                "test_end_step": test_end,
            }
        )
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg_dict)
        wandb.define_metric("episode")
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("test/*", step_metric="episode")

    train_envs: List[Any] = []
    models: List[Any] = []
    for building_idx in range(cfg.n_buildings):
        env = build_sb3_single_env(
            cfg=cfg,
            building_idx=building_idx,
            start_step=train_start,
            end_step=train_end,
            seed=cfg.seed + building_idx,
        )
        model = model_builder(cfg, env, building_idx, episode_steps)
        train_envs.append(env)
        models.append(model)

    print("=" * 65)
    print(
        f"{algorithm_label} | Climate: {cfg.climate} | buildings: {cfg.n_buildings} | "
        f"episodes: {cfg.n_episodes}"
    )
    print("=" * 65)

    all_rewards: List[float] = []
    all_primary_metrics: List[Dict[str, Any]] = []

    for episode in range(1, cfg.n_episodes + 1):
        for model in models:
            model.learn(total_timesteps=episode_steps, reset_num_timesteps=False, progress_bar=False)

        train_metrics = evaluate_portfolio_models(
            models=models,
            cfg=cfg,
            start_step=train_start,
            end_step=train_end,
            month=train_month,
            deterministic=True,
        )

        loss_metrics = _extract_logger_means(models, loss_key_map)
        all_rewards.append(float(train_metrics["portfolio/reward_sum"]))
        primary_metrics = {
            key: value for key, value in train_metrics.items() if key.startswith("primary/")
        }
        all_primary_metrics.append(primary_metrics)

        nmbe = primary_metrics.get("primary/load_tracking/nmbe_pct", float("nan"))
        cv_rmse = primary_metrics.get("primary/load_tracking/cv_rmse_pct", float("nan"))

        if episode % 10 == 0 or episode == 1:
            print(
                f"[train] Ep {episode:4d} | rew_sum {train_metrics['portfolio/reward_sum']:9.2f} | "
                f"NMBE {nmbe:.3f}% | CV-RMSE {cv_rmse:.3f}%"
            )

        if use_wandb:
            log_dict: Dict[str, Any] = {"episode": episode}
            log_dict.update(_prefix_public_metrics(train_metrics, "train"))
            for key, value in loss_metrics.items():
                log_dict[f"train/loss/{key}"] = value
            wandb.log(_filter_log_values(log_dict), step=episode)

        if episode % cfg.save_every == 0 or episode == cfg.n_episodes:
            save_plots(all_rewards, all_primary_metrics, save_dir, algorithm_label)
            save_sb3_checkpoint(models, save_dir, cfg)

    test_result_prefixed: Optional[Dict[str, Any]] = None
    raw_test_metrics: Optional[Dict[str, Any]] = None
    if cfg.do_test and test_start is not None and test_end is not None and test_month is not None:
        raw_test_metrics = evaluate_portfolio_models(
            models=models,
            cfg=cfg,
            start_step=test_start,
            end_step=test_end,
            month=test_month,
            deterministic=True,
        )
        test_result_prefixed = _prefix_public_metrics(raw_test_metrics, "test")
        test_result_prefixed["episode"] = cfg.n_episodes

        if use_wandb:
            wandb.log(_filter_log_values(test_result_prefixed), step=cfg.n_episodes)

        _export_test_metrics(test_result_prefixed, cfg, save_dir, test_start, test_end, test_month)
        _run_daily_pipeline(
            test_result_prefixed
            | {
                "_step_portfolio_loads": raw_test_metrics.get("_step_portfolio_loads", []),
                "_step_portfolio_loads_baseline": raw_test_metrics.get("_step_portfolio_loads_baseline", []),
                "_daily_primary_metrics": raw_test_metrics.get("_daily_primary_metrics", []),
                "_building_comfort_metrics": raw_test_metrics.get("_building_comfort_metrics", []),
            },
            cfg,
            save_dir,
            use_wandb,
            algorithm_label,
        )

    save_plots(all_rewards, all_primary_metrics, save_dir, algorithm_label)

    final_metrics: Dict[str, Any] = {
        "algorithm": algorithm_label,
        "train": {
            "n_episodes": cfg.n_episodes,
            "last_reward_sum": all_rewards[-1] if all_rewards else None,
            "train_month": train_month,
            "train_steps": f"{train_start}-{train_end}",
            "repro": repro_meta,
            **(all_primary_metrics[-1] if all_primary_metrics else {}),
        },
    }
    if test_result_prefixed is not None and raw_test_metrics is not None:
        public_test = {k: v for k, v in test_result_prefixed.items() if not k.startswith("_")}
        final_metrics["test"] = {
            "test_month": test_month,
            "test_steps": f"{test_start}-{test_end}",
            **public_test,
        }

    save_sb3_checkpoint(
        models=models,
        save_dir=save_dir,
        cfg=cfg,
        metrics=final_metrics,
        backup_dir=_make_backup_dir(cfg),
    )

    backup_dir = _make_backup_dir(cfg)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in list(save_dir.glob("*.json")) + list(save_dir.glob("*.png")) + list(save_dir.glob("*.csv")):
        (backup_dir / path.name).write_bytes(path.read_bytes())

    for env in train_envs:
        try:
            env.close()
        except Exception:
            pass

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print(f"  Artifacts: {save_dir}/")
    print(f"  Backup:    {backup_dir}/")
    return models
