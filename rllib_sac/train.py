"""
RLlib SAC Baseline — Training + Evaluation Script
===================================================

Uses RLlib's native SAC trainer with one independent policy per building.
No SAC logic is reimplemented from scratch; RLlib's SACConfig/SAC handles
replay, twin-Q updates, polyak averaging, and automatic entropy tuning.

Architecture
------------
- Environment: CityLearnMultiAgentEnv (see env.py) wraps CityLearnEnv so
  RLlib can drive it as a MultiAgentEnv.
- Policies: one policy per building ("building_i"), mapped by name.
  RLlib trains all policies concurrently but each only sees its own obs/reward,
  making training independent per building (no communication).
- SAC config: standard RLlib SACConfig with twin_q=True, automatic alpha.

W&B logging
-----------
- CE1Callback records per-building rewards and CityLearn KPIs as RLlib custom
  metrics.
- train() logs README-aligned primary/secondary metrics and SAC loss metrics
  using the same public W&B keys as SB3 Independent SAC and MAPPO.

Output artifacts (in save_dir/):
    checkpoint/        — RLlib checkpoint (actor/critic weights for all agents)
    run_config.json    — full hyperparameter snapshot
    latest_metrics.json
    test_metrics.json  — test-episode KPIs (comparable with other baselines)
    test_metrics.csv
    test_daily_metrics.png
    test_daily_metrics.csv
    test_daily_secondary_metrics.png
    test_daily_secondary_flexible_metrics.csv
    test_daily_secondary_baseline_metrics.csv
    training_curves.png

Run:
    python -m rllib_sac.train                        # VT: Jan train, Feb test
    python -m rllib_sac.train --climate TX           # TX: Aug train, Sep test
    python -m rllib_sac.train --climate TX --n_iterations 20 --seed 0
    python -m rllib_sac.train --no_test

Dependencies: ray[rllib]==2.10.0, pyarrow<14, gymnasium==0.28.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from annex96_reporting import (
    build_readme_secondary_daily_log,
    compute_secondary_daily_tables,
    export_secondary_daily_metrics,
    save_secondary_daily_metrics_plot,
)

# Ensure local CityLearn copy takes precedence
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

import ray
from ray import tune
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

from rllib_sac.env import CityLearnMultiAgentEnv, _build_citylearn
from mappo.utils import extract_episode_kpis, get_soc_stats, resolve_reference_baseline_series
from training_progress import ProgressTimer

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed — W&B logging disabled.")


# ---------------------------------------------------------------------------
# Month → step mappings  (identical to mappo/train.py and independent_sac/)
# ---------------------------------------------------------------------------

_MONTH_STARTS: Dict[int, int] = {
    1: 0,     2: 744,   3: 1416,  4: 2160,
    5: 2880,  6: 3624,  7: 4344,  8: 5088,
    9: 5832,  10: 6552, 11: 7296, 12: 8016,
}
_MONTH_ENDS: Dict[int, int] = {
    1: 743,   2: 1415,  3: 2159,  4: 2879,
    5: 3623,  6: 4343,  7: 5087,  8: 5831,
    9: 6551,  10: 7295, 11: 8015, 12: 8759,
}
_MONTH_NAMES: Dict[int, str] = {
    1: "January",   2: "February",  3: "March",    4: "April",
    5: "May",       6: "June",      7: "July",     8: "August",
    9: "September", 10: "October",  11: "November", 12: "December",
}
_CLIMATE_DEFAULTS: Dict[str, Dict[str, int]] = {
    "VT": {"train_month": 1, "test_month": 2},
    "TX": {"train_month": 8, "test_month": 9},
}
ALGORITHM_LABEL = "RLlib Independent SAC"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Dataset
    climate:     str = "VT"
    n_buildings: int = 25

    # RLlib SAC hyperparameters (native RLlib names preserved for clarity)
    hidden_sizes:       List[int] = None  # type: ignore[assignment]
    lr:                 float = 3e-4
    gamma:              float = 0.99
    tau:                float = 5e-3   # polyak coefficient
    initial_alpha:      float = 0.2
    buffer_size:        int   = 100_000
    train_batch_size:   int   = 256
    # Steps collected before first gradient update (RLlib param)
    num_steps_sampled_before_learning_starts: int = 1000

    # RLlib iteration / rollout settings
    n_iterations:        int = 100   # number of RLlib train() calls
    rollout_fragment_len: int = 200  # steps per rollout worker per iteration
    num_workers:         int = 0     # 0 = collect on local worker (no forking)
    target_network_update_freq: int = 1  # steps between target net soft updates

    # Time windows
    train_month:        Optional[int] = None
    test_month:         Optional[int] = None
    episode_time_steps: Optional[int] = None

    # Workflow
    do_test: bool = True

    # Logging
    seed:          int          = 42
    wandb_project: str          = "annex96-ce1"
    wandb_name:    str          = "rllib-independent-sac"
    save_dir:      str          = "results/rllib_sac"
    backup_dir:    Optional[str] = None

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [256, 256]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


def print_repro_metadata(
    cfg: Config, train_start: int, train_end: int,
    test_start: Optional[int], test_end: Optional[int],
) -> Dict:
    meta: Dict = {
        "seed":        cfg.seed,
        "torch":       torch.__version__,
        "ray":         ray.__version__,
        "climate":     cfg.climate,
        "train_month": cfg.train_month,
        "train_steps": f"{train_start}-{train_end}",
        "test_month":  cfg.test_month,
        "test_steps":  f"{test_start}-{test_end}" if test_start else "N/A",
    }
    print("\n" + "=" * 65)
    print("REPRODUCIBILITY METADATA")
    for k, v in meta.items():
        print(f"  {k:<24} {v}")
    print("=" * 65 + "\n")
    return meta


# ---------------------------------------------------------------------------
# WandB + KPI callback
# ---------------------------------------------------------------------------

class CE1Callback(DefaultCallbacks):
    """Record per-episode rewards, KPIs, and primary metrics for RLlib.

    Hooks used:
      on_episode_end   — log per-building rewards and district-level KPIs
                         extracted from CityLearn's env.evaluate().
      on_train_result  — log SAC training losses and alpha from RLlib result.
    """

    def __init__(self):
        super().__init__()
        # Accumulate portfolio metrics across episodes within each iteration.
        # RLlib may run multiple episodes per iteration.
        self._ep_portfolio_rewards: List[float] = []
        self._ep_kpis:              List[Dict]  = []

    def on_episode_end(
        self,
        *,
        worker:   Optional[RolloutWorker] = None,
        base_env: Optional[BaseEnv]       = None,
        policies: Optional[Dict[str, Policy]] = None,
        episode:  Union[Episode, Any],
        env_index: int = 0,
        **kwargs,
    ) -> None:
        """Called when each episode finishes.

        Extracts per-building episode rewards from the RLlib episode object,
        then pulls district-level KPIs from the underlying CityLearn env.
        """
        if base_env is None:
            return

        # Retrieve the CityLearnMultiAgentEnv from RLlib's BaseEnv
        try:
            cl_env: CityLearnMultiAgentEnv = base_env.get_sub_environments()[env_index]
        except Exception:
            return

        # Per-building episode rewards from RLlib's episode object
        agent_rewards: Dict[str, float] = {}
        try:
            for agent_id in sorted(cl_env.get_agent_ids(),
                                   key=lambda s: int(s.split("_")[1])):
                r = episode.agent_rewards.get((agent_id, agent_id), None)
                if r is not None:
                    agent_rewards[agent_id] = float(r)
        except Exception:
            pass

        portfolio_reward = sum(agent_rewards.values())
        self._ep_portfolio_rewards.append(portfolio_reward)

        # CityLearn KPIs from env.evaluate()
        try:
            kpis = extract_episode_kpis(cl_env.base_env)
        except Exception:
            kpis = {}
        self._ep_kpis.append(kpis)

        try:
            primary_metrics, _, _ = compute_primary_metric_tables(
                cl_env.base_env,
                _infer_month_from_window(getattr(cl_env, "_start_step", None), getattr(cl_env, "_end_step", None)),
            )
        except Exception:
            primary_metrics = {}

        # Write into episode custom metrics so RLlib aggregates them
        episode.custom_metrics["portfolio_reward_sum"] = portfolio_reward
        for k, v in kpis.items():
            if v is not None:
                episode.custom_metrics[k] = v
        for k, v in primary_metrics.items():
            if v is not None:
                episode.custom_metrics[k] = v
        for agent_id, r in agent_rewards.items():
            episode.custom_metrics[f"{agent_id}_reward"] = r

    def on_train_result(
        self,
        *,
        algorithm: Any,
        result: Dict,
        **kwargs,
    ) -> None:
        # W&B logging is centralized in train() so public keys stay aligned
        # with SB3 Independent SAC and MAPPO baselines.
        self._ep_portfolio_rewards.clear()
        self._ep_kpis.clear()


def _filter_wandb(d: Dict) -> Dict:
    """Drop None / NaN / inf values before logging."""
    out: Dict = {}
    for k, v in d.items():
        if v is None or k.startswith("_"):
            continue
        if isinstance(v, (dict, list, tuple)):
            continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            continue
        out[k] = v
    return out


def _collect_rllib_custom_metrics(custom_metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    metric_prefix = f"{prefix}/"
    suffix = "_mean"
    return {
        key[:-len(suffix)]: value
        for key, value in custom_metrics.items()
        if key.startswith(metric_prefix) and key.endswith(suffix)
    }


def _extract_rllib_sac_loss_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    learner = result.get("info", {}).get("learner", {})
    aliases = {
        "actor_loss": "actor_loss",
        "policy_loss": "actor_loss",
        "critic_loss": "critic_loss",
        "qf_loss": "critic_loss",
        "alpha_value": "ent_coef",
        "alpha": "ent_coef",
        "alpha_loss": "ent_coef_loss",
        "entropy": "entropy",
    }
    for policy_stats in learner.values():
        if not isinstance(policy_stats, dict):
            continue
        stats = policy_stats.get("learner_stats", policy_stats)
        if not isinstance(stats, dict):
            continue
        losses: Dict[str, float] = {}
        for source_key, target_key in aliases.items():
            if source_key not in stats:
                continue
            value = _safe_scalar(stats[source_key])
            if value is not None:
                losses[target_key] = value
        if losses:
            return losses
    return {}


# ---------------------------------------------------------------------------
# Build RLlib SACConfig
# ---------------------------------------------------------------------------

def build_sac_config(
    cfg:         Config,
    env_config:  Dict,
    agent_ids:   List[str],
    obs_space,
    act_space,
) -> SACConfig:
    """Assemble an RLlib SACConfig for independent per-building SAC.

    One policy is registered per building.  The policy_mapping_fn routes
    each agent's experience to its own policy.  This gives truly independent
    training — each building's replay buffer and networks are separate.

    RLlib-native configuration is used throughout; no custom SAC math.
    """
    from ray.rllib.policy.policy import PolicySpec
    from gymnasium import spaces as gym_spaces

    # Per-building observation and action spaces
    per_building_obs = obs_space
    per_building_act = act_space

    # Policy spec shared across buildings (same architecture, independent weights)
    policy_spec = PolicySpec(
        policy_class = None,  # use default SAC policy
        observation_space = per_building_obs,
        action_space      = per_building_act,
        config = {
            "model": {
                "fcnet_hiddens": cfg.hidden_sizes,
                "fcnet_activation": "relu",
            }
        },
    )

    policies: Dict[str, PolicySpec] = {aid: policy_spec for aid in agent_ids}

    def policy_mapping_fn(agent_id: str, episode=None, worker=None, **kwargs) -> str:
        """Each building always maps to its own dedicated policy."""
        return agent_id

    sac_cfg = (
        SACConfig()
        .environment(
            env=CityLearnMultiAgentEnv,
            env_config=env_config,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(agent_ids),
        )
        .training(
            twin_q             = True,   # clipped double-Q for stability
            initial_alpha      = cfg.initial_alpha,
            tau                = cfg.tau,
            gamma              = cfg.gamma,
            target_network_update_freq = cfg.target_network_update_freq,
            num_steps_sampled_before_learning_starts =
                cfg.num_steps_sampled_before_learning_starts,
            train_batch_size   = cfg.train_batch_size,
            replay_buffer_config = {
                "type":     "MultiAgentPrioritizedReplayBuffer",
                "capacity": cfg.buffer_size,
            },
            optimization_config = {
                "actor_learning_rate":   cfg.lr,
                "critic_learning_rate":  cfg.lr,
                "entropy_learning_rate": cfg.lr,
            },
        )
        .rollouts(
            num_rollout_workers    = cfg.num_workers,
            rollout_fragment_length = cfg.rollout_fragment_len,
        )
        .resources(
            num_gpus = int(torch.cuda.is_available()),
        )
        .callbacks(CE1Callback)
        .framework("torch")
        .debugging(seed=cfg.seed)
    )

    return sac_cfg


# ---------------------------------------------------------------------------
# Primary-metric overrides for Annex96 CE1 reporting
# ---------------------------------------------------------------------------

def _safe_pct(value: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-9:
        return float("nan")
    return float(value / denominator * 100.0)


def _safe_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _infer_month_from_window(start_step: Optional[int], end_step: Optional[int]) -> Optional[int]:
    for month, month_start in _MONTH_STARTS.items():
        month_end = _MONTH_ENDS[month]
        if start_step == month_start and end_step == month_end:
            return month
    return None


def _get_episode_time_resolution(base_env: Any) -> Tuple[int, float]:
    seconds_per_step = float(getattr(base_env, "seconds_per_time_step", 3600) or 3600)
    steps_per_day = max(int(round(24 * 3600 / seconds_per_step)), 1)
    step_hours = seconds_per_step / 3600.0
    return steps_per_day, step_hours


def _get_reference_baseline_series(base_env: Any) -> np.ndarray:
    for attr in [
        "net_electricity_consumption_without_storage_and_partial_load_and_pv",
        "net_electricity_consumption_without_storage_and_pv",
        "net_electricity_consumption_without_storage_and_partial_load",
        "net_electricity_consumption_without_storage",
    ]:
        series = getattr(base_env, attr, None)
        if series is None:
            continue
        arr = np.asarray(series, dtype=float)
        if arr.size > 0:
            return arr
    raise AttributeError("Unable to resolve baseline load series for Primary Metrics.")


def _get_season_comfort_bounds(month: Optional[int]) -> Tuple[str, float, float]:
    if month in {5, 6, 7, 8, 9}:
        return "cooling", 22.0, 26.0
    return "heating", 20.0, 24.0


def compute_primary_metric_tables(
    base_env: Any,
    month: Optional[int],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    steps_per_day, step_hours = _get_episode_time_resolution(base_env)

    actual = np.asarray(base_env.net_electricity_consumption, dtype=float)
    baseline = _get_reference_baseline_series(base_env)
    n_steps = min(len(actual), len(baseline))
    actual = actual[:n_steps]
    baseline = baseline[:n_steps]

    n_days = n_steps // steps_per_day
    actual = actual[:n_days * steps_per_day]
    baseline = baseline[:n_days * steps_per_day]

    daily_rows: List[Dict[str, Any]] = []
    reference_profile: List[float] = []

    for day_idx in range(n_days):
        start = day_idx * steps_per_day
        end = start + steps_per_day
        actual_day = actual[start:end]
        baseline_day = baseline[start:end]
        reference_mean = float(np.nanmean(baseline_day))
        reference_day = np.full_like(actual_day, reference_mean, dtype=float)
        diff = actual_day - reference_day
        rmse = float(np.sqrt(np.nanmean(np.square(diff))))

        daily_rows.append({
            "day": day_idx + 1,
            "reference_mean": reference_mean,
            "actual_mean": float(np.nanmean(actual_day)),
            "actual_energy": float(np.nansum(actual_day)),
            "reference_energy": float(reference_mean * len(actual_day)),
            "nmbe_pct": _safe_pct(float(np.nanmean(diff)), reference_mean),
            "cv_rmse_pct": _safe_pct(rmse, reference_mean),
            "rmse": rmse,
        })
        reference_profile.extend(reference_day.tolist())

    daily_df = pd.DataFrame(daily_rows)
    reference_profile_arr = np.asarray(reference_profile, dtype=float)
    actual_profile_arr = actual[:len(reference_profile_arr)]
    overall_ref_mean = float(np.nanmean(reference_profile_arr)) if reference_profile_arr.size else float("nan")
    overall_diff = actual_profile_arr - reference_profile_arr
    overall_rmse = float(np.sqrt(np.nanmean(np.square(overall_diff)))) if overall_diff.size else float("nan")

    _, comfort_low_c, comfort_high_c = _get_season_comfort_bounds(month)
    building_rows: List[Dict[str, Any]] = []
    daily_exceed_hours_total = np.zeros(n_days, dtype=float)

    for building_idx, building in enumerate(base_env.buildings):
        indoor = np.asarray(building.indoor_dry_bulb_temperature, dtype=float)[:len(reference_profile_arr)]
        indoor = indoor[:n_days * steps_per_day]
        exceed_mask = (indoor < comfort_low_c) | (indoor > comfort_high_c)
        exceed_hours = float(np.sum(exceed_mask) * step_hours)
        total_hours = len(indoor) * step_hours

        building_rows.append({
            "building_id": building_idx,
            "building_name": getattr(building, "name", f"building_{building_idx}"),
            "temperature_exceedance_hours": exceed_hours,
            "temperature_exceedance_pct": _safe_pct(exceed_hours, total_hours),
        })

        if n_days > 0:
            daily_mask = exceed_mask.reshape(n_days, steps_per_day)
            daily_exceed_hours_total += daily_mask.sum(axis=1).astype(float) * step_hours

    comfort_df = pd.DataFrame(building_rows)
    if not daily_df.empty and len(base_env.buildings) > 0:
        portfolio_hours_per_day = len(base_env.buildings) * steps_per_day * step_hours
        daily_df["temperature_exceedance_hours_total"] = daily_exceed_hours_total
        daily_df["temperature_exceedance_pct_portfolio"] = (
            daily_exceed_hours_total / portfolio_hours_per_day * 100.0
        )

    total_exceed_hours = (
        float(comfort_df["temperature_exceedance_hours"].sum())
        if not comfort_df.empty else 0.0
    )
    total_portfolio_hours = len(base_env.buildings) * len(actual_profile_arr) * step_hours

    summary = {
        "primary/load_tracking/nmbe_pct": _safe_pct(
            float(np.nanmean(overall_diff)) if overall_diff.size else float("nan"),
            overall_ref_mean,
        ),
        "primary/load_tracking/cv_rmse_pct": _safe_pct(overall_rmse, overall_ref_mean),
        "primary/load_tracking/reference_mean": overall_ref_mean,
        "primary/load_tracking/actual_mean": (
            float(np.nanmean(actual_profile_arr)) if actual_profile_arr.size else float("nan")
        ),
        "primary/load_tracking/reference_peak": (
            float(np.nanmax(reference_profile_arr)) if reference_profile_arr.size else float("nan")
        ),
        "primary/load_tracking/actual_peak": (
            float(np.nanmax(actual_profile_arr)) if actual_profile_arr.size else float("nan")
        ),
        "primary/thermal_comfort/temperature_exceedance_hours_total": total_exceed_hours,
        "primary/thermal_comfort/temperature_exceedance_hours_mean": (
            float(comfort_df["temperature_exceedance_hours"].mean())
            if not comfort_df.empty else 0.0
        ),
        "primary/thermal_comfort/temperature_exceedance_hours_max": (
            float(comfort_df["temperature_exceedance_hours"].max())
            if not comfort_df.empty else 0.0
        ),
        "primary/thermal_comfort/portfolio_exceedance_pct": _safe_pct(
            total_exceed_hours, total_portfolio_hours
        ),
        "primary/thermal_comfort/lower_bound_c": comfort_low_c,
        "primary/thermal_comfort/upper_bound_c": comfort_high_c,
    }

    return summary, daily_df, comfort_df


def save_plots(
    rewards: List[float],
    primary_metrics: List[Dict[str, Any]],
    save_dir: Path,
) -> None:
    if not rewards:
        return

    iters = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{ALGORITHM_LABEL} - Primary Metrics", fontsize=14)

    axes[0, 0].plot(iters, rewards, color="#4e79a7")
    axes[0, 0].set_title("Mean Episode Reward")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].grid(True, alpha=0.3)

    metric_specs = [
        ("primary/load_tracking/nmbe_pct", "Portfolio NMBE", "%", "#e15759"),
        ("primary/load_tracking/cv_rmse_pct", "Portfolio CV-RMSE", "%", "#f28e2b"),
        ("primary/thermal_comfort/portfolio_exceedance_pct", "Portfolio Exceedance Hours", "%", "#59a14f"),
    ]
    for ax, (key, title, ylabel, color) in zip(axes.flat[1:], metric_specs):
        values = [_safe_scalar(metrics.get(key)) for metrics in primary_metrics]
        valid = [(ep, value) for ep, value in zip(iters, values) if value is not None]
        if valid:
            x, y = zip(*valid)
            ax.plot(x, y, color=color, linewidth=1.5)
        if key.endswith("nmbe_pct"):
            ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(str(out), dpi=110)
    plt.close()
    print(f"  [plot] saved -> {out}")


def save_daily_metrics_plot(
    daily_df: pd.DataFrame,
    comfort_df: pd.DataFrame,
    save_dir: Path,
    climate: str,
    month_name: str,
) -> Path:
    days = daily_df["day"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"{ALGORITHM_LABEL} - Primary Metrics | {climate} | {month_name}",
        fontsize=13,
    )

    def _line(ax: plt.Axes, column: str, title: str, ylabel: str, color: str) -> None:
        values = daily_df[column].tolist()
        ax.plot(days, values, marker="o", markersize=3, color=color, linewidth=1.2)
        ax.fill_between(days, values, alpha=0.15, color=color)
        mean_value = float(np.nanmean(values))
        ax.axhline(
            mean_value,
            color=color,
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
            label=f"mean={mean_value:.2f}",
        )
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
    print(f"  [daily] figure -> {out}")
    return out


def _run_daily_pipeline(
    test_result: Dict,
    cfg: Config,
    save_dir: Path,
    use_wandb: bool,
) -> Optional[pd.DataFrame]:
    daily_records = test_result.get("_daily_primary_metrics")
    comfort_records = test_result.get("_building_comfort_metrics")
    step_loads: List[float] = test_result.get("_step_portfolio_loads", [])
    baseline_loads: List[float] = test_result.get("_step_portfolio_loads_baseline", [])
    if not daily_records and not step_loads:
        print("[warn] No daily records - skipping daily artifacts.")
        return None

    daily_df = pd.DataFrame(daily_records or [])
    comfort_df = pd.DataFrame(comfort_records or [])
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    if not daily_df.empty:
        save_daily_metrics_plot(daily_df, comfort_df, save_dir, cfg.climate, month_name)

    daily_csv = Path(save_dir) / "test_daily_metrics.csv"
    daily_df.to_csv(daily_csv, index=False)
    print(f"  [daily] CSV -> {daily_csv}")

    comfort_csv = Path(save_dir) / "test_building_comfort_metrics.csv"
    comfort_df.to_csv(comfort_csv, index=False)
    print(f"  [daily] building comfort CSV -> {comfort_csv}")

    secondary_flexible_df, secondary_baseline_df = compute_secondary_daily_tables(
        step_loads,
        baseline_loads,
        steps_per_day=24,
    ) if step_loads and baseline_loads else (
        compute_secondary_daily_tables(step_loads, [], steps_per_day=24)[0] if step_loads else pd.DataFrame(),
        pd.DataFrame(),
    )

    if not secondary_flexible_df.empty or not secondary_baseline_df.empty:
        secondary_fig = save_secondary_daily_metrics_plot(
            secondary_flexible_df,
            secondary_baseline_df,
            save_dir,
            cfg.climate,
            month_name,
            ALGORITHM_LABEL,
        )
        if secondary_fig is not None:
            print(f"  [daily] secondary figure -> {secondary_fig}")
        flexible_csv, baseline_csv = export_secondary_daily_metrics(
            secondary_flexible_df,
            secondary_baseline_df,
            save_dir,
        )
        if flexible_csv is not None:
            print(f"  [daily] secondary flexible CSV -> {flexible_csv}")
        if baseline_csv is not None:
            print(f"  [daily] secondary baseline CSV -> {baseline_csv}")

    if use_wandb:
        wandb.define_metric("test_day")
        if not secondary_flexible_df.empty:
            wandb.define_metric("test/daily/*", step_metric="test_day")
            wandb.define_metric("test/daily_secondary/*", step_metric="test_day")
            for _, row in secondary_flexible_df.iterrows():
                wandb.log(_filter_wandb({
                    "test_day": int(row["day"]),
                    "test/daily/ramping": row["ramping"],
                    "test/daily/peak": row["daily_peak"],
                    "test/daily/load_factor": row["load_factor"],
                    "test/daily/pvr": row["pvr"],
                    "test/daily/energy": row["energy"],
                    **build_readme_secondary_daily_log(row, "test/daily_secondary"),
                }))
        if not secondary_baseline_df.empty:
            wandb.define_metric("test/daily_baseline/*", step_metric="test_day")
            wandb.define_metric("test/daily_secondary_baseline/*", step_metric="test_day")
            for _, row in secondary_baseline_df.iterrows():
                wandb.log(_filter_wandb({
                    "test_day": int(row["day"]),
                    "test/daily_baseline/ramping": row["ramping"],
                    "test/daily_baseline/peak": row["daily_peak"],
                    "test/daily_baseline/load_factor": row["load_factor"],
                    "test/daily_baseline/pvr": row["pvr"],
                    "test/daily_baseline/energy": row["energy"],
                    **build_readme_secondary_daily_log(row, "test/daily_secondary_baseline"),
                }))
        if not daily_df.empty:
            wandb.define_metric("test/daily_primary/*", step_metric="test_day")
            for _, row in daily_df.iterrows():
                wandb.log(_filter_wandb({
                    "test_day": int(row["day"]),
                    "test/daily_primary/nmbe_pct": row["nmbe_pct"],
                    "test/daily_primary/cv_rmse_pct": row["cv_rmse_pct"],
                    "test/daily_primary/temperature_exceedance_pct_portfolio": row["temperature_exceedance_pct_portfolio"],
                    "test/daily_primary/temperature_exceedance_hours_total": row["temperature_exceedance_hours_total"],
                }))

    return daily_df


# ---------------------------------------------------------------------------
# Test evaluation (no gradient updates)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    algorithm:    SAC,
    cfg:          Config,
    test_start:   int,
    test_end:     int,
    agent_ids:    List[str],
    action_spaces,
    use_wandb:    bool,
    iteration:    Optional[int] = None,
) -> Dict:
    """Run one deterministic episode on the test window using trained policies.

    Manually steps through the env using algorithm.compute_single_action
    so we stay deterministic (no exploration noise).
    """
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    print(f"\n{'='*65}")
    print(f"TEST EVALUATION | {cfg.climate} | {month_name} (steps {test_start}-{test_end})")
    print(f"{'='*65}")

    test_env_obj = CityLearnMultiAgentEnv({
        "climate":     cfg.climate,
        "n_buildings": cfg.n_buildings,
        "start_step":  test_start,
        "end_step":    test_end,
        "seed":        cfg.seed,
    })

    obs_dict, _ = test_env_obj.reset(seed=cfg.seed)
    per_building_rewards: Dict[str, float] = {aid: 0.0 for aid in agent_ids}
    step_portfolio_rewards: List[float] = []
    step_portfolio_loads: List[float] = []

    terminated = False
    while not terminated:
        action_dict: Dict[str, np.ndarray] = {}
        for aid in agent_ids:
            obs = obs_dict.get(aid, np.zeros_like(list(obs_dict.values())[0]))
            # Deterministic policy evaluation
            action = algorithm.compute_single_action(
                obs,
                policy_id=aid,
                explore=False,
            )
            action_dict[aid] = np.asarray(action, dtype=np.float32)

        next_obs, rewards, terminateds, truncateds, _ = test_env_obj.step(action_dict)
        terminated = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        for aid in agent_ids:
            per_building_rewards[aid] += float(rewards.get(aid, 0.0))
        step_portfolio_rewards.append(sum(rewards.get(aid, 0.0) for aid in agent_ids))

        try:
            net_load = sum(
                float(b.net_electricity_consumption[-1])
                for b in test_env_obj.base_env.buildings
            )
            step_portfolio_loads.append(net_load)
        except Exception:
            step_portfolio_loads.append(float("nan"))
        obs_dict = next_obs

    test_kpis = extract_episode_kpis(test_env_obj.base_env)
    test_soc  = get_soc_stats(test_env_obj.base_env)
    primary_metrics, daily_primary_df, comfort_building_df = compute_primary_metric_tables(
        test_env_obj.base_env,
        cfg.test_month,
    )
    portfolio_reward = sum(per_building_rewards.values())

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "nan"
        try:
            return f"{float(v):.3f}" if np.isfinite(float(v)) else "nan"
        except Exception:
            return "nan"

    print(
        f"  reward_sum {portfolio_reward:9.2f} | "
        f"NMBE {_fmt(primary_metrics.get('primary/load_tracking/nmbe_pct'))}% | "
        f"CV-RMSE {_fmt(primary_metrics.get('primary/load_tracking/cv_rmse_pct'))}% | "
        f"comfort {_fmt(primary_metrics.get('primary/thermal_comfort/portfolio_exceedance_pct'))}%"
    )

    result: Dict = {
        "test/portfolio/reward_sum":  portfolio_reward,
        "test/portfolio/reward_mean": float(np.mean(list(per_building_rewards.values()))),
        "test/step_reward_mean":      float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0,
        **{f"test/{k}": v for k, v in primary_metrics.items()},
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
        "_step_portfolio_loads": step_portfolio_loads,
        "_step_portfolio_loads_baseline": resolve_reference_baseline_series(test_env_obj.base_env)[: len(step_portfolio_loads)].tolist(),
        "_daily_primary_metrics": daily_primary_df.to_dict(orient="records"),
        "_building_comfort_metrics": comfort_building_df.to_dict(orient="records"),
    }
    for aid, r in per_building_rewards.items():
        result[f"test/{aid}/reward"] = r
    if iteration is not None:
        result["episode"] = iteration

    if use_wandb:
        wandb.log(_filter_wandb(result), step=iteration)

    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_run_config(cfg: Config, save_dir: Path) -> None:
    p = save_dir / "run_config.json"
    p.write_text(json.dumps(vars(cfg), indent=2))
    print(f"  [cfg] run_config.json → {p}")


def export_test_metrics(
    test_result: Dict,
    cfg:         Config,
    save_dir:    Path,
    test_start:  int,
    test_end:    int,
    test_month:  int,
) -> None:
    public_result = {k: v for k, v in test_result.items() if not k.startswith("_")}
    payload = {
        "climate":         cfg.climate,
        "seed":            cfg.seed,
        "test_month":      test_month,
        "test_month_name": _MONTH_NAMES.get(test_month, str(test_month)),
        "test_start_step": test_start,
        "test_end_step":   test_end,
        **public_result,
    }
    json_path = save_dir / "test_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  [test] metrics JSON → {json_path}")

    flat = {k: v for k, v in payload.items() if v is not None and not isinstance(v, (dict, list))}
    csv_path = save_dir / "test_metrics.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"  [test] metrics CSV  → {csv_path}")


def _make_backup_dir(cfg: Config) -> Path:
    if cfg.backup_dir:
        return Path(cfg.backup_dir).resolve()
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(cfg.save_dir).resolve() / "backups" / f"{ts}_{cfg.climate}_seed{cfg.seed}"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> SAC:
    """Full RLlib SAC training loop followed by optional test evaluation."""

    # ── Resolve time windows ──────────────────────────────────────────────
    defaults    = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    train_month = cfg.train_month or defaults.get("train_month")
    test_month  = cfg.test_month  or defaults.get("test_month")

    if train_month is None or train_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid train_month={train_month}. Must be 1-12.")

    train_start = _MONTH_STARTS[train_month]
    train_end   = _MONTH_ENDS[train_month]
    test_start  = _MONTH_STARTS[test_month] if test_month else None
    test_end    = _MONTH_ENDS[test_month]   if test_month else None

    cfg.train_month = train_month
    cfg.test_month  = test_month

    # ── Seed & device ─────────────────────────────────────────────────────
    seed_everything(cfg.seed)
    print_repro_metadata(cfg, train_start, train_end, test_start, test_end)

    # ── Output directory ──────────────────────────────────────────────────
    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, save_dir)

    # ── W&B init ──────────────────────────────────────────────────────────
    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = vars(cfg).copy()
        cfg_dict.update({
            "train_start_step": train_start,
            "train_end_step":   train_end,
            "test_start_step":  test_start,
            "test_end_step":    test_end,
        })
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg_dict)
        wandb.define_metric("episode")
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("test/*",  step_metric="episode")

    # ── Probe environment to get spaces ──────────────────────────────────
    print(f"Probing environment: {cfg.climate} | "
          f"{_MONTH_NAMES.get(train_month, str(train_month))} "
          f"(steps {train_start}–{train_end})")

    probe_env_config: Dict = {
        "climate":     cfg.climate,
        "n_buildings": cfg.n_buildings,
        "start_step":  train_start,
        "end_step":    train_end,
        "seed":        cfg.seed,
    }
    probe_env = CityLearnMultiAgentEnv(probe_env_config)
    probe_env.reset()

    # Sorted agent IDs for deterministic ordering
    agent_ids   = sorted(probe_env.get_agent_ids(),
                         key=lambda s: int(s.split("_")[1]))
    sample_aid  = agent_ids[0]
    obs_space   = probe_env.observation_space[sample_aid]   # already [-5, 5]
    act_space   = probe_env.action_space[sample_aid]
    action_spaces = {aid: probe_env.action_space[aid] for aid in agent_ids}

    print(f"  agents: {len(agent_ids)} | obs_dim: {obs_space.shape[0]} | "
          f"act_dim: {act_space.shape[0]}")
    probe_env.close()

    # ── Build RLlib config ────────────────────────────────────────────────
    sac_config = build_sac_config(
        cfg         = cfg,
        env_config  = probe_env_config,
        agent_ids   = agent_ids,
        obs_space   = obs_space,
        act_space   = act_space,
    )

    # ── Ray init ──────────────────────────────────────────────────────────
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    algorithm: SAC = sac_config.build()

    # ── Training loop ─────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{ALGORITHM_LABEL} | Climate: {cfg.climate} | agents: {len(agent_ids)} | "
          f"iterations: {cfg.n_iterations}")
    print("=" * 65)

    all_rewards:         List[float] = []
    all_primary_metrics: List[Dict] = []
    ckpt_dir_path: Optional[str] = None
    progress = ProgressTimer(cfg.n_iterations, unit="Iter", label="train")

    for iteration in range(1, cfg.n_iterations + 1):
        result = algorithm.train()

        ep_reward_mean = result.get("episode_reward_mean", float("nan"))
        all_rewards.append(ep_reward_mean)

        cm   = result.get("custom_metrics", {})
        kpis = _collect_rllib_custom_metrics(cm, "kpi")
        primary_metrics = _collect_rllib_custom_metrics(cm, "primary")
        secondary_metrics = _collect_rllib_custom_metrics(cm, "secondary")
        all_primary_metrics.append(primary_metrics)

        if iteration % 10 == 0:
            portfolio_mean = cm.get("portfolio_reward_sum_mean", float("nan"))
            nmbe_val       = primary_metrics.get("primary/load_tracking/nmbe_pct", float("nan"))
            cv_rmse_val    = primary_metrics.get("primary/load_tracking/cv_rmse_pct", float("nan"))
            nmbe_str       = f"{nmbe_val:.3f}" if isinstance(nmbe_val, float) and np.isfinite(nmbe_val) else "nan"
            cv_rmse_str    = f"{cv_rmse_val:.3f}" if isinstance(cv_rmse_val, float) and np.isfinite(cv_rmse_val) else "nan"
            print(
                f"[train] Iter {iteration:4d} | ep_rew_mean {ep_reward_mean:9.2f} | "
                f"portfolio_sum {portfolio_mean:9.2f} | "
                f"NMBE {nmbe_str}% | CV-RMSE {cv_rmse_str}%"
            )

        if use_wandb:
            train_log: Dict[str, Any] = {
                "episode": iteration,
                "train/portfolio/reward_sum": cm.get("portfolio_reward_sum_mean"),
                "train/portfolio/reward_max": cm.get("portfolio_reward_sum_max"),
                "train/episode_reward_mean": result.get("episode_reward_mean"),
                "train/episode_len_mean": result.get("episode_len_mean"),
                "train/episodes_this_iter": result.get("episodes_this_iter"),
                **{f"train/{k}": v for k, v in primary_metrics.items()},
                **{f"train/{k}": v for k, v in secondary_metrics.items()},
                **{f"train/{k}": v for k, v in kpis.items()},
            }
            for key, value in _extract_rllib_sac_loss_metrics(result).items():
                train_log[f"train/loss/{key}"] = value
            for key, value in cm.items():
                if key.endswith("_reward_mean") and key.startswith("building_"):
                    train_log[f"train/{key.replace('_reward_mean', '')}/reward"] = value
            wandb.log(_filter_wandb(train_log), step=iteration)

        # Periodic checkpoint
        if iteration % max(cfg.n_iterations // 10, 1) == 0 or iteration == cfg.n_iterations:
            ckpt = algorithm.save(str(save_dir / "checkpoint"))
            ckpt_dir_path = str(ckpt)
            print(f"  [ckpt] iter {iteration} → {ckpt_dir_path}")
            save_plots(all_rewards, all_primary_metrics, save_dir)

        progress.step(iteration)

    # ── Test evaluation ───────────────────────────────────────────────────
    test_result: Optional[Dict] = None
    if cfg.do_test and test_start is not None:
        test_result = evaluate_on_test(
            algorithm    = algorithm,
            cfg          = cfg,
            test_start   = test_start,
            test_end     = test_end,
            agent_ids    = agent_ids,
            action_spaces = action_spaces,
            use_wandb    = use_wandb,
            iteration    = cfg.n_iterations,
        )
        export_test_metrics(
            test_result, cfg, save_dir, test_start, test_end, test_month
        )
        _run_daily_pipeline(test_result, cfg, save_dir, use_wandb)

    # ── Final artifacts ───────────────────────────────────────────────────
    save_plots(all_rewards, all_primary_metrics, save_dir)

    final_metrics: Dict = {
        "algorithm": ALGORITHM_LABEL,
        "train": {
            "n_iterations":       cfg.n_iterations,
            "last_ep_reward_mean": all_rewards[-1] if all_rewards else None,
            "checkpoint":         ckpt_dir_path,
            "train_month":        train_month,
            "train_steps":        f"{train_start}-{train_end}",
            "last_primary_metrics": all_primary_metrics[-1] if all_primary_metrics else {},
        }
    }
    if test_result is not None:
        final_metrics["test"] = {k: v for k, v in test_result.items() if not k.startswith("_")}

    (save_dir / "latest_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    # Backup
    backup_dir = _make_backup_dir(cfg)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in list(save_dir.glob("*.json")) + list(save_dir.glob("*.png")) + list(save_dir.glob("*.csv")):
        shutil.copy2(f, backup_dir / f.name)
    print(f"  [backup] → {backup_dir}/")

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print(f"  Artifacts: {save_dir}/")
    print(f"  Checkpoint: {ckpt_dir_path}")
    return algorithm


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="RLlib SAC Baseline for CityLearn Annex96-CE1"
    )
    parser.add_argument("--climate",      default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings",  type=int,   default=25)

    # SAC hyperparameters
    parser.add_argument("--lr",                    type=float, default=3e-4)
    parser.add_argument("--gamma",                 type=float, default=0.99)
    parser.add_argument("--tau",                   type=float, default=5e-3)
    parser.add_argument("--initial_alpha",         type=float, default=0.2)
    parser.add_argument("--buffer_size",           type=int,   default=100_000)
    parser.add_argument("--train_batch_size",      type=int,   default=256)
    parser.add_argument("--learning_starts",       type=int,   default=1000,
                        dest="num_steps_sampled_before_learning_starts")

    # Iteration / rollout settings
    parser.add_argument("--n_iterations",           type=int,   default=100)
    parser.add_argument("--rollout_fragment_len",   type=int,   default=200)
    parser.add_argument("--num_workers",            type=int,   default=0)

    # Logging
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name",    default="rllib-independent-sac")
    parser.add_argument("--save_dir",      default="results/rllib_sac")
    parser.add_argument("--backup_dir",    default=None)

    # Time windows
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month",  type=int, default=None)

    # Workflow
    parser.add_argument("--no_test", action="store_true")

    args = parser.parse_args()

    return Config(
        climate             = args.climate,
        n_buildings         = args.n_buildings,
        lr                  = args.lr,
        gamma               = args.gamma,
        tau                 = args.tau,
        initial_alpha       = args.initial_alpha,
        buffer_size         = args.buffer_size,
        train_batch_size    = args.train_batch_size,
        num_steps_sampled_before_learning_starts =
            args.num_steps_sampled_before_learning_starts,
        n_iterations        = args.n_iterations,
        rollout_fragment_len = args.rollout_fragment_len,
        num_workers         = args.num_workers,
        train_month         = args.train_month,
        test_month          = args.test_month,
        do_test             = not args.no_test,
        seed                = args.seed,
        wandb_project       = args.wandb_project,
        wandb_name          = args.wandb_name,
        save_dir            = args.save_dir,
        backup_dir          = args.backup_dir,
    )


if __name__ == "__main__":
    train(parse_args())
