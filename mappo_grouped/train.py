"""
Grouped-actor MAPPO — CityLearn Annex96-CE1
============================================

Extends mappo_standard with K actor networks (one per building group) and
a single shared centralised critic.  Algorithm internals (PPO, GAE, value
normalisation) are reused unchanged from on-policy-main.

Architecture
------------
  Env      : CityLearnMAPPOEnv (same as mappo_standard)
  Clusters : K-means on [BES_capacity, HVAC_total] → K groups (K ∈ {4, 5})
  Actors   : K separate R_Actor networks, one per group
             (each R_MAPPOPolicy[k] owns actors[k])
  Critic   : ONE shared R_Critic owned by policies[0]; policies[1..K-1]
             hold a reference to the same object and same optimiser.
  Value norm: one shared ValueNorm instance (shared across all K trainers).
  Buffers  : K SharedReplayBuffers (n_agents = |group_k| each); global
             share_obs (all 25 buildings concat) stored in each buffer.

Rollout
-------
  For every timestep, buildings in group k select actions via actors[k].
  The shared critic evaluates V(s_global) for all buildings.
  Per-group buffers receive only the data for their respective buildings.

Update (per episode)
--------------------
  for k in 0..K-1:
      trainers[k].train(buffers[k])   → updates actors[k] + shared critic
  Net effect: shared critic is updated K times sequentially (one pass per
  group), ensuring all group-specific return targets are incorporated.

W&B metrics
-----------
  train/portfolio/reward_sum
  train/loss/value_loss        (mean over K groups)
  train/loss/policy_loss       (mean over K groups)
  train/loss/dist_entropy      (mean over K groups)
  train/group_{k}/policy_loss  per-group actor losses
  train/kpi/*                  same KPIs as mappo_standard
  test/*                       identical schema to mappo_standard

Output artifacts (same names as mappo_standard)
--------------
  run_config.json   latest_metrics.json   checkpoint.pt
  training_curves.png  test_metrics.json  test_metrics.csv
  test_daily_metrics.png  test_daily_metrics.csv
  building_cluster_assignment.csv  cluster_centers.csv  cluster_summary.json

Run
---
  python -m mappo_grouped.train --climate VT
  python -m mappo_grouped.train --climate TX --group_k_candidates 4 5
  python -m mappo_grouped.train --climate VT --seed 0 --n_episodes 100
  python -m mappo_grouped.train --no_test
  python -m mappo_grouped.train --test_only --checkpoint_dir results/mappo_grouped_vt

Smoke test
----------
  python -m mappo_grouped.train --climate VT --n_episodes 2 --no_test
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from annex96_reporting import (
    build_readme_secondary_daily_log,
    collect_building_temperature_timeseries,
    compute_secondary_daily_tables,
    export_building_temperature_artifacts,
    export_secondary_daily_metrics,
    save_secondary_daily_metrics_plot,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent.parent
ONPOLICY  = REPO_DIR / "on-policy-main"

for _p in [str(REPO_DIR), str(ONPOLICY)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# on-policy-main imports  (direct reuse — no reimplementation)
# ---------------------------------------------------------------------------
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO                              # noqa: E402
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy         # noqa: E402
from onpolicy.utils.shared_buffer import SharedReplayBuffer                           # noqa: E402

# ---------------------------------------------------------------------------
# CityLearn adapter and KPI utilities
# ---------------------------------------------------------------------------
from mappo_grouped.env import CityLearnMAPPOEnv                                      # noqa: E402
from mappo_grouped.cluster import run_clustering                                      # noqa: E402
from mappo.utils import extract_episode_kpis, get_soc_stats, resolve_reference_baseline_series  # noqa: E402

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed — W&B logging disabled.")


# ---------------------------------------------------------------------------
# Month index mapping (identical to mappo_standard)
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Dataset
    climate:     str = "VT"
    n_buildings: int = 25

    # Clustering
    group_k_candidates: List[int] = field(default_factory=lambda: [4, 5])
    cluster_seed:       int = 0
    cluster_retries:    int = 10
    cluster_artifact_dir: Optional[str] = None   # None → same as save_dir

    # MAPPO network
    hidden_size: int   = 256
    layer_N:     int   = 2

    # MAPPO hyperparameters
    lr:              float = 3e-4
    critic_lr:       float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_param:      float = 0.2
    ppo_epoch:       int   = 10
    num_mini_batch:  int   = 4
    value_loss_coef: float = 1.0
    entropy_coef:    float = 0.01
    max_grad_norm:   float = 10.0

    # Training loop
    n_episodes: int = 100

    # Time windows (None → filled from _CLIMATE_DEFAULTS)
    train_month: Optional[int] = None
    test_month:  Optional[int] = None

    # Workflow
    do_test:        bool          = True
    test_only:      bool          = False
    checkpoint_dir: Optional[str] = None
    test_save_dir:  Optional[str] = None

    # Logging & output
    seed:          int   = 42
    wandb_project: str   = "annex96-ce1"
    wandb_name:    str   = "mappo-grouped"
    save_dir:      str   = "results/mappo_grouped"
    backup_dir:    Optional[str] = None


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


# ---------------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------------

def _filter_wandb(d: Dict) -> Dict:
    """Drop None, NaN, inf, tensor, and non-scalar values before W&B logging."""
    out: Dict = {}
    for k, v in d.items():
        if v is None or k.startswith("_"):
            continue
        if isinstance(v, (list, dict, tuple)):
            continue
        if isinstance(v, torch.Tensor):
            try:
                v = float(v.item())
            except Exception:
                continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(float(v)):
            continue
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Build on-policy-main args Namespace  (identical to mappo_standard)
# ---------------------------------------------------------------------------

def build_mappo_args(
    cfg:          Config,
    obs_dim:      int,
    act_dim:      int,
    cent_obs_dim: int,
) -> argparse.Namespace:
    """Construct the argparse.Namespace expected by R_MAPPOPolicy / R_MAPPO / SharedReplayBuffer."""
    return argparse.Namespace(
        # Network
        hidden_size               = cfg.hidden_size,
        layer_N                   = cfg.layer_N,
        use_ReLU                  = True,
        use_orthogonal            = True,
        gain                      = 0.01,
        use_feature_normalization = True,
        stacked_frames            = 1,
        # RNN (disabled)
        use_naive_recurrent_policy = False,
        use_recurrent_policy       = False,
        recurrent_N                = 1,
        data_chunk_length          = 10,
        # Optimisers
        lr           = cfg.lr,
        critic_lr    = cfg.critic_lr,
        opti_eps     = 1e-5,
        weight_decay = 0,
        # PPO
        clip_param             = cfg.clip_param,
        ppo_epoch              = cfg.ppo_epoch,
        num_mini_batch         = cfg.num_mini_batch,
        value_loss_coef        = cfg.value_loss_coef,
        entropy_coef           = cfg.entropy_coef,
        max_grad_norm          = cfg.max_grad_norm,
        huber_delta            = 10.0,
        use_max_grad_norm      = True,
        use_clipped_value_loss = True,
        use_huber_loss         = True,
        use_popart             = False,
        use_valuenorm          = True,
        use_value_active_masks = True,
        use_policy_active_masks= True,
        # GAE / Buffer
        gamma               = cfg.gamma,
        gae_lambda          = cfg.gae_lambda,
        use_gae             = True,
        use_proper_time_limits = False,
        # Misc
        algorithm_name    = "rmappo",
        use_centralized_V = True,
    )


# ---------------------------------------------------------------------------
# Build grouped MAPPO components
# ---------------------------------------------------------------------------

def _build_grouped_components(
    cfg:              Config,
    env:              CityLearnMAPPOEnv,
    group_assignments: np.ndarray,
    device:           torch.device,
) -> Tuple[argparse.Namespace, List[R_MAPPOPolicy], List[R_MAPPO],
           List[SharedReplayBuffer], List[np.ndarray]]:
    """Create K actor policies (shared critic), K trainers, and K replay buffers.

    Returns
    -------
    mappo_args    : shared Namespace used by all components
    policies      : list of K R_MAPPOPolicy; policies[1..K-1].critic == policies[0].critic
    trainers      : list of K R_MAPPO; trainers[1..K-1].value_normalizer == trainers[0].value_normalizer
    buffers       : list of K SharedReplayBuffer (n_agents = |group_k|)
    group_indices : list of K np.ndarray of building indices per group
    """
    K            = int(group_assignments.max()) + 1
    group_indices = [np.where(group_assignments == k)[0] for k in range(K)]
    group_sizes   = [int(len(idx)) for idx in group_indices]
    episode_length = env._base_env.episode_time_steps

    mappo_args = build_mappo_args(cfg, env.obs_dim, env.act_dim, env.cent_obs_dim)
    mappo_args.episode_length    = episode_length
    mappo_args.n_rollout_threads = 1

    # ── K actor policies ─────────────────────────────────────────────────────
    policies: List[R_MAPPOPolicy] = []
    for _ in range(K):
        policy = R_MAPPOPolicy(
            args          = mappo_args,
            obs_space     = env.obs_space,
            cent_obs_space= env.cent_obs_space,
            act_space     = env.act_space,
            device        = device,
        )
        policies.append(policy)

    # Share critic (and critic optimiser) — policies[1..K-1] reference policies[0]
    shared_critic     = policies[0].critic
    shared_critic_opt = policies[0].critic_optimizer
    for k in range(1, K):
        policies[k].critic           = shared_critic
        policies[k].critic_optimizer = shared_critic_opt

    # ── K trainers ────────────────────────────────────────────────────────────
    trainers: List[R_MAPPO] = [
        R_MAPPO(args=mappo_args, policy=policies[k], device=device)
        for k in range(K)
    ]

    # Share value normaliser across all trainers (consistent with shared critic)
    if trainers[0].value_normalizer is not None:
        shared_vn = trainers[0].value_normalizer
        for k in range(1, K):
            trainers[k].value_normalizer = shared_vn

    # ── K replay buffers ──────────────────────────────────────────────────────
    buffers: List[SharedReplayBuffer] = []
    for k in range(K):
        buf = SharedReplayBuffer(
            args          = mappo_args,
            num_agents    = group_sizes[k],
            obs_space     = env.obs_space,
            cent_obs_space= env.cent_obs_space,
            act_space     = env.act_space,
        )
        # Fix log_prob buffer shape: R_Actor returns (batch, 1) not (batch, act_dim)
        buf.action_log_probs = np.zeros(
            (episode_length, 1, group_sizes[k], 1),
            dtype=np.float32,
        )
        buffers.append(buf)

    return mappo_args, policies, trainers, buffers, group_indices


# ---------------------------------------------------------------------------
# Plotting (identical to mappo_standard)
# ---------------------------------------------------------------------------

def save_plots(
    rewards:  List[float],
    kpis:     List[Dict],
    save_dir: Path,
) -> None:
    if not rewards:
        return
    iters = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Grouped MAPPO (on-policy-main) — CityLearn CE1", fontsize=14)

    axes[0].plot(iters, rewards)
    axes[0].set_title("Episode Reward Sum (train)")
    axes[0].set_xlabel("Episode")

    for ax, key, title in [
        (axes[1], "kpi/ramping",    "Ramping (avg)"),
        (axes[2], "kpi/daily_peak", "Daily Peak (avg)"),
    ]:
        vals  = [k.get(key) for k in kpis]
        valid = [(e, v) for e, v in zip(iters, vals) if v is not None]
        if valid:
            ev, vv = zip(*valid)
            ax.plot(ev, vv, label="Grouped MAPPO")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="No-control (=1)")
        ax.set_title(f"KPI: {title}")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  [plot] saved → {out}")


# ---------------------------------------------------------------------------
# Daily metrics helpers (identical to mappo_standard)
# ---------------------------------------------------------------------------

def compute_daily_metrics(
    step_loads:    List[float],
    steps_per_day: int = 24,
) -> pd.DataFrame:
    arr    = np.array(step_loads, dtype=float)
    n_days = len(arr) // steps_per_day
    rows: List[Dict] = []
    for d in range(n_days):
        sl  = d * steps_per_day
        el  = sl + steps_per_day
        day = arr[sl:el]

        peak = float(day.max())
        low  = float(day.min())
        mean = float(day.mean())

        diffs       = np.abs(np.diff(day))
        ramping     = float(diffs.sum()) if len(diffs) > 0 else 0.0
        load_factor = mean / peak if peak > 0 else float("nan")
        pvr         = peak / low  if low  > 0 else float("nan")

        rows.append({
            "day":         d + 1,
            "ramping":     ramping,
            "daily_peak":  peak,
            "daily_min":   low,
            "load_factor": load_factor,
            "pvr":         pvr,
            "energy":      float(day.sum()),
        })
    return pd.DataFrame(rows)


def save_daily_metrics_plot(
    daily_df:   pd.DataFrame,
    save_dir:   Path,
    climate:    str,
    month_name: str,
) -> Path:
    days = daily_df["day"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        f"Grouped MAPPO — Daily Test Metrics | {climate} | {month_name}",
        fontsize=13,
    )

    def _plot(ax, col: str, title: str, ylabel: str, color: str) -> None:
        vals = daily_df[col].tolist()
        ax.plot(days, vals, marker="o", markersize=3, color=color, linewidth=1.2)
        ax.fill_between(days, vals, alpha=0.15, color=color)
        mean_v = float(np.nanmean(vals))
        ax.axhline(mean_v, color=color, linestyle="--", linewidth=0.8, alpha=0.7,
                   label=f"mean={mean_v:.2f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Day of test month")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "ramping",     "Daily Ramping (∑|Δload|)",  "kW", "#e15759")
    _plot(axes[0, 1], "daily_peak",  "Daily Peak Load",            "kW", "#f28e2b")
    _plot(axes[0, 2], "load_factor", "Load Factor (mean/peak)",    "—",  "#4e79a7")
    _plot(axes[1, 0], "pvr",         "Peak-to-Valley Ratio",       "—",  "#76b7b2")
    _plot(axes[1, 1], "energy",      "Daily Energy Consumption",   "kWh","#59a14f")

    ax_s = axes[1, 2]
    ax_s.axis("off")
    summary_cols = ["ramping", "daily_peak", "load_factor", "pvr", "energy"]
    col_labels   = ["Metric", "Mean", "Std", "Min", "Max"]
    cell_text    = []
    for col in summary_cols:
        v = daily_df[col].dropna()
        cell_text.append([col, f"{v.mean():.3f}", f"{v.std():.3f}",
                          f"{v.min():.3f}", f"{v.max():.3f}"])
    tbl = ax_s.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.4)
    ax_s.set_title("Summary Statistics", fontsize=10)

    plt.tight_layout()
    out = Path(save_dir) / "test_daily_metrics.png"
    plt.savefig(str(out), dpi=120)
    plt.close()
    print(f"  [daily] figure → {out}")
    return out


def _run_daily_pipeline(
    test_result: Dict,
    cfg:         Config,
    save_dir:    Path,
    use_wandb:   bool,
) -> Optional[pd.DataFrame]:
    step_loads: List[float] = test_result.get("_step_portfolio_loads", [])
    if not step_loads:
        print("[warn] No step-level load data — skipping daily metrics.")
        return None

    daily_df   = compute_daily_metrics(step_loads, steps_per_day=24)
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    save_daily_metrics_plot(daily_df, save_dir, cfg.climate, month_name)

    csv_out = Path(save_dir) / "test_daily_metrics.csv"
    daily_df.to_csv(csv_out, index=False)
    print(f"  [daily] CSV → {csv_out}")

    if use_wandb:
        wandb.define_metric("test_day")
        wandb.define_metric("test/daily/*", step_metric="test_day")
        wandb.define_metric("test/daily_secondary/*", step_metric="test_day")
        for _, row in daily_df.iterrows():
            wandb.log({
                "test_day":               int(row["day"]),
                "test/daily/ramping":     row["ramping"],
                "test/daily/peak":        row["daily_peak"],
                "test/daily/load_factor": row["load_factor"],
                "test/daily/pvr":         row["pvr"],
                "test/daily/energy":      row["energy"],
                **build_readme_secondary_daily_log(row, "test/daily_secondary"),
            })

    return daily_df


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

    season, comfort_low_c, comfort_high_c = _get_season_comfort_bounds(month)
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

    episodes = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Grouped MAPPO - Primary Metrics", fontsize=14)

    axes[0, 0].plot(episodes, rewards, color="#4e79a7")
    axes[0, 0].set_title("Episode Reward Sum")
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
    print(f"  [plot] saved 鈫?{out}")


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
        f"Grouped MAPPO - Primary Metrics | {climate} | {month_name}",
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
    print(f"  [daily] figure 鈫?{out}")
    return out


def _run_daily_pipeline(
    test_result: Dict,
    cfg: Config,
    save_dir: Path,
    use_wandb: bool,
) -> Optional[pd.DataFrame]:
    daily_records = test_result.get("_daily_primary_metrics")
    comfort_records = test_result.get("_building_comfort_metrics")
    temperature_records = test_result.get("_building_temperature_timeseries")
    step_loads: List[float] = test_result.get("_step_portfolio_loads", [])
    baseline_loads: List[float] = test_result.get("_step_portfolio_loads_baseline", [])
    if not daily_records and not step_loads:
        print("[warn] No daily records - skipping daily artifacts.")
        return None

    daily_df = pd.DataFrame(daily_records or [])
    comfort_df = pd.DataFrame(comfort_records or [])
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    save_daily_metrics_plot(daily_df, comfort_df, save_dir, cfg.climate, month_name)

    daily_csv = Path(save_dir) / "test_daily_metrics.csv"
    daily_df.to_csv(daily_csv, index=False)
    print(f"  [daily] CSV 鈫?{daily_csv}")

    comfort_csv = Path(save_dir) / "test_building_comfort_metrics.csv"
    comfort_df.to_csv(comfort_csv, index=False)
    print(f"  [daily] building comfort CSV 鈫?{comfort_csv}")

    temperature_df = pd.DataFrame(temperature_records or [])
    temperature_csv, temperature_full_fig, temperature_week_fig = export_building_temperature_artifacts(
        temperature_df,
        save_dir,
        cfg.climate,
        month_name,
        algorithm_label="Grouped MAPPO Temperatures",
    )
    if temperature_csv is not None:
        print(f"  [daily] building temperature CSV -> {temperature_csv}")
    if temperature_full_fig is not None:
        print(f"  [daily] building temperature figure -> {temperature_full_fig}")
    if temperature_week_fig is not None:
        print(f"  [daily] building temperature week1 figure -> {temperature_week_fig}")

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
            "Grouped MAPPO",
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
        if not secondary_flexible_df.empty:
            wandb.define_metric("test_day")
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
            wandb.define_metric("test_day")
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
        wandb.define_metric("test_day")
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
# Test metrics export (identical schema to mappo_standard)
# ---------------------------------------------------------------------------

def export_test_metrics(
    test_result: Dict,
    cfg:         Config,
    save_dir:    Path,
    test_start:  int,
    test_end:    int,
    test_month:  int,
) -> None:
    public  = {k: v for k, v in test_result.items() if not k.startswith("_")}
    payload = {
        "climate":         cfg.climate,
        "seed":            cfg.seed,
        "test_month":      test_month,
        "test_month_name": _MONTH_NAMES.get(test_month, str(test_month)),
        "test_start_step": test_start,
        "test_end_step":   test_end,
        **public,
    }

    json_path = save_dir / "test_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  [test] metrics JSON → {json_path}")

    flat     = {k: v for k, v in payload.items()
                if v is not None and not isinstance(v, (dict, list))}
    csv_path = save_dir / "test_metrics.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"  [test] metrics CSV  → {csv_path}")


def save_run_config(cfg: Config, save_dir: Path) -> None:
    p = save_dir / "run_config.json"
    p.write_text(json.dumps(vars(cfg), indent=2))
    print(f"  [cfg] run_config.json → {p}")


# ---------------------------------------------------------------------------
# Checkpoint helpers — supports K actors + 1 shared critic
# ---------------------------------------------------------------------------

_CKPT_NAME = "checkpoint.pt"


def save_checkpoint(
    policies:         List[R_MAPPOPolicy],
    group_assignments: np.ndarray,
    mappo_args:       argparse.Namespace,
    save_dir:         Path,
    episode:          int,
    cfg:              Config,
) -> Path:
    K    = len(policies)
    ckpt = {
        "episode":                  episode,
        "K":                        K,
        "group_assignments":        group_assignments.tolist(),
        "policy_critic_state_dict": policies[0].critic.state_dict(),
        "critic_optimizer":         policies[0].critic_optimizer.state_dict(),
        "mappo_args":               vars(mappo_args),
        "cfg":                      vars(cfg),
    }
    for k in range(K):
        ckpt[f"policy_actor_{k}_state_dict"] = policies[k].actor.state_dict()
        ckpt[f"actor_{k}_optimizer"]         = policies[k].actor_optimizer.state_dict()

    path = save_dir / _CKPT_NAME
    torch.save(ckpt, path)
    print(f"  [ckpt] episode {episode} → {path}")
    return path


def load_checkpoint(
    policies:  List[R_MAPPOPolicy],
    ckpt_path: Path,
    device:    torch.device,
) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    K    = ckpt.get("K", len(policies))

    # Critic (shared — loaded into policies[0], already shared with others)
    policies[0].critic.load_state_dict(ckpt["policy_critic_state_dict"])
    if "critic_optimizer" in ckpt:
        policies[0].critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

    # Per-group actors
    for k in range(min(K, len(policies))):
        key = f"policy_actor_{k}_state_dict"
        if key in ckpt:
            policies[k].actor.load_state_dict(ckpt[key])
        opt_key = f"actor_{k}_optimizer"
        if opt_key in ckpt:
            policies[k].actor_optimizer.load_state_dict(ckpt[opt_key])

    episode = ckpt.get("episode", 0)
    print(f"  [ckpt] loaded {K}-group checkpoint from {ckpt_path}  (episode {episode})")
    return episode


# ---------------------------------------------------------------------------
# Test evaluation — grouped actors, deterministic, identical output to mappo_standard
# ---------------------------------------------------------------------------

def evaluate_on_test(
    policies:          List[R_MAPPOPolicy],
    group_indices:     List[np.ndarray],
    mappo_args:        argparse.Namespace,
    cfg:               Config,
    test_start:        int,
    test_end:          int,
    device:            torch.device,
    use_wandb:         bool,
    iteration:         Optional[int] = None,
    log_per_step:      bool = False,
) -> Dict:
    """Run one deterministic grouped episode on the test window.

    Each group uses its own actor (deterministic=True); critic not called.
    Output dict schema is identical to mappo_standard.evaluate_on_test.
    """
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    K          = len(policies)
    print(f"\n{'='*65}")
    print(f"TEST EVALUATION (grouped, K={K}) | {cfg.climate} | {month_name} "
          f"(steps {test_start}–{test_end})")
    print(f"{'='*65}")

    test_env = CityLearnMAPPOEnv(
        climate     = cfg.climate,
        n_buildings = cfg.n_buildings,
        start_step  = test_start,
        end_step    = test_end,
    )

    n_agents    = test_env.n_agents
    recurrent_N = mappo_args.recurrent_N
    hidden_size = mappo_args.hidden_size

    obs, _ = test_env.reset(seed=cfg.seed)

    # RNN states (zeros — feed-forward mode)
    rnn_a = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((n_agents, 1), dtype=np.float32)

    per_building_rewards:   Dict[str, float] = {f"building_{i}": 0.0 for i in range(n_agents)}
    step_portfolio_rewards: List[float]      = []
    step_portfolio_loads:   List[float]      = []
    cumulative_reward = 0.0
    test_step         = 0

    for k in range(K):
        policies[k].actor.eval()

    done = False
    while not done:
        all_actions = np.zeros((n_agents, test_env.act_dim), dtype=np.float32)
        rnn_a_new   = rnn_a.copy()

        with torch.no_grad():
            for k, idx_k in enumerate(group_indices):
                n_k     = len(idx_k)
                obs_k   = obs[idx_k]         # (n_k, obs_dim)
                rnn_a_k = rnn_a[idx_k]       # (n_k, recurrent_N, hidden_size)
                masks_k = masks[idx_k]       # (n_k, 1)

                acts_k, rnn_a_new_k = policies[k].act(
                    obs_k, rnn_a_k, masks_k, deterministic=True
                )
                all_actions[idx_k]  = acts_k.cpu().numpy()
                rnn_a_new[idx_k]    = rnn_a_new_k.cpu().numpy().reshape(
                    n_k, recurrent_N, hidden_size
                )

        all_actions = np.clip(all_actions, -1.0, 1.0)
        rnn_a       = rnn_a_new

        next_obs, _, rewards, done, _ = test_env.step(all_actions)

        step_rew = float(rewards.sum())
        for i in range(n_agents):
            per_building_rewards[f"building_{i}"] += float(rewards[i])
        step_portfolio_rewards.append(step_rew)
        cumulative_reward += step_rew

        try:
            net_load = sum(
                float(b.net_electricity_consumption[-1])
                for b in test_env.base_env.buildings
            )
            step_portfolio_loads.append(net_load)
        except Exception:
            step_portfolio_loads.append(float("nan"))

        if log_per_step and use_wandb:
            step_log: Dict = {
                "test_step":              test_step,
                "test/step_reward":       step_rew,
                "test/cumulative_reward": cumulative_reward,
            }
            try:
                soc_vals = [
                    float(b.electrical_storage.soc[-1])
                    for b in test_env.base_env.buildings
                ]
                step_log["test/soc/mean"] = float(np.mean(soc_vals))
                step_log["test/soc/min"]  = float(np.min(soc_vals))
            except Exception:
                pass
            wandb.log(_filter_wandb(step_log))

        obs = next_obs
        if done:
            masks = np.zeros((n_agents, 1), dtype=np.float32)
        test_step += 1

    test_kpis        = extract_episode_kpis(test_env.base_env)
    test_soc         = get_soc_stats(test_env.base_env)
    primary_metrics, daily_primary_df, comfort_building_df = compute_primary_metric_tables(
        test_env.base_env,
        cfg.test_month,
    )
    portfolio_reward = sum(per_building_rewards.values())

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "nan"
        try:
            f = float(v)
            return f"{f:.3f}" if np.isfinite(f) else "nan"
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
        "test/step_reward_mean": (
            float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0
        ),
        **{f"test/{k}": v for k, v in primary_metrics.items()},
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
        "_step_portfolio_loads": step_portfolio_loads,
        "_step_portfolio_loads_baseline": resolve_reference_baseline_series(test_env.base_env)[: len(step_portfolio_loads)].tolist(),
        "_daily_primary_metrics": daily_primary_df.to_dict(orient="records"),
        "_building_comfort_metrics": comfort_building_df.to_dict(orient="records"),
        "_building_temperature_timeseries": collect_building_temperature_timeseries(test_env.base_env).to_dict(orient="records"),
    }
    for bid, r in per_building_rewards.items():
        result[f"test/{bid}/reward"] = r
    if iteration is not None:
        result["episode"] = iteration

    if use_wandb and not log_per_step:
        wandb.log(_filter_wandb(result), step=iteration)
    elif use_wandb and log_per_step:
        summary = {k: v for k, v in result.items() if not k.startswith("test/building_")}
        summary["test_step"] = test_step - 1
        wandb.log(_filter_wandb(summary))

    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    """Full grouped MAPPO training loop followed by optional test evaluation."""

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Training window : {cfg.climate} | "
          f"{_MONTH_NAMES.get(train_month)} (steps {train_start}–{train_end})")
    if test_start is not None:
        print(f"Test window     : {cfg.climate} | "
              f"{_MONTH_NAMES.get(test_month)} (steps {test_start}–{test_end})")

    # ── Output directories ────────────────────────────────────────────────
    save_dir    = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = Path(cfg.cluster_artifact_dir).resolve() if cfg.cluster_artifact_dir \
                  else save_dir
    cluster_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, save_dir)

    # ── Building clustering ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"BUILDING CLUSTERING | {cfg.climate} | K ∈ {cfg.group_k_candidates}")
    print("=" * 65)

    group_assignments, cluster_result = run_clustering(
        climate      = cfg.climate,
        save_dir     = cluster_dir,
        n_buildings  = cfg.n_buildings,
        k_candidates = cfg.group_k_candidates,
        cluster_seed = cfg.cluster_seed,
        retries      = cfg.cluster_retries,
        repo_dir     = REPO_DIR,
    )
    K = int(group_assignments.max()) + 1
    print(f"\n[cluster] Selected: K={K} | {cluster_result['reason']}")

    # ── W&B init ──────────────────────────────────────────────────────────
    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = vars(cfg).copy()
        cfg_dict.update({
            "train_start_step":  train_start,
            "train_end_step":    train_end,
            "test_start_step":   test_start,
            "test_end_step":     test_end,
            "algorithm":         "mappo_grouped",
            "n_groups":          K,
            "group_sizes":       cluster_result["sizes"],
            "cluster_balance":   cluster_result["balance"],
        })
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg_dict)
        wandb.define_metric("episode")
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("test/*",  step_metric="episode")

    # ── Build training environment ────────────────────────────────────────
    train_env = CityLearnMAPPOEnv(
        climate     = cfg.climate,
        n_buildings = cfg.n_buildings,
        start_step  = train_start,
        end_step    = train_end,
    )

    # ── Build grouped MAPPO components ────────────────────────────────────
    mappo_args, policies, trainers, buffers, group_indices = _build_grouped_components(
        cfg, train_env, group_assignments, device
    )

    episode_length = train_env._base_env.episode_time_steps
    n_agents       = train_env.n_agents
    obs_dim        = train_env.obs_dim
    cent_obs_dim   = train_env.cent_obs_dim
    act_dim        = train_env.act_dim
    recurrent_N    = mappo_args.recurrent_N
    hidden_size    = mappo_args.hidden_size
    group_sizes    = [len(idx) for idx in group_indices]

    print(f"\nGrouped MAPPO components built:")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  cent_obs_dim={cent_obs_dim}")
    print(f"  n_agents={n_agents}  K={K}  episode_length={episode_length}")
    print(f"  group_sizes={group_sizes}")
    print(f"  hidden_size={cfg.hidden_size}  ppo_epoch={cfg.ppo_epoch}")
    print(f"  Shared critic: policies[0].critic  "
          f"(all {K} policies reference same object)")

    # ── Training loop ─────────────────────────────────────────────────────
    print("=" * 65)
    print(f"Grouped MAPPO | Climate: {cfg.climate} | K={K} | episodes: {cfg.n_episodes}")
    print("=" * 65)

    all_rewards: List[float] = []
    all_primary_metrics: List[Dict[str, Any]] = []
    save_every = max(cfg.n_episodes // 10, 1)

    for episode in range(1, cfg.n_episodes + 1):

        # ── Init buffers for this episode ─────────────────────────────────
        obs, share_obs = train_env.reset(seed=cfg.seed + episode)
        # (n_agents, obs_dim), (n_agents, cent_obs_dim)

        for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
            n_k = group_sizes[k]
            buf.step = 0
            buf.share_obs[0] = share_obs[idx_k][np.newaxis]     # (1, n_k, cent_obs_dim)
            buf.obs[0]       = obs[idx_k][np.newaxis]           # (1, n_k, obs_dim)
            buf.masks[0]     = np.ones((1, n_k, 1), dtype=np.float32)
            buf.rnn_states[0]        = np.zeros(
                (1, n_k, recurrent_N, hidden_size), dtype=np.float32)
            buf.rnn_states_critic[0] = np.zeros_like(buf.rnn_states[0])

        # Global tracking arrays (25 buildings)
        rnn_a  = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
        rnn_c  = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
        masks  = np.ones((n_agents, 1), dtype=np.float32)

        episode_reward = 0.0

        for trainer in trainers:
            trainer.prep_rollout()

        # ── Rollout ───────────────────────────────────────────────────────
        for step in range(episode_length):
            all_values   = np.zeros((n_agents, 1),      dtype=np.float32)
            all_actions  = np.zeros((n_agents, act_dim), dtype=np.float32)
            all_log_probs = np.zeros((n_agents, 1),     dtype=np.float32)
            rnn_a_new    = rnn_a.copy()
            rnn_c_new    = rnn_c.copy()

            with torch.no_grad():
                for k, idx_k in enumerate(group_indices):
                    n_k         = group_sizes[k]
                    obs_k       = obs[idx_k]         # (n_k, obs_dim)
                    share_obs_k = share_obs[idx_k]   # (n_k, cent_obs_dim)
                    rnn_a_k     = rnn_a[idx_k]
                    rnn_c_k     = rnn_c[idx_k]
                    masks_k     = masks[idx_k]

                    vals_k, acts_k, logp_k, rnn_a_new_k, rnn_c_new_k = \
                        policies[k].get_actions(
                            share_obs_k, obs_k, rnn_a_k, rnn_c_k, masks_k
                        )

                    all_values[idx_k]    = vals_k.cpu().numpy()
                    all_actions[idx_k]   = acts_k.cpu().numpy()
                    all_log_probs[idx_k] = logp_k.cpu().numpy()
                    rnn_a_new[idx_k] = rnn_a_new_k.cpu().numpy().reshape(
                        n_k, recurrent_N, hidden_size)
                    rnn_c_new[idx_k] = rnn_c_new_k.cpu().numpy().reshape(
                        n_k, recurrent_N, hidden_size)

            actions_clipped = np.clip(all_actions, -1.0, 1.0)
            next_obs, next_share_obs, rewards, done, _ = train_env.step(actions_clipped)
            episode_reward += float(rewards.sum())

            masks_np = (
                np.zeros((n_agents, 1), dtype=np.float32)
                if done
                else np.ones((n_agents, 1), dtype=np.float32)
            )

            # Insert into per-group buffers
            for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
                n_k = group_sizes[k]
                buf.insert(
                    share_obs         = next_share_obs[idx_k][np.newaxis],
                    obs               = next_obs[idx_k][np.newaxis],
                    rnn_states_actor  = rnn_a_new[idx_k].reshape(
                                            1, n_k, recurrent_N, hidden_size),
                    rnn_states_critic = rnn_c_new[idx_k].reshape(
                                            1, n_k, recurrent_N, hidden_size),
                    actions           = all_actions[idx_k].reshape(1, n_k, act_dim),
                    action_log_probs  = all_log_probs[idx_k].reshape(1, n_k, 1),
                    value_preds       = all_values[idx_k].reshape(1, n_k, 1),
                    rewards           = rewards[idx_k].reshape(1, n_k, 1),
                    masks             = masks_np[idx_k].reshape(1, n_k, 1),
                )

            obs       = next_obs
            share_obs = next_share_obs
            rnn_a     = rnn_a_new
            rnn_c     = rnn_c_new
            masks     = masks_np

            if done:
                break

        # ── Bootstrap values + compute returns for each group buffer ──────
        with torch.no_grad():
            for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
                n_k          = group_sizes[k]
                share_last   = buf.share_obs[-1].reshape(n_k, cent_obs_dim)
                masks_last   = buf.masks[-1].reshape(n_k, 1)
                rnn_c_last   = buf.rnn_states_critic[-1].reshape(
                                   n_k, recurrent_N, hidden_size)
                next_values  = policies[k].get_values(share_last, rnn_c_last, masks_last)
                buf.compute_returns(
                    next_values.cpu().numpy().reshape(1, n_k, 1),
                    value_normalizer=trainers[0].value_normalizer,   # shared
                )

        # ── PPO updates: one pass per group ──────────────────────────────
        group_train_infos: List[Dict] = []
        for k in range(K):
            trainers[k].prep_training()
            info_k = trainers[k].train(buffers[k])
            group_train_infos.append(info_k)

        for k in range(K):
            trainers[k].prep_rollout()
            buffers[k].after_update()

        # Aggregate losses across groups
        def _avg_info(key: str) -> float:
            vals = []
            for info in group_train_infos:
                v = info.get(key, 0.0)
                if isinstance(v, torch.Tensor):
                    v = float(v.item())
                vals.append(float(v))
            return float(np.mean(vals)) if vals else 0.0

        train_info = {
            "value_loss":        _avg_info("value_loss"),
            "policy_loss":       _avg_info("policy_loss"),
            "dist_entropy":      _avg_info("dist_entropy"),
            "actor_grad_norm":   _avg_info("actor_grad_norm"),
            "critic_grad_norm":  _avg_info("critic_grad_norm"),
            "ratio":             _avg_info("ratio"),
        }

        # ── Logging ───────────────────────────────────────────────────────
        all_rewards.append(episode_reward)
        kpis = extract_episode_kpis(train_env.base_env)
        primary_metrics, _daily_primary_df, _comfort_building_df = compute_primary_metric_tables(
            train_env.base_env,
            cfg.train_month,
        )
        all_primary_metrics.append(primary_metrics)

        if episode % 10 == 0 or episode == 1:
            nmbe = _safe_scalar(primary_metrics.get("primary/load_tracking/nmbe_pct"))
            cv_rmse = _safe_scalar(primary_metrics.get("primary/load_tracking/cv_rmse_pct"))
            nmbe_str = f"{nmbe:.3f}" if nmbe is not None else "nan"
            cv_rmse_str = f"{cv_rmse:.3f}" if cv_rmse is not None else "nan"
            print(
                f"[train] Ep {episode:4d} | ep_rew {episode_reward:9.2f} | "
                f"v_loss {train_info['value_loss']:.4f} | "
                f"p_loss {train_info['policy_loss']:.4f} | "
                f"entropy {train_info['dist_entropy']:.4f} | "
                f"NMBE {nmbe_str}% | CV-RMSE {cv_rmse_str}%"
            )

        if use_wandb:
            log: Dict[str, Any] = {"episode": episode}
            log["train/portfolio/reward_sum"]  = episode_reward
            log["train/loss/value_loss"]       = train_info["value_loss"]
            log["train/loss/policy_loss"]      = train_info["policy_loss"]
            log["train/loss/dist_entropy"]     = train_info["dist_entropy"]
            log["train/loss/actor_grad_norm"]  = train_info["actor_grad_norm"]
            log["train/loss/critic_grad_norm"] = train_info["critic_grad_norm"]
            log["train/loss/ratio"]            = train_info["ratio"]
            # Per-group actor losses
            for k, info_k in enumerate(group_train_infos):
                log[f"train/group_{k}/policy_loss"]  = float(info_k.get("policy_loss", 0.0))
                log[f"train/group_{k}/dist_entropy"] = float(info_k.get("dist_entropy", 0.0))
            for kk, v in primary_metrics.items():
                if v is not None:
                    log[f"train/{kk}"] = v
            for kk, v in kpis.items():
                if v is not None:
                    log[f"train/{kk}"] = v
            wandb.log(_filter_wandb(log), step=episode)

        # ── Periodic checkpoint & plot ─────────────────────────────────────
        if episode % save_every == 0 or episode == cfg.n_episodes:
            save_checkpoint(policies, group_assignments, mappo_args, save_dir, episode, cfg)
            save_plots(all_rewards, all_primary_metrics, save_dir)

    # ── Test evaluation ───────────────────────────────────────────────────
    test_result: Optional[Dict] = None
    if cfg.do_test and test_start is not None:
        test_result = evaluate_on_test(
            policies       = policies,
            group_indices  = group_indices,
            mappo_args     = mappo_args,
            cfg            = cfg,
            test_start     = test_start,
            test_end       = test_end,
            device         = device,
            use_wandb      = use_wandb,
            iteration      = cfg.n_episodes,
        )
        export_test_metrics(test_result, cfg, save_dir, test_start, test_end, test_month)
        _run_daily_pipeline(test_result, cfg, save_dir, use_wandb)

    # ── Final artifacts ───────────────────────────────────────────────────
    save_plots(all_rewards, all_primary_metrics, save_dir)

    final_metrics: Dict = {
        "algorithm":  "mappo_grouped",
        "n_groups":   K,
        "group_sizes": group_sizes,
        "train": {
            "n_episodes":         cfg.n_episodes,
            "last_ep_reward_sum": all_rewards[-1] if all_rewards else None,
            "train_month":        train_month,
            "train_steps":        f"{train_start}-{train_end}",
            **(all_primary_metrics[-1] if all_primary_metrics else {}),
        },
    }
    if test_result is not None:
        public_test = {k: v for k, v in test_result.items() if not k.startswith("_")}
        final_metrics["test"] = {
            "test_month":  test_month,
            "test_steps":  f"{test_start}-{test_end}",
            **public_test,
        }

    (save_dir / "latest_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(f"  [metrics] latest_metrics.json → {save_dir}/latest_metrics.json")

    # Backup JSON and PNG artifacts
    backup_dir = (
        Path(cfg.backup_dir).resolve()
        if cfg.backup_dir
        else save_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in list(save_dir.glob("*.json")) + list(save_dir.glob("*.png")) + \
             list(save_dir.glob("*.csv")):
        shutil.copy2(f, backup_dir / f.name)
    print(f"  [backup] → {backup_dir}/")

    if use_wandb:
        wandb.finish()

    print("\nGrouped MAPPO training complete.")
    print(f"  Artifacts: {save_dir}/")


# ---------------------------------------------------------------------------
# Test-only entry point
# ---------------------------------------------------------------------------

def run_test_only(cfg: Config) -> Dict:
    """Load a saved checkpoint and run deterministic grouped test evaluation."""

    defaults    = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    test_month  = cfg.test_month  or defaults.get("test_month")
    train_month = cfg.train_month or defaults.get("train_month")

    if test_month is None or test_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid test_month={test_month}. Must be 1-12.")

    cfg.test_month  = test_month
    cfg.train_month = train_month
    test_start  = _MONTH_STARTS[test_month]
    test_end    = _MONTH_ENDS[test_month]
    train_start = _MONTH_STARTS[train_month]
    train_end   = _MONTH_ENDS[train_month]
    month_name  = _MONTH_NAMES.get(test_month, str(test_month))

    # Resolve checkpoint path
    default_ckpt = Path(cfg.save_dir).resolve() / _CKPT_NAME
    ckpt_dir     = Path(cfg.checkpoint_dir).resolve() if cfg.checkpoint_dir else None
    ckpt_path    = (
        ckpt_dir / _CKPT_NAME if ckpt_dir and ckpt_dir.is_dir()
        else ckpt_dir          if ckpt_dir and ckpt_dir.is_file()
        else default_ckpt
    )
    test_save_dir = Path(cfg.test_save_dir or str(ckpt_path.parent)).resolve()
    test_save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("GROUPED MAPPO — TEST-ONLY MODE")
    print(f"  Climate:    {cfg.climate}")
    print(f"  Test:       {month_name} (steps {test_start}–{test_end})")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Output:     {test_save_dir}")
    print("=" * 65)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run training first or pass --checkpoint_dir <path>."
        )

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to recover K and group_assignments
    ckpt = torch.load(ckpt_path, map_location=device)
    K    = ckpt.get("K", 4)
    group_assignments = np.array(ckpt.get("group_assignments",
                                          list(range(cfg.n_buildings))))

    group_indices = [np.where(group_assignments == k)[0] for k in range(K)]

    # Build probe env for network dimensions
    probe_env = CityLearnMAPPOEnv(
        climate     = cfg.climate,
        n_buildings = cfg.n_buildings,
        start_step  = train_start,
        end_step    = train_end,
    )

    mappo_args = build_mappo_args(cfg, probe_env.obs_dim, probe_env.act_dim, probe_env.cent_obs_dim)
    mappo_args.episode_length    = probe_env._base_env.episode_time_steps
    mappo_args.n_rollout_threads = 1

    # Build K policies with shared critic
    policies: List[R_MAPPOPolicy] = []
    for _ in range(K):
        policy = R_MAPPOPolicy(
            args          = mappo_args,
            obs_space     = probe_env.obs_space,
            cent_obs_space= probe_env.cent_obs_space,
            act_space     = probe_env.act_space,
            device        = device,
        )
        policies.append(policy)

    for k in range(1, K):
        policies[k].critic           = policies[0].critic
        policies[k].critic_optimizer = policies[0].critic_optimizer

    load_checkpoint(policies, ckpt_path, device)

    # W&B init (test-only)
    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = vars(cfg).copy()
        cfg_dict.update({
            "mode":            "test_only",
            "K":               K,
            "test_start_step": test_start,
            "test_end_step":   test_end,
        })
        wandb.init(
            project = cfg.wandb_project,
            name    = f"{cfg.wandb_name}-test-only",
            config  = cfg_dict,
        )
        wandb.define_metric("test_step")
        wandb.define_metric("test/*", step_metric="test_step")

    test_result = evaluate_on_test(
        policies      = policies,
        group_indices = group_indices,
        mappo_args    = mappo_args,
        cfg           = cfg,
        test_start    = test_start,
        test_end      = test_end,
        device        = device,
        use_wandb     = use_wandb,
        iteration     = None,
        log_per_step  = True,
    )

    export_test_metrics(test_result, cfg, test_save_dir, test_start, test_end, test_month)
    _run_daily_pipeline(test_result, cfg, test_save_dir, use_wandb)

    if use_wandb:
        wandb.finish()

    print("\nGrouped test-only evaluation complete.")
    print(f"  Outputs: {test_save_dir}/")
    return test_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Grouped-actor MAPPO for CityLearn Annex96-CE1: "
                    "K actors (one per building cluster) + 1 shared centralized critic."
    )

    # Dataset
    parser.add_argument("--climate",     default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)

    # Clustering
    parser.add_argument(
        "--group_k_candidates", type=int, nargs="+", default=[4, 5],
        metavar="K",
        help="K values to try for K-means (default: 4 5)",
    )
    parser.add_argument("--cluster_seed",    type=int, default=0,
                        help="Base seed for K-means (retries use seed+i)")
    parser.add_argument("--cluster_retries", type=int, default=10,
                        help="Number of random seeds to try per K value")
    parser.add_argument("--cluster_artifact_dir", default=None,
                        help="Save cluster CSV/JSON here (default: same as save_dir)")

    # MAPPO network
    parser.add_argument("--hidden_size", type=int,   default=256)
    parser.add_argument("--layer_N",     type=int,   default=2)

    # MAPPO hyperparameters
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--critic_lr",       type=float, default=3e-4)
    parser.add_argument("--gamma",           type=float, default=0.99)
    parser.add_argument("--gae_lambda",      type=float, default=0.95)
    parser.add_argument("--clip_param",      type=float, default=0.2)
    parser.add_argument("--ppo_epoch",       type=int,   default=10)
    parser.add_argument("--num_mini_batch",  type=int,   default=4)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef",    type=float, default=0.01)
    parser.add_argument("--max_grad_norm",   type=float, default=10.0)

    # Training
    parser.add_argument("--n_episodes", type=int, default=100)

    # Time windows
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month",  type=int, default=None)

    # Logging
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name",    default="mappo-grouped")
    parser.add_argument("--save_dir",      default="results/mappo_grouped")
    parser.add_argument("--backup_dir",    default=None)

    # Workflow
    parser.add_argument("--no_test",   action="store_true",
                        help="Skip test evaluation after training")
    parser.add_argument("--test_only", action="store_true",
                        help="Load checkpoint and evaluate; skip training")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Directory or file containing the checkpoint to load")
    parser.add_argument("--test_save_dir",  default=None,
                        help="Write test outputs here (default: same as checkpoint dir)")

    args = parser.parse_args()

    return Config(
        climate              = args.climate,
        n_buildings          = args.n_buildings,
        group_k_candidates   = args.group_k_candidates,
        cluster_seed         = args.cluster_seed,
        cluster_retries      = args.cluster_retries,
        cluster_artifact_dir = args.cluster_artifact_dir,
        hidden_size          = args.hidden_size,
        layer_N              = args.layer_N,
        lr                   = args.lr,
        critic_lr            = args.critic_lr,
        gamma                = args.gamma,
        gae_lambda           = args.gae_lambda,
        clip_param           = args.clip_param,
        ppo_epoch            = args.ppo_epoch,
        num_mini_batch       = args.num_mini_batch,
        value_loss_coef      = args.value_loss_coef,
        entropy_coef         = args.entropy_coef,
        max_grad_norm        = args.max_grad_norm,
        n_episodes           = args.n_episodes,
        train_month          = args.train_month,
        test_month           = args.test_month,
        do_test              = not args.no_test,
        test_only            = args.test_only,
        checkpoint_dir       = args.checkpoint_dir,
        test_save_dir        = args.test_save_dir,
        seed                 = args.seed,
        wandb_project        = args.wandb_project,
        wandb_name           = args.wandb_name,
        save_dir             = args.save_dir,
        backup_dir           = args.backup_dir,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_args()

    if cfg.test_only:
        run_test_only(cfg)
    else:
        train(cfg)
