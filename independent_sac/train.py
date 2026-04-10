"""
Independent SAC Baseline — Training + Evaluation Script
========================================================

Each of the 25 CE1 buildings trains its own SAC policy independently.
No message passing, no shared latent communication, no coordination.
Portfolio metrics are computed by summing/aggregating all buildings' outputs.

Run:
    python -m independent_sac.train                      # TX: train Aug, test Sep
    python -m independent_sac.train --climate VT         # VT: train Jan, test Feb
    python -m independent_sac.train --climate TX --n_episodes 50 --seed 0
    python -m independent_sac.train --no_test            # skip test episode

Train-test time windows (same convention as mappo/train.py):
    TX: train August   (steps 5088–5831), test September (steps 5832–6551)
    VT: train January  (steps    0– 743), test February  (steps  744–1415)

W&B metrics:
    train/portfolio/reward_sum     — sum of all buildings' episode rewards
    train/portfolio/reward_mean    — mean per building
    train/building_i/reward        — per-building episode reward
    train/loss_critic, train/loss_actor, train/alpha, train/entropy
    train/kpi/*                    — district-level CityLearn KPIs
    test/portfolio/reward_sum, test/portfolio/reward_mean
    test/building_i/reward         — per-building test reward
    test/kpi/*                     — district-level test KPIs

Output artifacts (in save_dir/):
    building_{i}_actor.pt          — actor weights for building i
    building_{i}_critic.pt
    building_{i}_critic_target.pt
    building_{i}_log_alpha.pt
    run_config.json
    latest_metrics.json
    test_metrics.json / test_metrics.csv   — compatible with compare.py
    training_curves.png
    backups/{ts}_{climate}_seed{seed}/     — timestamped backup

Design notes vs CityLearn's built-in SAC (citylearn/agents/sac.py):
    - Pure PyTorch, no RLC/RBC inheritance chain.
    - Automatic entropy temperature tuning (SAC paper §4.2).
    - Twin Q-networks with polyak-averaged target networks.
    - Truly independent: each building's agent sees only its own obs/reward.
    - Same time-window, seeding, wandb, and export conventions as mappo/train.py.
"""

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

# Ensure the local CityLearn copy is used (not any pip-installed version)
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

from independent_sac.agent import SACAgent
# Reuse KPI extraction and SOC utilities from mappo to keep outputs comparable
from mappo.utils import extract_episode_kpis, get_soc_stats

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed — W&B logging disabled.")


# ---------------------------------------------------------------------------
# Month → hourly time-step index mapping (non-leap year, 8 760 h total)
# Identical to mappo/train.py to ensure comparable train/test windows.
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

# Default train/test months per climate (from CE1 README)
_CLIMATE_DEFAULTS: Dict[str, Dict[str, int]] = {
    "VT": {"train_month": 1, "test_month": 2},   # January / February
    "TX": {"train_month": 8, "test_month": 9},   # August  / September
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Dataset
    climate:     str = "TX"   # "TX" (cooling-dominated) or "VT" (heating-dominated)
    n_buildings: int = 25     # max 25 in CE1

    # SAC hyperparameters
    hidden_dim:       int   = 256      # hidden layer width for all networks
    lr:               float = 3e-4    # learning rate for actor, critics, alpha
    gamma:            float = 0.99    # discount factor
    tau:              float = 5e-3    # polyak soft-target update coefficient
    alpha_init:       float = 0.2     # initial entropy temperature
    buffer_capacity:  int   = 100_000 # replay buffer capacity per building
    batch_size:       int   = 256     # minibatch size for gradient updates
    updates_per_step: int   = 1       # SAC gradient steps per env step
    learning_starts:  int   = 1000   # env steps before first gradient update
    max_grad_norm:    float = 1.0     # gradient clipping norm (actor + critic)

    # Training
    n_episodes: int = 100

    # Time windows (None → filled from _CLIMATE_DEFAULTS)
    train_month:        Optional[int] = None
    test_month:         Optional[int] = None
    episode_time_steps: Optional[int] = None  # None → full training month

    # Workflow
    do_test: bool = True

    # Logging & output
    save_every:    int          = 10
    seed:          int          = 42
    wandb_project: str          = "annex96-ce1"
    wandb_name:    str          = "independent-sac"
    save_dir:      str          = "results/independent_sac"
    backup_dir:    Optional[str] = None


# ---------------------------------------------------------------------------
# Reproducibility  (mirrors mappo/train.py exactly)
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_repro_metadata(
    cfg:         Config,
    device:      torch.device,
    train_start: int,
    train_end:   int,
    test_start:  Optional[int],
    test_end:    Optional[int],
) -> Dict:
    """Print and return reproducibility metadata dict."""
    meta: Dict = {
        "seed":          cfg.seed,
        "device":        str(device),
        "torch_version": torch.__version__,
        "climate":       cfg.climate,
        "train_month":   cfg.train_month,
        "train_steps":   f"{train_start}-{train_end}",
        "test_month":    cfg.test_month,
        "test_steps":    f"{test_start}-{test_end}" if test_start is not None else "N/A",
    }
    if torch.cuda.is_available():
        meta["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        meta["cudnn_benchmark"]     = torch.backends.cudnn.benchmark

    print("\n" + "=" * 65)
    print("REPRODUCIBILITY METADATA")
    for k, v in meta.items():
        print(f"  {k:<24} {v}")
    print("=" * 65 + "\n")
    return meta


# ---------------------------------------------------------------------------
# Environment helpers  (mirrors mappo/train.py build_env)
# ---------------------------------------------------------------------------

def build_env(
    cfg:        Config,
    start_step: Optional[int] = None,
    end_step:   Optional[int] = None,
) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate and wrap CityLearnEnv.

    central_agent=False so each building has its own observation/action space.
    start_step/end_step override the schema time window (train vs test split).
    """
    dataset_name = f"annex96_ce1_{cfg.climate.lower()}_neighborhood"
    dataset_dir  = REPO_DIR / "data" / "datasets" / dataset_name
    schema_path  = dataset_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found: {schema_path}\n"
            f"Available datasets: "
            f"{[d.name for d in (REPO_DIR / 'data' / 'datasets').iterdir()]}"
        )

    env_kwargs: Dict = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=False,
        buildings=list(range(cfg.n_buildings)),
    )

    if start_step is not None and end_step is not None:
        window_len = end_step - start_step + 1
        env_kwargs["simulation_start_time_step"] = start_step
        env_kwargs["simulation_end_time_step"]   = end_step
        env_kwargs["episode_time_steps"] = (
            cfg.episode_time_steps if cfg.episode_time_steps is not None else window_len
        )
    elif cfg.episode_time_steps is not None:
        env_kwargs["episode_time_steps"] = cfg.episode_time_steps

    base_env = CityLearnEnv(**env_kwargs)
    env      = NormalizedObservationWrapper(base_env)
    return env, base_env


def scale_action(raw: np.ndarray, action_space: spaces.Box) -> List[float]:
    """Scale actor output ∈ (-1, 1) to the actual action-space bounds.

    Same scaling convention as mappo/train.py assemble_env_actions.
    CityLearn expects actions as List[float] per building.
    """
    low  = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    scaled = low + (raw + 1.0) * 0.5 * (high - low)
    return np.clip(scaled, low, high).tolist()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_checkpoint(
    agents:     List[SACAgent],
    save_dir:   Path,
    cfg:        Config,
    metrics:    Optional[Dict] = None,
    backup_dir: Optional[Path] = None,
) -> None:
    """Save all agents, config, and metrics; verify; optionally backup.

    Each building's files are prefixed with building_{i}_.
    """
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for i, agent in enumerate(agents):
        prefix = str(save_dir / f"building_{i}")
        agent.save(prefix)
        for suffix in ["_actor.pt", "_critic.pt", "_critic_target.pt", "_log_alpha.pt"]:
            saved.append(Path(prefix + suffix))

    cfg_path = save_dir / "run_config.json"
    cfg_path.write_text(json.dumps(vars(cfg), indent=2))
    saved.append(cfg_path)

    if metrics is not None:
        m_path = save_dir / "latest_metrics.json"
        m_path.write_text(json.dumps(metrics, indent=2))
        saved.append(m_path)

    # Verify all files are non-empty
    for p in saved:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint verification failed: {p} is missing or empty.")

    print(f"  [ckpt] {len(saved)} files → {save_dir}/")

    if backup_dir is not None:
        backup_dir = Path(backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in saved:
            shutil.copy2(p, backup_dir / p.name)
        print(f"  [backup] → {backup_dir}/")


def _make_backup_dir(cfg: Config) -> Path:
    if cfg.backup_dir is not None:
        return Path(cfg.backup_dir).resolve()
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_{cfg.climate}_seed{cfg.seed}"
    return Path(cfg.save_dir).resolve() / "backups" / tag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    rewards:  List[float],
    kpis:     List[Dict],
    save_dir: Path,
) -> None:
    """Save training-curve figure with reward and key KPIs."""
    eps = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Independent SAC — CityLearn CE1", fontsize=14)

    axes[0].plot(eps, rewards)
    axes[0].set_title("Portfolio Reward Sum (train)")
    axes[0].set_xlabel("Episode")

    for ax, key, title in [
        (axes[1], "kpi/ramping",    "Ramping (avg)"),
        (axes[2], "kpi/daily_peak", "Daily Peak (avg)"),
    ]:
        vals  = [k.get(key) for k in kpis]
        valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
        if valid:
            ev, vv = zip(*valid)
            ax.plot(ev, vv, label="SAC")
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
# Test-metrics export
# ---------------------------------------------------------------------------

def _export_test_metrics(
    test_result: Dict,
    cfg:         Config,
    save_dir:    Path,
    test_start:  int,
    test_end:    int,
    test_month:  int,
) -> None:
    """Write test metrics to JSON and CSV — same schema as mappo/train.py."""
    payload = {
        "climate":         cfg.climate,
        "seed":            cfg.seed,
        "test_month":      test_month,
        "test_month_name": _MONTH_NAMES.get(test_month, str(test_month)),
        "test_start_step": test_start,
        "test_end_step":   test_end,
        **test_result,
    }

    json_path = save_dir / "test_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  [test] metrics JSON → {json_path}")

    # Flatten for CSV (skip None and nested dict values)
    flat     = {k: v for k, v in payload.items() if v is not None and not isinstance(v, dict)}
    csv_path = save_dir / "test_metrics.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"  [test] metrics CSV  → {csv_path}")


def _filter_log_values(d: Dict) -> Dict:
    """Drop None / NaN / inf values before W&B logging."""
    clean: Dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            continue
        clean[k] = v
    return clean


# ---------------------------------------------------------------------------
# Test evaluation (no parameter updates)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    agents:        List[SACAgent],
    action_spaces: List[spaces.Box],
    cfg:           Config,
    test_start:    int,
    test_end:      int,
    use_wandb:     bool,
    episode:       Optional[int] = None,
) -> Dict:
    """Run one deterministic episode on the test window.

    All agents are set to eval mode; no gradient updates are performed.
    Returns a dict with portfolio and per-building test rewards + KPIs.
    """
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    print(f"\n{'='*65}")
    print(f"TEST EVALUATION | {cfg.climate} | {month_name} (steps {test_start}–{test_end})")
    print(f"{'='*65}")

    test_env, test_base_env = build_env(cfg, start_step=test_start, end_step=test_end)
    n_buildings = len(test_base_env.buildings)

    for agent in agents:
        agent.eval_mode()

    obs_list, _ = test_env.reset(seed=cfg.seed)
    per_building_rewards:   List[float] = [0.0] * n_buildings
    step_portfolio_rewards: List[float] = []

    while not test_base_env.terminated:
        env_actions = []
        for i in range(n_buildings):
            raw = agents[i].select_action(np.array(obs_list[i]), deterministic=True)
            env_actions.append(scale_action(raw, action_spaces[i]))

        next_obs_list, rewards, terminated, truncated, _ = test_env.step(env_actions)
        for i in range(n_buildings):
            per_building_rewards[i] += float(rewards[i])
        step_portfolio_rewards.append(float(np.sum(rewards)))
        obs_list = next_obs_list

    for agent in agents:
        agent.train_mode()

    test_kpis = extract_episode_kpis(test_base_env)
    test_soc  = get_soc_stats(test_base_env)
    portfolio_reward = sum(per_building_rewards)

    result: Dict = {
        "test/portfolio/reward_sum":  portfolio_reward,
        "test/portfolio/reward_mean": float(np.mean(per_building_rewards)),
        "test/step_reward_mean":      (
            float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0
        ),
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
    }
    for i, r in enumerate(per_building_rewards):
        result[f"test/building_{i}/reward"] = r

    if episode is not None:
        result["episode"] = episode

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "nan"
        try:
            f = float(v)
            return f"{f:.3f}" if np.isfinite(f) else "nan"
        except (TypeError, ValueError):
            return "nan"

    print(
        f"  reward_sum {portfolio_reward:9.2f} | "
        f"ramp {_fmt(test_kpis.get('kpi/ramping'))} | "
        f"peak {_fmt(test_kpis.get('kpi/daily_peak'))} | "
        f"cost {_fmt(test_kpis.get('kpi/cost'))}"
    )

    if use_wandb:
        wandb.log(_filter_log_values(result), step=episode)

    return result


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Config) -> List[SACAgent]:
    """Full Independent SAC training loop followed by optional test evaluation.

    Returns the list of trained SACAgent objects (one per building).
    """
    # ── Resolve time windows ──────────────────────────────────────────────
    defaults    = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    train_month = cfg.train_month or defaults.get("train_month")
    test_month  = cfg.test_month  or defaults.get("test_month")

    if train_month is None or train_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid train_month={train_month}. Must be 1-12.")
    if test_month is not None and test_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid test_month={test_month}. Must be 1-12.")

    train_start = _MONTH_STARTS[train_month]
    train_end   = _MONTH_ENDS[train_month]
    test_start  = _MONTH_STARTS[test_month] if test_month is not None else None
    test_end    = _MONTH_ENDS[test_month]   if test_month is not None else None

    # Stash resolved months so they appear in run_config.json
    cfg.train_month = train_month
    cfg.test_month  = test_month

    # ── Seeding & device ─────────────────────────────────────────────────
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory ─────────────────────────────────────────────────
    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility metadata ─────────────────────────────────────────
    repro_meta = print_repro_metadata(
        cfg, device, train_start, train_end, test_start, test_end
    )

    # ── W&B setup ────────────────────────────────────────────────────────
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

    # ── Training environment ─────────────────────────────────────────────
    train_month_name = _MONTH_NAMES.get(train_month, str(train_month))
    print(
        f"Training window: {cfg.climate} | {train_month_name} "
        f"(steps {train_start}–{train_end})"
    )

    env, base_env = build_env(cfg, start_step=train_start, end_step=train_end)
    n_buildings   = len(base_env.buildings)
    obs_dims      = [env.observation_space[i].shape[0] for i in range(n_buildings)]
    act_dims      = [env.action_space[i].shape[0]      for i in range(n_buildings)]
    action_spaces = [base_env.action_space[i]           for i in range(n_buildings)]

    print(
        f"Environment: {cfg.climate} | buildings: {n_buildings} | "
        f"obs/building: {obs_dims[0]} | act/building: {act_dims[0]}"
    )

    # ── One SAC agent per building (independent, no communication) ────────
    agents: List[SACAgent] = [
        SACAgent(
            obs_dim         = obs_dims[i],
            action_dim      = act_dims[i],
            hidden_dim      = cfg.hidden_dim,
            lr              = cfg.lr,
            gamma           = cfg.gamma,
            tau             = cfg.tau,
            alpha_init      = cfg.alpha_init,
            buffer_capacity = cfg.buffer_capacity,
            max_grad_norm   = cfg.max_grad_norm,
            device          = device,
        )
        for i in range(n_buildings)
    ]
    print(f"Created {n_buildings} independent SAC agents (one per building).")

    # ── Optional RBC baseline for KPI comparison in plots ────────────────
    rbc_kpis: Optional[pd.DataFrame] = None
    rbc_kpi_file = REPO_DIR / "notebooks" / "rbc_baseline_kpi_summary.csv"
    if rbc_kpi_file.exists():
        try:
            rbc_kpis = pd.read_csv(rbc_kpi_file)
            print(f"Loaded RBC baseline from {rbc_kpi_file}")
        except Exception as exc:
            print(f"[warn] Could not load RBC baseline: {exc}")
    else:
        print("[info] RBC baseline file not found — skipping comparison lines in plots.")

    # ── Metric history ────────────────────────────────────────────────────
    all_rewards: List[float] = []
    all_kpis:    List[Dict]  = []
    total_steps: int         = 0

    print("=" * 65)
    print(
        f"Independent SAC | Climate: {cfg.climate} | "
        f"buildings: {n_buildings} | episodes: {cfg.n_episodes}"
    )
    print("=" * 65)

    for episode in range(1, cfg.n_episodes + 1):

        # ================================================================
        # 1. ROLLOUT — collect one full training episode
        # ================================================================
        # Vary seed per episode for diverse initial states while remaining
        # fully reproducible (same convention as mappo/train.py).
        ep_seed = cfg.seed if episode == 1 else cfg.seed + episode
        obs_list, _ = env.reset(seed=ep_seed)

        per_building_rewards:   List[float]        = [0.0] * n_buildings
        step_portfolio_rewards: List[float]         = []
        ep_losses:              Dict[str, List[float]] = {
            "critic_loss": [], "actor_loss": [], "alpha": [], "entropy": [],
        }

        while not base_env.terminated:
            # Each building selects its action independently
            actions_raw: List[np.ndarray] = []
            env_actions: List[List[float]] = []
            for i in range(n_buildings):
                raw = agents[i].select_action(
                    np.array(obs_list[i], dtype=np.float32), deterministic=False
                )
                actions_raw.append(raw)
                env_actions.append(scale_action(raw, action_spaces[i]))

            next_obs_list, rewards, terminated, truncated, _ = env.step(env_actions)
            done = terminated or truncated

            # Push each building's transition to its own replay buffer
            for i in range(n_buildings):
                agents[i].push(
                    obs      = np.array(obs_list[i],      dtype=np.float32),
                    action   = actions_raw[i],
                    reward   = float(rewards[i]),
                    next_obs = np.array(next_obs_list[i], dtype=np.float32),
                    done     = done,
                )
                per_building_rewards[i] += float(rewards[i])

            step_portfolio_rewards.append(float(np.sum(rewards)))
            total_steps += 1
            obs_list = next_obs_list

            # ============================================================
            # 2. SAC GRADIENT UPDATES (off-policy, per step)
            # ============================================================
            if total_steps >= cfg.learning_starts:
                for _ in range(cfg.updates_per_step):
                    for i in range(n_buildings):
                        if len(agents[i].replay_buffer) >= cfg.batch_size:
                            info = agents[i].update(cfg.batch_size)
                            for k in ep_losses:
                                if k in info:
                                    ep_losses[k].append(info[k])

        # ================================================================
        # 3. EPISODE METRICS & LOGGING
        # ================================================================
        portfolio_reward         = sum(per_building_rewards)
        mean_reward_per_building = float(np.mean(per_building_rewards))
        all_rewards.append(portfolio_reward)

        kpis      = extract_episode_kpis(base_env)
        soc_stats = get_soc_stats(base_env)
        all_kpis.append(kpis)

        def _mean_or_nan(lst: List[float]) -> float:
            return float(np.mean(lst)) if lst else float("nan")

        log_dict: Dict = {
            "episode":                     episode,
            "train/portfolio/reward_sum":  portfolio_reward,
            "train/portfolio/reward_mean": mean_reward_per_building,
            "train/step_reward_mean":      _mean_or_nan(step_portfolio_rewards),
            "train/loss_critic":           _mean_or_nan(ep_losses["critic_loss"]),
            "train/loss_actor":            _mean_or_nan(ep_losses["actor_loss"]),
            "train/alpha":                 _mean_or_nan(ep_losses["alpha"]),
            "train/entropy":               _mean_or_nan(ep_losses["entropy"]),
            **{f"train/{k}": v for k, v in kpis.items()},
            **{f"train/{k}": v for k, v in soc_stats.items()},
        }
        # Per-building train rewards (useful for diagnosing per-building divergence)
        for i, r in enumerate(per_building_rewards):
            log_dict[f"train/building_{i}/reward"] = r

        if use_wandb:
            wandb.log(_filter_log_values(log_dict), step=episode)

        if episode % 10 == 0:
            print(
                f"[train] Ep {episode:4d} | rew_sum {portfolio_reward:9.2f} | "
                f"steps {total_steps:7d} | "
                f"a_loss {_mean_or_nan(ep_losses['actor_loss']):7.4f} | "
                f"ramp {kpis.get('kpi/ramping', float('nan')):.3f} | "
                f"peak {kpis.get('kpi/daily_peak', float('nan')):.3f}"
            )

        # ================================================================
        # 4. PERIODIC CHECKPOINT
        # ================================================================
        if episode % cfg.save_every == 0:
            save_plots(all_rewards, all_kpis, save_dir)
            save_checkpoint(
                agents, save_dir, cfg,
                metrics={"train": {"episode": episode, "reward_sum": portfolio_reward}},
                backup_dir=None,
            )

    # ================================================================
    # 5. FINAL CHECKPOINT
    # ================================================================
    save_plots(all_rewards, all_kpis, save_dir)
    train_summary = {
        "n_episodes":           cfg.n_episodes,
        "last_reward_sum":      all_rewards[-1] if all_rewards else None,
        "mean_reward_last10":   float(np.mean(all_rewards[-10:])) if all_rewards else None,
        "last_kpis":            all_kpis[-1] if all_kpis else {},
        "train_month":          train_month,
        "train_steps":          f"{train_start}-{train_end}",
        "total_env_steps":      total_steps,
        "repro":                repro_meta,
    }

    # ================================================================
    # 6. TEST EVALUATION
    # ================================================================
    test_result: Optional[Dict] = None
    if cfg.do_test and test_start is not None:
        test_result = evaluate_on_test(
            agents, action_spaces, cfg,
            test_start, test_end, use_wandb,
            episode=cfg.n_episodes,
        )

    final_metrics: Dict = {"train": train_summary}
    if test_result is not None:
        final_metrics["test"] = {
            "test_month":  test_month,
            "test_steps":  f"{test_start}-{test_end}",
            **test_result,
        }

    save_checkpoint(
        agents, save_dir, cfg,
        metrics=final_metrics,
        backup_dir=_make_backup_dir(cfg),
    )

    if test_result is not None:
        _export_test_metrics(test_result, cfg, save_dir, test_start, test_end, test_month)

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print(f"  Artifacts: {save_dir}/")
    print(f"  Backup:    {_make_backup_dir(cfg)}/")
    return agents


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Independent SAC Baseline for CityLearn Annex96-CE1"
    )
    # Dataset
    parser.add_argument("--climate",     default="TX",   choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)

    # SAC hyperparameters
    parser.add_argument("--hidden_dim",       type=int,   default=256)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--gamma",            type=float, default=0.99)
    parser.add_argument("--tau",              type=float, default=5e-3)
    parser.add_argument("--alpha_init",       type=float, default=0.2)
    parser.add_argument("--buffer_capacity",  type=int,   default=100_000)
    parser.add_argument("--batch_size",       type=int,   default=256)
    parser.add_argument("--updates_per_step", type=int,   default=1)
    parser.add_argument("--learning_starts",  type=int,   default=1000)
    parser.add_argument("--max_grad_norm",    type=float, default=1.0)

    # Training
    parser.add_argument("--n_episodes",         type=int, default=100)
    parser.add_argument("--episode_time_steps", type=int, default=None)

    # Logging & checkpointing
    parser.add_argument("--save_every",    type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name",    default="independent-sac")
    parser.add_argument("--save_dir",      default="results/independent_sac")
    parser.add_argument("--backup_dir",    default=None)

    # Time windows
    parser.add_argument("--train_month", type=int, default=None,
                        help="Training month 1-12 (default: VT=1, TX=8)")
    parser.add_argument("--test_month",  type=int, default=None,
                        help="Test month 1-12 (default: VT=2, TX=9)")

    # Workflow
    parser.add_argument("--no_test", action="store_true",
                        help="Skip test evaluation after training")

    args = parser.parse_args()

    return Config(
        climate             = args.climate,
        n_buildings         = args.n_buildings,
        hidden_dim          = args.hidden_dim,
        lr                  = args.lr,
        gamma               = args.gamma,
        tau                 = args.tau,
        alpha_init          = args.alpha_init,
        buffer_capacity     = args.buffer_capacity,
        batch_size          = args.batch_size,
        updates_per_step    = args.updates_per_step,
        learning_starts     = args.learning_starts,
        max_grad_norm       = args.max_grad_norm,
        n_episodes          = args.n_episodes,
        episode_time_steps  = args.episode_time_steps,
        train_month         = args.train_month,
        test_month          = args.test_month,
        do_test             = not args.no_test,
        save_every          = args.save_every,
        seed                = args.seed,
        wandb_project       = args.wandb_project,
        wandb_name          = args.wandb_name,
        save_dir            = args.save_dir,
        backup_dir          = args.backup_dir,
    )


if __name__ == "__main__":
    train(parse_args())
