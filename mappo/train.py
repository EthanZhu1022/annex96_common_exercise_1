"""
Hierarchical MAPPO with Communication — Training + Evaluation Script
=====================================================================

Run:
    python -m mappo.train                          # VT: train Jan, test Feb
    python -m mappo.train --climate TX             # TX: train Aug, test Sep
    python -m mappo.train --n_clusters 5
    python -m mappo.train --no_comm
    python -m mappo.train --n_episodes 200 --save_every 20
    python -m mappo.train --no_test                # skip test evaluation

Train-test workflow
-------------------
Default windows (per README):
  VT: train January  (steps    0– 743), test February   (steps  744–1415)
  TX: train August   (steps 5088–5831), test September  (steps 5832–6551)

Each training episode:
  1. Collect a full trajectory on the training time window.
  2. Compute GAE(γ, λ) advantages and discounted returns.
  3. PPO update over minibatches for `ppo_epochs` epochs.
  4. Log per-episode metrics (train/ prefix) to W&B and stdout.
  5. Every save_every episodes: save checkpoint + verify files.

After training:
  6. Force final checkpoint even if not on save interval.
  7. Run one evaluation episode on the test window (no param updates).
  8. Export test KPIs to test_metrics.json and test_metrics.csv.
  9. Log test metrics (test/ prefix) to W&B and stdout.

Checkpoints
-----------
Each checkpoint writes to save_dir (resolved to an absolute path):
  actor_0.pt … actor_{n_clusters-1}.pt
  critic.pt
  comm_net_i.pt  (if communication is enabled)
  run_config.json     — full hyperparameters and time-window settings
  latest_metrics.json — latest train + test summary

A timestamped backup copy is written to save_dir/backups/{ts}_{climate}_seed{seed}/
(override with --backup_dir).  All files are verified to be non-zero after saving.

Reproducibility
---------------
seed_everything(seed) sets random / numpy / torch / cudnn seeds uniformly.
env.reset(seed=…) is called explicitly on every episode.
Reproducibility metadata is printed and saved with every checkpoint.

See agent.py for CTDE design rationale and utils.py for GAE + KPI details.
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
matplotlib.use("Agg")  # non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces

# ── Ensure the local CityLearn copy is used (not any pip-installed version) ──
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

from mappo.agent import Actor, Critic
from mappo.communication import CommunicationNet
from mappo.utils import RolloutBuffer, extract_episode_kpis, get_soc_stats

try:
    import wandb
    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed — W&B logging disabled.")


# ---------------------------------------------------------------------------
# Month → hourly time-step index mapping (non-leap year, 8760 h total)
#
# Verified against the CE1 building CSV files (column 'month').
#   January  rows 0–743,    August    rows 5088–5831
#   February rows 744–1415, September rows 5832–6551
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

# Default train/test months per climate (from README instructions)
_CLIMATE_DEFAULTS: Dict[str, Dict[str, int]] = {
    "VT": {"train_month": 1, "test_month": 2},  # January / February
    "TX": {"train_month": 8, "test_month": 9},  # August / September
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────
    climate:     str = "VT"   # "VT" (heating-dominated) or "TX" (cooling-dominated)
    n_buildings: int = 25     # how many buildings to include (max 25 in CE1)

    # ── Clustering ────────────────────────────────────────────────────────
    # Buildings are assigned round-robin to clusters to preserve CityLearn
    # ordering and handle cases where n_buildings % n_clusters != 0.
    n_clusters: int = 5       # number of cluster agents (each controls ~5 buildings)

    # ── Episode horizon ───────────────────────────────────────────────────
    # None → use the full training-month window (from _MONTH_STARTS/ENDS).
    # Set an int to override, e.g. 168 for a one-week horizon.
    episode_time_steps: Optional[int] = None

    # ── Network architecture ──────────────────────────────────────────────
    hidden_dim:        int  = 256   # hidden layer width for Actor and Critic
    msg_dim:           int  = 32    # per-agent communication message size
    use_communication: bool = True  # set False to ablate communication

    # ── Training ──────────────────────────────────────────────────────────
    n_episodes:    int   = 100
    gamma:         float = 0.99    # discount factor
    gae_lambda:    float = 0.95    # GAE λ
    clip_eps:      float = 0.2     # PPO clip ε
    entropy_coeff: float = 0.01    # entropy bonus coefficient
    value_coeff:   float = 0.5     # critic loss coefficient
    lr_actor:      float = 3e-4
    lr_critic:     float = 1e-3
    ppo_epochs:    int   = 4       # number of PPO update epochs per episode
    minibatch_size: int  = 64
    max_grad_norm: float = 0.5     # gradient clipping

    # ── Time windows ──────────────────────────────────────────────────────
    # Month indices 1-12.  None → filled from _CLIMATE_DEFAULTS at init time.
    train_month: Optional[int] = None
    test_month:  Optional[int] = None

    # ── Workflow ──────────────────────────────────────────────────────────
    do_test: bool = True    # run one test episode after training (no param updates)

    # ── Logging & output ──────────────────────────────────────────────────
    save_every:    int  = 10              # checkpoint every N episodes
    eval_freq:     int  = 10             # kept for backward compat (alias of save_every)
    seed:          int  = 42
    wandb_project: str  = "citylearn-mappo"
    wandb_name:    str  = "clustered-mappo"
    save_dir:      str  = "results/mappo"
    backup_dir:    Optional[str] = None  # None → auto: save_dir/backups/{ts}_{climate}_seed{seed}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for reproducibility.

    When CUDA is available, also enables deterministic cuDNN algorithms.
    Note: deterministic mode may reduce throughput slightly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_repro_metadata(
    cfg: "Config",
    device: torch.device,
    train_start: int,
    train_end: int,
    test_start: Optional[int],
    test_end: Optional[int],
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
# Environment helpers
# ---------------------------------------------------------------------------

def build_env(
    cfg: "Config",
    start_step: Optional[int] = None,
    end_step:   Optional[int] = None,
) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate and wrap CityLearnEnv.

    start_step / end_step override simulation_start/end_time_step in the schema,
    enabling train/test window switching without editing schema.json.

    We use central_agent=False so that:
      - env.observation_space[i] → Box for building i
      - env.action_space[i]      → Box for building i
      - env.step(actions)        expects List[List[float]], one per building
      - reward from step()       is List[float], one per building
    """
    dataset_name = f"annex96_ce1_{cfg.climate.lower()}_neighborhood"
    dataset_dir  = REPO_DIR / "data" / "datasets" / dataset_name
    schema_path  = dataset_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found: {schema_path}\n"
            f"Available datasets: {[d.name for d in (REPO_DIR / 'data' / 'datasets').iterdir()]}"
        )

    env_kwargs: Dict = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=False,
        buildings=list(range(cfg.n_buildings)),
    )

    if start_step is not None and end_step is not None:
        # Override time window (train or test month) without touching schema.json.
        # Also set episode_time_steps to the full window so CityLearn treats the
        # month as a single episode regardless of what the schema says.
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


def create_clusters(n_buildings: int, n_clusters: int) -> List[List[int]]:
    """Assign buildings to clusters in round-robin order.

    Round-robin spreads the load evenly when n_buildings % n_clusters != 0.

    Example (25 buildings, 5 clusters):
        cluster 0: [0, 5, 10, 15, 20]
        cluster 1: [1, 6, 11, 16, 21]
        ...
    """
    clusters: List[List[int]] = [[] for _ in range(n_clusters)]
    for b in range(n_buildings):
        clusters[b % n_clusters].append(b)
    return clusters


# ---------------------------------------------------------------------------
# Obs / action assembly helpers
# ---------------------------------------------------------------------------

def build_cluster_obs(
    obs_list: List[List[float]],
    clusters: List[List[int]],
) -> List[np.ndarray]:
    """Concatenate per-building obs for each cluster."""
    return [
        np.concatenate([obs_list[b] for b in cluster], dtype=np.float32)
        for cluster in clusters
    ]


def build_global_obs(obs_list: List[List[float]]) -> np.ndarray:
    """Concatenate ALL buildings' obs into one global state vector (for critic)."""
    return np.concatenate(obs_list, dtype=np.float32)


def build_actor_inputs(
    cluster_obs:       List[np.ndarray],
    comm_nets:         List[Optional[CommunicationNet]],
    use_communication: bool,
    device:            torch.device,
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Compute communication messages and build actor inputs.

    When use_communication=False, actor inputs are just the cluster obs.
    """
    n_agents = len(cluster_obs)

    if use_communication:
        with torch.no_grad():
            msgs = []
            for i in range(n_agents):
                x = torch.from_numpy(cluster_obs[i]).unsqueeze(0).to(device)
                m = comm_nets[i](x).squeeze(0).cpu().numpy()
                msgs.append(m)
        M_np = np.concatenate(msgs, dtype=np.float32)
        actor_inputs = [
            np.concatenate([cluster_obs[i], M_np], dtype=np.float32)
            for i in range(n_agents)
        ]
    else:
        M_np         = None
        actor_inputs = [cluster_obs[i].copy() for i in range(n_agents)]

    return actor_inputs, M_np


def assemble_env_actions(
    cluster_actions: List[np.ndarray],
    clusters:        List[List[int]],
    act_dims:        List[int],
    action_spaces:   List[spaces.Box],
    n_buildings:     int,
) -> List[List[float]]:
    """Split cluster action vectors back into per-building action lists.

    CityLearn expects actions as List[List[float]] where actions[b] is the
    action vector for building b.
    """
    env_actions: List[Optional[List[float]]] = [None] * n_buildings
    for cluster_idx, building_indices in enumerate(clusters):
        vec    = cluster_actions[cluster_idx]
        offset = 0
        for b in building_indices:
            dim = act_dims[b]
            raw_action   = vec[offset: offset + dim]
            action_space = action_spaces[b]
            low  = action_space.low.astype(np.float32)
            high = action_space.high.astype(np.float32)
            scaled = low + (raw_action + 1.0) * 0.5 * (high - low)
            env_actions[b] = np.clip(scaled, low, high).tolist()
            offset += dim
    return env_actions  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    actors:     List[Actor],
    critic:     Critic,
    actor_opts: List[Adam],
    critic_opt: Adam,
    buffer:     RolloutBuffer,
    cfg:        Config,
    device:     torch.device,
) -> Tuple[float, float, float]:
    """Run PPO_EPOCHS passes over the rollout buffer.

    Returns (avg_actor_loss, avg_critic_loss, avg_entropy) over all updates.
    """
    n_clusters       = len(actors)
    total_actor_loss = 0.0
    total_crit_loss  = 0.0
    total_entropy    = 0.0
    n_batches        = 0

    for _ in range(cfg.ppo_epochs):
        for batch in buffer.get_minibatches(cfg.minibatch_size, device):
            adv = batch["advantages"]

            for i in range(n_clusters):
                new_lp, entropy = actors[i].evaluate_actions(
                    batch["actor_inputs"][i],
                    batch["actions"][i],
                )
                ratio = (new_lp - batch["old_log_probs"][i]).exp()
                surr1 = ratio * adv
                surr2 = ratio.clamp(1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - cfg.entropy_coeff * entropy.mean()

                actor_opts[i].zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actors[i].parameters(), cfg.max_grad_norm)
                actor_opts[i].step()

                total_actor_loss += actor_loss.item()
                total_entropy    += entropy.mean().item()

            v_pred    = critic(batch["global_obs"])
            crit_loss = cfg.value_coeff * F.mse_loss(v_pred, batch["returns"])

            critic_opt.zero_grad()
            crit_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            total_crit_loss += crit_loss.item()
            n_batches       += 1

    denom = max(n_batches, 1)
    return (
        total_actor_loss / (denom * n_clusters),
        total_crit_loss  / denom,
        total_entropy    / (denom * n_clusters),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(
    episode_rewards: List[float],
    actor_losses:    List[float],
    critic_losses:   List[float],
    entropies:       List[float],
    kpis_list:       List[Dict],
    save_dir:        Path,
    rbc_kpis:        Optional[pd.DataFrame] = None,
) -> None:
    """Save training-curve and KPI-trend figures."""
    eps = list(range(1, len(episode_rewards) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("MAPPO Training — CityLearn CE1", fontsize=14)

    axes[0, 0].plot(eps, episode_rewards)
    axes[0, 0].set_title("Episode Reward (train)")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(eps, actor_losses,  label="Actor")
    axes[0, 1].plot(eps, critic_losses, label="Critic")
    axes[0, 1].legend()
    axes[0, 1].set_title("Loss (train)")
    axes[0, 1].set_xlabel("Episode")

    axes[0, 2].plot(eps, entropies)
    axes[0, 2].set_title("Policy Entropy (train)")
    axes[0, 2].set_xlabel("Episode")

    kpi_cfg = [
        ("kpi/ramping",      "Ramping (avg)",    "ramping_average"),
        ("kpi/daily_peak",   "Daily Peak (avg)", "daily_peak_average"),
        ("kpi/all_time_peak","All-Time Peak",    "all_time_peak_average"),
    ]
    for ax, (key, title, rbc_col) in zip(axes[1, :], kpi_cfg):
        vals  = [k.get(key) for k in kpis_list]
        valid = [(e, v) for e, v in zip(eps, vals) if v is not None]
        if valid:
            ep_v, val_v = zip(*valid)
            ax.plot(ep_v, val_v, label="MAPPO")
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="No-control (=1)")
        if rbc_kpis is not None and rbc_col in rbc_kpis.columns:
            try:
                rbc_val = float(rbc_kpis[rbc_col].iloc[0])
                ax.axhline(rbc_val, color="orange", linestyle="--", alpha=0.7, label="RBC")
            except Exception:
                pass
        ax.set_title(f"KPI: {title}")
        ax.set_xlabel("Episode")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  [plot] saved → {out}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_checkpoint(
    actors:      List[Actor],
    critic:      Critic,
    comm_nets:   List[Optional[CommunicationNet]],
    save_dir:    Path,
    cfg:         Config,
    metrics:     Optional[Dict] = None,
    backup_dir:  Optional[Path] = None,
) -> None:
    """Save models, config, and metrics; verify files; write backup.

    Raises RuntimeError if any file is missing or zero-size after saving.
    """
    save_dir = Path(save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    # ── Model weights ────────────────────────────────────────────────────
    for i, actor in enumerate(actors):
        p = save_dir / f"actor_{i}.pt"
        torch.save(actor.state_dict(), p)
        saved.append(p)

    p = save_dir / "critic.pt"
    torch.save(critic.state_dict(), p)
    saved.append(p)

    for i, cn in enumerate(comm_nets):
        if cn is not None:
            p = save_dir / f"comm_net_{i}.pt"
            torch.save(cn.state_dict(), p)
            saved.append(p)

    # ── Config ───────────────────────────────────────────────────────────
    cfg_dict = {k: v for k, v in vars(cfg).items()}
    p = save_dir / "run_config.json"
    p.write_text(json.dumps(cfg_dict, indent=2))
    saved.append(p)

    # ── Metrics ──────────────────────────────────────────────────────────
    if metrics is not None:
        p = save_dir / "latest_metrics.json"
        p.write_text(json.dumps(metrics, indent=2))
        saved.append(p)

    # ── Verify primary saves ─────────────────────────────────────────────
    for p in saved:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(
                f"Checkpoint verification failed: {p} is missing or empty."
            )

    print(f"  [ckpt] {len(saved)} files saved → {save_dir}/")

    # ── Backup ───────────────────────────────────────────────────────────
    if backup_dir is not None:
        backup_dir = Path(backup_dir).resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        for p in saved:
            shutil.copy2(p, backup_dir / p.name)
        # Verify backup
        for p in saved:
            bp = backup_dir / p.name
            if not bp.exists() or bp.stat().st_size == 0:
                raise RuntimeError(
                    f"Backup verification failed: {bp} is missing or empty."
                )
        print(f"  [backup] {len(saved)} files copied → {backup_dir}/")


def _make_backup_dir(cfg: Config) -> Path:
    """Build the backup directory path from config fields."""
    if cfg.backup_dir is not None:
        return Path(cfg.backup_dir).resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_{cfg.climate}_seed{cfg.seed}"
    return (Path(cfg.save_dir).resolve() / "backups" / tag)


# ---------------------------------------------------------------------------
# Test evaluation (no parameter updates)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    actors:      List[Actor],
    critic:      Critic,
    comm_nets:   List[Optional[CommunicationNet]],
    clusters:    List[List[int]],
    act_dims:    List[int],
    action_spaces: List[spaces.Box],
    cfg:         Config,
    device:      torch.device,
    test_start:  int,
    test_end:    int,
    use_wandb:   bool,
    episode:     Optional[int] = None,
) -> Dict:
    """Run one episode on the test time window with no parameter updates.

    Returns a dict with test_reward and test KPIs.
    """
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    print(f"\n{'='*65}")
    print(
        f"TEST EVALUATION | {cfg.climate} | {month_name} "
        f"(steps {test_start}–{test_end})"
    )
    print(f"{'='*65}")

    test_env, test_base_env = build_env(cfg, start_step=test_start, end_step=test_end)

    n_clusters    = len(actors)
    n_buildings   = len(test_base_env.buildings)

    # Switch to eval mode (disables dropout / batch-norm updates if present)
    for actor in actors:
        actor.eval()
    critic.eval()
    for cn in comm_nets:
        if cn is not None:
            cn.eval()

    obs_list, _ = test_env.reset(seed=cfg.seed)
    test_reward  = 0.0
    step_rewards: List[float] = []

    with torch.no_grad():
        while not test_base_env.terminated:
            cluster_obs = build_cluster_obs(obs_list, clusters)
            actor_inputs, _ = build_actor_inputs(
                cluster_obs, comm_nets, cfg.use_communication, device
            )

            actions_np: List[np.ndarray] = []
            for i in range(n_clusters):
                x = torch.from_numpy(actor_inputs[i]).unsqueeze(0).to(device)
                a, _ = actors[i].act(x)
                actions_np.append(a.squeeze(0).cpu().numpy())

            env_actions = assemble_env_actions(
                actions_np, clusters, act_dims, action_spaces, n_buildings
            )
            next_obs_list, rewards, terminated, truncated, _ = test_env.step(env_actions)
            global_reward = float(sum(rewards))
            test_reward  += global_reward
            step_rewards.append(global_reward)
            obs_list = next_obs_list

    # Restore train mode
    for actor in actors:
        actor.train()
    critic.train()
    for cn in comm_nets:
        if cn is not None:
            cn.train()

    test_kpis     = extract_episode_kpis(test_base_env)
    test_soc      = get_soc_stats(test_base_env)

    result: Dict = {
        "test/episode_reward":   test_reward,
        "test/step_reward_mean": float(np.mean(step_rewards)) if step_rewards else 0.0,
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
    }
    if episode is not None:
        result["episode"] = episode

    # ── Console summary ──────────────────────────────────────────────────
    print(
        f"  reward {test_reward:9.2f} | "
        f"ramp {test_kpis.get('kpi/ramping', float('nan')):.3f} | "
        f"peak {test_kpis.get('kpi/daily_peak', float('nan')):.3f} | "
        f"cost {test_kpis.get('kpi/cost', float('nan')):.3f}"
    )

    if use_wandb:
        wandb.log(_filter_log_values(result), step=episode)

    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: Config) -> Tuple[List[Actor], Critic]:
    """Full MAPPO training loop followed by optional test evaluation.

    Returns the trained actors and critic.
    """
    # ── Resolve time windows ──────────────────────────────────────────────
    defaults = _CLIMATE_DEFAULTS.get(cfg.climate, {})
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

    # Sync save_every / eval_freq aliases (eval_freq takes priority for backward compat)
    save_every = cfg.eval_freq if cfg.eval_freq != 10 else cfg.save_every

    # Stash resolved months back into cfg so they are recorded in run_config.json
    cfg.train_month = train_month
    cfg.test_month  = test_month

    # ── Seeding & device ─────────────────────────────────────────────────
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory (always absolute) ───────────────────────────────
    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility metadata ─────────────────────────────────────────
    repro_meta = print_repro_metadata(
        cfg, device, train_start, train_end, test_start, test_end
    )

    # ── W&B ──────────────────────────────────────────────────────────────
    use_wandb = _WANDB_OK
    cfg_dict  = vars(cfg).copy()
    cfg_dict.update({
        "train_start_step": train_start,
        "train_end_step":   train_end,
        "test_start_step":  test_start,
        "test_end_step":    test_end,
    })
    if use_wandb:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg_dict)
        wandb.define_metric("episode")
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("test/*", step_metric="episode")

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
    global_obs_dim = sum(obs_dims)

    print(
        f"Environment: {cfg.climate} | buildings: {n_buildings} | "
        f"obs/building: {obs_dims[0]} | act/building: {act_dims[0]} | "
        f"global obs: {global_obs_dim}"
    )

    # ── Clusters ─────────────────────────────────────────────────────────
    clusters         = create_clusters(n_buildings, cfg.n_clusters)
    n_clusters       = len(clusters)
    cluster_obs_dims = [sum(obs_dims[b] for b in cl) for cl in clusters]
    cluster_act_dims = [sum(act_dims[b] for b in cl) for cl in clusters]

    print(f"Clusters ({n_clusters}): {[len(c) for c in clusters]} buildings each")

    # ── Communication networks ────────────────────────────────────────────
    total_msg_dim = n_clusters * cfg.msg_dim if cfg.use_communication else 0
    comm_nets: List[Optional[CommunicationNet]] = (
        [CommunicationNet(cluster_obs_dims[i], cfg.msg_dim).to(device)
         for i in range(n_clusters)]
        if cfg.use_communication else [None] * n_clusters
    )

    # ── Actors and Critic ────────────────────────────────────────────────
    actor_input_dims = [cluster_obs_dims[i] + total_msg_dim for i in range(n_clusters)]
    actors = [
        Actor(actor_input_dims[i], cluster_act_dims[i], cfg.hidden_dim).to(device)
        for i in range(n_clusters)
    ]
    critic = Critic(global_obs_dim, cfg.hidden_dim).to(device)

    # ── Optimizers ────────────────────────────────────────────────────────
    actor_param_groups = [
        list(actors[i].parameters()) +
        (list(comm_nets[i].parameters()) if comm_nets[i] is not None else [])
        for i in range(n_clusters)
    ]
    actor_opts = [Adam(pg, lr=cfg.lr_actor) for pg in actor_param_groups]
    critic_opt = Adam(critic.parameters(), lr=cfg.lr_critic)

    # ── Rollout buffer ────────────────────────────────────────────────────
    buffer = RolloutBuffer(n_clusters)

    # ── Optional RBC baseline ─────────────────────────────────────────────
    rbc_kpis: Optional[pd.DataFrame] = None
    rbc_kpi_file = REPO_DIR / "notebooks" / "rbc_baseline_kpi_summary.csv"
    if rbc_kpi_file.exists():
        try:
            rbc_kpis = pd.read_csv(rbc_kpi_file)
            print(f"Loaded RBC baseline from {rbc_kpi_file}")
        except Exception as exc:
            print(f"[warn] Could not load RBC baseline: {exc}")
    else:
        print("[info] RBC baseline file not found — skipping comparison.")

    # ── Metric history ────────────────────────────────────────────────────
    all_rewards:      List[float] = []
    all_actor_losses: List[float] = []
    all_crit_losses:  List[float] = []
    all_entropies:    List[float] = []
    all_kpis:         List[Dict]  = []

    print("=" * 65)
    print(
        f"MAPPO | Climate: {cfg.climate} | comm: {cfg.use_communication} | "
        f"episodes: {cfg.n_episodes} | save_every: {save_every}"
    )
    print("=" * 65)

    for episode in range(1, cfg.n_episodes + 1):

        # ================================================================
        # 1. ROLLOUT COLLECTION
        # ================================================================
        # Episode 1 uses the base seed; subsequent episodes use seed+episode
        # so each episode starts from a slightly different initial state
        # while remaining fully reproducible across runs.
        ep_seed = cfg.seed if episode == 1 else cfg.seed + episode
        obs_list, _ = env.reset(seed=ep_seed)
        buffer.clear()

        episode_reward = 0.0
        step_rewards:  List[float] = []

        while not base_env.terminated:
            cluster_obs = build_cluster_obs(obs_list, clusters)
            global_obs  = build_global_obs(obs_list)

            actor_inputs, _ = build_actor_inputs(
                cluster_obs, comm_nets, cfg.use_communication, device
            )

            actions_np:   List[np.ndarray] = []
            log_probs_np: List[float]       = []
            with torch.no_grad():
                for i in range(n_clusters):
                    x = torch.from_numpy(actor_inputs[i]).unsqueeze(0).to(device)
                    a, lp = actors[i].act(x)
                    actions_np.append(a.squeeze(0).cpu().numpy())
                    log_probs_np.append(lp.item())

                g_t   = torch.from_numpy(global_obs).unsqueeze(0).to(device)
                value = critic(g_t).item()

            env_actions = assemble_env_actions(
                actions_np, clusters, act_dims, action_spaces, n_buildings
            )

            next_obs_list, rewards, terminated, truncated, _ = env.step(env_actions)
            global_reward = float(sum(rewards))
            done = terminated or truncated

            buffer.add(
                actor_inputs=actor_inputs,
                actions=actions_np,
                log_probs=log_probs_np,
                global_obs=global_obs,
                reward=global_reward,
                done=done,
                value=value,
            )

            episode_reward += global_reward
            step_rewards.append(global_reward)
            obs_list = next_obs_list

        # Bootstrap value for the last state
        with torch.no_grad():
            last_g_obs = build_global_obs(obs_list)
            last_val   = (
                critic(torch.from_numpy(last_g_obs).unsqueeze(0).to(device)).item()
                if not base_env.terminated else 0.0
            )
        buffer.compute_returns_and_advantages(last_val, cfg.gamma, cfg.gae_lambda)

        # ================================================================
        # 2. PPO UPDATE
        # ================================================================
        actor_loss, crit_loss, entropy = ppo_update(
            actors, critic, actor_opts, critic_opt, buffer, cfg, device
        )

        # ================================================================
        # 3. KPI EXTRACTION & LOGGING
        # ================================================================
        kpis      = extract_episode_kpis(base_env)
        soc_stats = get_soc_stats(base_env)

        all_rewards.append(episode_reward)
        all_actor_losses.append(actor_loss)
        all_crit_losses.append(crit_loss)
        all_entropies.append(entropy)
        all_kpis.append(kpis)

        log_dict = {
            "episode":                  episode,
            "train/episode_reward":     episode_reward,
            "train/loss_actor":         actor_loss,
            "train/loss_critic":        crit_loss,
            "train/entropy":            entropy,
            "train/step_reward_mean":   float(np.mean(step_rewards)),
            **{f"train/{k}": v for k, v in kpis.items()},
            **{f"train/{k}": v for k, v in soc_stats.items()},
        }
        if use_wandb:
            wandb.log(_filter_log_values(log_dict), step=episode)

        if episode % 10 == 0:
            print(
                f"[train] Ep {episode:4d} | rew {episode_reward:9.2f} | "
                f"a_loss {actor_loss:7.4f} | c_loss {crit_loss:7.4f} | "
                f"ent {entropy:5.3f} | "
                f"ramp {kpis.get('kpi/ramping', float('nan')):.3f} | "
                f"peak {kpis.get('kpi/daily_peak', float('nan')):.3f}"
            )

        # ================================================================
        # 4. PERIODIC CHECKPOINT
        # ================================================================
        if episode % save_every == 0:
            save_plots(
                all_rewards, all_actor_losses, all_crit_losses,
                all_entropies, all_kpis, save_dir, rbc_kpis,
            )
            save_checkpoint(
                actors, critic, comm_nets, save_dir, cfg,
                metrics={
                    "train": {
                        "episode": episode,
                        "episode_reward": episode_reward,
                        "actor_loss": actor_loss,
                        "critic_loss": crit_loss,
                        "entropy": entropy,
                        "kpis": kpis,
                    }
                },
                backup_dir=None,  # backup only at final checkpoint
            )

    # ================================================================
    # 5. FINAL CHECKPOINT (forced, even if last episode was on interval)
    # ================================================================
    save_plots(
        all_rewards, all_actor_losses, all_crit_losses,
        all_entropies, all_kpis, save_dir, rbc_kpis,
    )
    train_summary = {
        "n_episodes":     cfg.n_episodes,
        "last_reward":    all_rewards[-1]    if all_rewards    else None,
        "last_kpis":      all_kpis[-1]       if all_kpis       else {},
        "mean_reward_last10": float(np.mean(all_rewards[-10:])) if all_rewards else None,
        "train_month":    train_month,
        "train_steps":    f"{train_start}-{train_end}",
        "repro":          repro_meta,
    }

    # ================================================================
    # 6. TEST EVALUATION
    # ================================================================
    test_result: Optional[Dict] = None
    if cfg.do_test and test_start is not None:
        test_result = evaluate_on_test(
            actors, critic, comm_nets, clusters,
            act_dims, action_spaces, cfg, device,
            test_start, test_end, use_wandb, episode=cfg.n_episodes,
        )

    # ── Build final metrics dict ──────────────────────────────────────
    final_metrics: Dict = {"train": train_summary}
    if test_result is not None:
        final_metrics["test"] = {
            "test_month":  test_month,
            "test_steps":  f"{test_start}-{test_end}",
            **test_result,
        }

    save_checkpoint(
        actors, critic, comm_nets, save_dir, cfg,
        metrics=final_metrics,
        backup_dir=_make_backup_dir(cfg),
    )

    # ── Export test metrics to CSV / JSON ────────────────────────────
    if test_result is not None:
        _export_test_metrics(test_result, cfg, save_dir, test_start, test_end, test_month)

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print(f"  Artifacts: {save_dir}/")
    print(f"  Backup:    {_make_backup_dir(cfg)}/")
    return actors, critic


# ---------------------------------------------------------------------------
# Test-metrics export helpers
# ---------------------------------------------------------------------------

def _export_test_metrics(
    test_result:  Dict,
    cfg:          Config,
    save_dir:     Path,
    test_start:   int,
    test_end:     int,
    test_month:   int,
) -> None:
    """Write test metrics to JSON and CSV in save_dir."""
    payload = {
        "climate":    cfg.climate,
        "seed":       cfg.seed,
        "test_month": test_month,
        "test_month_name": _MONTH_NAMES.get(test_month, str(test_month)),
        "test_start_step": test_start,
        "test_end_step":   test_end,
        **test_result,
    }

    json_path = save_dir / "test_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"  [test]  metrics JSON → {json_path}")

    # Flatten for CSV (skip None values)
    flat = {k: v for k, v in payload.items() if v is not None and not isinstance(v, dict)}
    csv_path = save_dir / "test_metrics.csv"
    pd.DataFrame([flat]).to_csv(csv_path, index=False)
    print(f"  [test]  metrics CSV  → {csv_path}")


def _filter_log_values(metrics: Dict) -> Dict:
    """Drop None/NaN/inf values before logging to W&B."""
    clean: Dict = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            continue
        clean[key] = value
    return clean


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Hierarchical MAPPO with Communication for CityLearn CE1"
    )
    # ── Core ─────────────────────────────────────────────────────────────
    parser.add_argument("--climate",            default="VT",  choices=["VT", "TX"])
    parser.add_argument("--n_buildings",        type=int,   default=25)
    parser.add_argument("--n_clusters",         type=int,   default=5)
    parser.add_argument("--episode_time_steps", type=int,   default=None,
                        help="Episode length in hours (default: full training month)")
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--msg_dim",            type=int,   default=32)
    parser.add_argument("--no_comm",            action="store_true",
                        help="Disable communication between agents")
    parser.add_argument("--n_episodes",         type=int,   default=100)
    parser.add_argument("--ppo_epochs",         type=int,   default=4)
    parser.add_argument("--minibatch_size",     type=int,   default=64)
    parser.add_argument("--lr_actor",           type=float, default=3e-4)
    parser.add_argument("--lr_critic",          type=float, default=1e-3)

    # ── Logging & checkpointing ───────────────────────────────────────────
    parser.add_argument("--save_every",  type=int,   default=None,
                        help="Checkpoint every N episodes (default: 10)")
    parser.add_argument("--eval_freq",   type=int,   default=None,
                        help="Alias for --save_every (backward compat)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--wandb_project", default="citylearn-mappo")
    parser.add_argument("--wandb_name",    default="clustered-mappo")
    parser.add_argument("--save_dir",    default="results/mappo")
    parser.add_argument("--backup_dir",  default=None,
                        help="Backup directory (default: save_dir/backups/{ts}_{climate}_seed{seed})")

    # ── Time windows ──────────────────────────────────────────────────────
    parser.add_argument("--train_month", type=int, default=None,
                        help="Training month 1-12 (default: VT=1, TX=8)")
    parser.add_argument("--test_month",  type=int, default=None,
                        help="Testing month 1-12 (default: VT=2, TX=9)")

    # ── Workflow ──────────────────────────────────────────────────────────
    parser.add_argument("--no_test",     action="store_true",
                        help="Skip test evaluation after training")

    args = parser.parse_args()

    # Resolve save_every / eval_freq aliases
    save_every_val = args.save_every or args.eval_freq or 10

    return Config(
        climate            = args.climate,
        n_buildings        = args.n_buildings,
        n_clusters         = args.n_clusters,
        episode_time_steps = args.episode_time_steps,
        hidden_dim         = args.hidden_dim,
        msg_dim            = args.msg_dim,
        use_communication  = not args.no_comm,
        n_episodes         = args.n_episodes,
        ppo_epochs         = args.ppo_epochs,
        minibatch_size     = args.minibatch_size,
        lr_actor           = args.lr_actor,
        lr_critic          = args.lr_critic,
        train_month        = args.train_month,
        test_month         = args.test_month,
        do_test            = not args.no_test,
        save_every         = save_every_val,
        eval_freq          = save_every_val,
        seed               = args.seed,
        wandb_project      = args.wandb_project,
        wandb_name         = args.wandb_name,
        save_dir           = args.save_dir,
        backup_dir         = args.backup_dir,
    )


if __name__ == "__main__":
    train(parse_args())
