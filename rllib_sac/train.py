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
- RLlib result dicts are logged at every training iteration.
- Custom WandBCallback (below) also logs per-building episode rewards and
  district-level CityLearn KPIs extracted after each episode ends.

Output artifacts (in save_dir/):
    checkpoint/        — RLlib checkpoint (actor/critic weights for all agents)
    run_config.json    — full hyperparameter snapshot
    test_metrics.json  — test-episode KPIs (comparable with other baselines)
    test_metrics.csv
    training_curves.png

Run:
    python -m rllib_sac.train                        # TX: Aug train, Sep test
    python -m rllib_sac.train --climate VT           # VT: Jan train, Feb test
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
from mappo.utils import extract_episode_kpis, get_soc_stats

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # Dataset
    climate:     str = "TX"
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
    wandb_name:    str          = "rllib-sac"
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
    """RLlib callback that logs per-episode rewards, KPIs, and portfolio sums.

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

        # Write into episode custom metrics so RLlib aggregates them
        episode.custom_metrics["portfolio_reward_sum"] = portfolio_reward
        for k, v in kpis.items():
            if v is not None:
                episode.custom_metrics[k.replace("/", "_")] = v
        for agent_id, r in agent_rewards.items():
            episode.custom_metrics[f"{agent_id}_reward"] = r

    def on_train_result(
        self,
        *,
        algorithm: Any,
        result: Dict,
        **kwargs,
    ) -> None:
        """Called after each training iteration with the aggregated result.

        Logs everything to W&B if available.
        """
        if not _WANDB_OK:
            return

        iteration = result.get("training_iteration", 0)

        log: Dict[str, Any] = {"episode": iteration}

        # Portfolio reward aggregated by RLlib from custom_metrics
        cm = result.get("custom_metrics", {})
        for key in ["portfolio_reward_sum_mean", "portfolio_reward_sum_max"]:
            if key in cm:
                log[f"train/portfolio/{key}"] = cm[key]

        # Per-building rewards from custom_metrics
        for k, v in cm.items():
            if k.endswith("_reward_mean") and k.startswith("building_"):
                log[f"train/{k.replace('_reward_mean', '')}/reward"] = v

        # KPIs
        for k, v in cm.items():
            if k.startswith("kpi_") and k.endswith("_mean"):
                kpi_name = k[len("kpi_"):-len("_mean")].replace("_", "/")
                log[f"train/kpi/{kpi_name}"] = v

        # RLlib SAC-specific loss metrics
        info = result.get("info", {}).get("learner", {})
        for policy_id, policy_stats in info.items():
            if not isinstance(policy_stats, dict):
                continue
            for stat_key in ["actor_loss", "critic_loss", "alpha_value", "entropy"]:
                if stat_key in policy_stats:
                    log[f"train/loss/{policy_id}/{stat_key}"] = policy_stats[stat_key]
            break  # log first policy only to keep W&B tidy

        # Episode stats from result
        if "episode_reward_mean" in result:
            log["train/episode_reward_mean"] = result["episode_reward_mean"]
        if "episodes_this_iter" in result:
            log["train/episodes_this_iter"] = result["episodes_this_iter"]

        wandb.log(_filter_wandb(log), step=iteration)

        # Clear accumulator
        self._ep_portfolio_rewards.clear()
        self._ep_kpis.clear()


def _filter_wandb(d: Dict) -> Dict:
    """Drop None / NaN / inf values before logging."""
    out: Dict = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            continue
        out[k] = v
    return out


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
# Plotting
# ---------------------------------------------------------------------------

def save_plots(rewards: List[float], kpis: List[Dict], save_dir: Path) -> None:
    if not rewards:
        return
    iters = list(range(1, len(rewards) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("RLlib SAC — CityLearn CE1", fontsize=14)

    axes[0].plot(iters, rewards)
    axes[0].set_title("Mean Episode Reward (train)")
    axes[0].set_xlabel("Iteration")

    for ax, key, title in [
        (axes[1], "kpi/ramping",    "Ramping (avg)"),
        (axes[2], "kpi/daily_peak", "Daily Peak (avg)"),
    ]:
        vals  = [k.get(key) for k in kpis]
        valid = [(e, v) for e, v in zip(iters, vals) if v is not None]
        if valid:
            ev, vv = zip(*valid)
            ax.plot(ev, vv)
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="No-control (=1)")
        ax.set_title(f"KPI: {title}")
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = save_dir / "training_curves.png"
    plt.savefig(str(out), dpi=100)
    plt.close()
    print(f"  [plot] saved → {out}")


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
    print(f"TEST EVALUATION | {cfg.climate} | {month_name} (steps {test_start}–{test_end})")
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
        obs_dict = next_obs

    test_kpis = extract_episode_kpis(test_env_obj.base_env)
    test_soc  = get_soc_stats(test_env_obj.base_env)
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
        f"ramp {_fmt(test_kpis.get('kpi/ramping'))} | "
        f"peak {_fmt(test_kpis.get('kpi/daily_peak'))} | "
        f"cost {_fmt(test_kpis.get('kpi/cost'))}"
    )

    result: Dict = {
        "test/portfolio/reward_sum":  portfolio_reward,
        "test/portfolio/reward_mean": float(np.mean(list(per_building_rewards.values()))),
        "test/step_reward_mean":      float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0,
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
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

    flat = {k: v for k, v in payload.items() if v is not None and not isinstance(v, dict)}
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
    print(f"RLlib SAC | Climate: {cfg.climate} | agents: {len(agent_ids)} | "
          f"iterations: {cfg.n_iterations}")
    print("=" * 65)

    all_rewards:  List[float] = []
    all_kpis:     List[Dict]  = []
    ckpt_dir_path: Optional[str] = None

    for iteration in range(1, cfg.n_iterations + 1):
        result = algorithm.train()

        ep_reward_mean = result.get("episode_reward_mean", float("nan"))
        all_rewards.append(ep_reward_mean)

        # Extract KPI from custom_metrics if available
        cm   = result.get("custom_metrics", {})
        kpis = {k.replace("kpi_", "kpi/").replace("_mean", ""):
                cm.get(k) for k in cm if k.startswith("kpi_") and k.endswith("_mean")}
        all_kpis.append(kpis)

        if iteration % 10 == 0:
            portfolio_mean = cm.get("portfolio_reward_sum_mean", float("nan"))
            ramp_val       = cm.get("kpi_ramping_average_mean", float("nan"))
            print(
                f"[train] Iter {iteration:4d} | ep_rew_mean {ep_reward_mean:9.2f} | "
                f"portfolio_sum {portfolio_mean:9.2f} | "
                f"ramp {ramp_val:.3f}" if np.isfinite(ramp_val) else
                f"[train] Iter {iteration:4d} | ep_rew_mean {ep_reward_mean:9.2f}"
            )

        # Periodic checkpoint
        if iteration % max(cfg.n_iterations // 10, 1) == 0 or iteration == cfg.n_iterations:
            ckpt = algorithm.save(str(save_dir / "checkpoint"))
            ckpt_dir_path = str(ckpt)
            print(f"  [ckpt] iter {iteration} → {ckpt_dir_path}")
            save_plots(all_rewards, all_kpis, save_dir)

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

    # ── Final artifacts ───────────────────────────────────────────────────
    save_plots(all_rewards, all_kpis, save_dir)

    final_metrics: Dict = {
        "train": {
            "n_iterations":       cfg.n_iterations,
            "last_ep_reward_mean": all_rewards[-1] if all_rewards else None,
            "checkpoint":         ckpt_dir_path,
        }
    }
    if test_result is not None:
        final_metrics["test"] = test_result

    (save_dir / "latest_metrics.json").write_text(json.dumps(final_metrics, indent=2))

    # Backup
    backup_dir = _make_backup_dir(cfg)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in save_dir.glob("*.json"):
        shutil.copy2(f, backup_dir / f.name)
    for f in save_dir.glob("*.png"):
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
    parser.add_argument("--climate",      default="TX", choices=["VT", "TX"])
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
    parser.add_argument("--wandb_name",    default="rllib-sac")
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
