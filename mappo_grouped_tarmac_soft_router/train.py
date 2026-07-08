"""
Grouped MAPPO with hybrid TarMAC-style global communication and soft actor routing.

Actor path:
  obs(group_k) -> encoder_k -> local/TarMAC hybrid communication
               -> state-conditioned soft mixture over grouped action heads
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from annex96_reporting import collect_building_temperature_timeseries
import torch.nn as nn

REPO_DIR = Path(__file__).resolve().parent.parent
ONPOLICY = REPO_DIR / "on-policy-main"

for _p in [str(REPO_DIR), str(ONPOLICY)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO  # noqa: E402
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy  # noqa: E402

from mappo.utils import extract_episode_kpis, get_soc_stats, resolve_reference_baseline_series  # noqa: E402
from mappo_grouped_comm.buffer import GroupedSharedReplayBuffer  # noqa: E402
from mappo_grouped_comm.train import (  # noqa: E402
    _CLIMATE_DEFAULTS,
    _MONTH_ENDS,
    _MONTH_NAMES,
    _MONTH_STARTS,
    _filter_wandb,
    _run_daily_pipeline,
    build_mappo_args,
    compute_primary_metric_tables,
    export_test_metrics,
    save_plots,
    save_run_config,
    seed_everything,
)
from mappo_grouped_comm_v2.train import (  # noqa: E402
    _compute_advantages,
    _joint_actor_generator,
    load_checkpoint,
    save_checkpoint,
)
from mappo_grouped_tarmac_soft_router.cluster import run_clustering  # noqa: E402
from mappo_grouped_tarmac_soft_router.env import CityLearnMAPPOEnv  # noqa: E402
from mappo_grouped_tarmac_soft_router.hybrid_tarmac_comm import (  # noqa: E402
    HybridTarMACCommunicationModule,
)
from mappo_grouped_tarmac_soft_router.soft_router_actor import (  # noqa: E402
    SoftRouterGlobalCommActorController,
)

try:
    import wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    print("[warn] wandb not installed - W&B logging disabled.")


@dataclass
class Config:
    climate: str = "VT"
    n_buildings: int = 25

    group_k_candidates: List[int] = field(default_factory=lambda: [4, 5])
    cluster_seed: int = 0
    cluster_retries: int = 10
    cluster_artifact_dir: Optional[str] = None
    grouping_method: str = "agglomerative"
    grouping_feature_set: str = "control_profile"
    grouping_feature_month: Optional[int] = None
    grouping_feature_columns: Optional[List[str]] = field(
        default_factory=lambda: [
            "bes_capacity_kwh",
            "hvac_total_kw",
            "heating_mean",
            "nsl_mean",
            "comfort_lower_excess_mean",
        ]
    )

    hidden_size: int = 256
    layer_N: int = 2

    lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epoch: int = 10
    num_mini_batch: int = 4
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 10.0

    n_episodes: int = 100
    train_month: Optional[int] = None
    test_month: Optional[int] = None

    do_test: bool = True
    test_only: bool = False
    checkpoint_dir: Optional[str] = None
    test_save_dir: Optional[str] = None

    seed: int = 42
    wandb_project: str = "annex96-ce1"
    wandb_name: str = "mappo-grouped-tarmac-soft-router"
    save_dir: str = "results/mappo_grouped_tarmac_soft_router"
    backup_dir: Optional[str] = None

    comm_method: str = "tarmac_hybrid"
    comm_hidden_dim: int = 64
    comm_rounds: int = 1
    comm_use_residual: bool = True
    comm_dropout: float = 0.0
    comm_scope: str = "global"
    comm_key_dim: int = 32
    comm_value_dim: int = 64
    comm_fusion_mode: str = "linear"

    router_hidden_dim: int = 64
    router_temperature: float = 0.7
    router_prior_start: float = 0.0
    router_prior_end: float = 0.7
    router_warmup_episodes: int = 100
    router_entropy_scale: float = 0.02
    router_use_capacity_features: bool = True
    router_eval_hard: bool = True


def _build_grouped_components(
    cfg: Config,
    env: CityLearnMAPPOEnv,
    group_assignments: np.ndarray,
    device: torch.device,
    router_obs_indices: Optional[Dict[str, int]] = None,
    building_capacity_norm: Optional[np.ndarray] = None,
    building_hvac_norm: Optional[np.ndarray] = None,
) -> Tuple[
    argparse.Namespace,
    List[R_MAPPOPolicy],
    List[R_MAPPO],
    List[GroupedSharedReplayBuffer],
    List[np.ndarray],
    SoftRouterGlobalCommActorController,
    torch.optim.Optimizer,
]:
    K = int(group_assignments.max()) + 1
    group_indices = [np.where(group_assignments == k)[0] for k in range(K)]
    group_sizes = [int(len(idx)) for idx in group_indices]
    episode_length = env._base_env.episode_time_steps

    mappo_args = build_mappo_args(cfg, env.obs_dim, env.act_dim, env.cent_obs_dim)
    mappo_args.episode_length = episode_length
    mappo_args.n_rollout_threads = 1

    policies: List[R_MAPPOPolicy] = []
    for _ in range(K):
        policy = R_MAPPOPolicy(
            args=mappo_args,
            obs_space=env.obs_space,
            cent_obs_space=env.cent_obs_space,
            act_space=env.act_space,
            device=device,
        )
        policies.append(policy)

    shared_critic = policies[0].critic
    shared_critic_opt = policies[0].critic_optimizer
    for k in range(1, K):
        policies[k].critic = shared_critic
        policies[k].critic_optimizer = shared_critic_opt

    global_comm = HybridTarMACCommunicationModule(
        hidden_dim=cfg.hidden_size,
        comm_hidden_dim=cfg.comm_hidden_dim,
        key_dim=cfg.comm_key_dim,
        value_dim=cfg.comm_value_dim,
        comm_rounds=cfg.comm_rounds,
        use_residual=cfg.comm_use_residual,
        dropout=cfg.comm_dropout,
        fusion_mode=cfg.comm_fusion_mode,
    ).to(device)
    controller = SoftRouterGlobalCommActorController(
        actors=[policy.actor for policy in policies],
        comm=global_comm,
        group_indices=group_indices,
        router_hidden_dim=cfg.router_hidden_dim,
        router_temperature=cfg.router_temperature,
        router_entropy_scale=cfg.router_entropy_scale,
        router_alpha=cfg.router_prior_start,
        router_obs_indices=router_obs_indices,
        building_capacity_norm=building_capacity_norm,
        building_hvac_norm=building_hvac_norm,
        use_capacity_router_features=cfg.router_use_capacity_features,
        deterministic_hard_routing=cfg.router_eval_hard,
    ).to(device)
    actor_optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=mappo_args.lr,
        eps=mappo_args.opti_eps,
        weight_decay=mappo_args.weight_decay,
    )

    trainers: List[R_MAPPO] = [
        R_MAPPO(args=mappo_args, policy=policies[k], device=device)
        for k in range(K)
    ]
    if trainers[0].value_normalizer is not None:
        shared_vn = trainers[0].value_normalizer
        for k in range(1, K):
            trainers[k].value_normalizer = shared_vn

    buffers: List[GroupedSharedReplayBuffer] = []
    for k in range(K):
        buf = GroupedSharedReplayBuffer(
            args=mappo_args,
            num_agents=group_sizes[k],
            obs_space=env.obs_space,
            cent_obs_space=env.cent_obs_space,
            act_space=env.act_space,
        )
        buf.action_log_probs = np.zeros(
            (episode_length, 1, group_sizes[k], 1),
            dtype=np.float32,
        )
        buffers.append(buf)

    return mappo_args, policies, trainers, buffers, group_indices, controller, actor_optimizer


def _router_alpha_for_episode(cfg: Config, episode: int) -> float:
    if cfg.router_warmup_episodes <= 0:
        return float(cfg.router_prior_end)
    progress = min(max((episode - 1) / float(cfg.router_warmup_episodes), 0.0), 1.0)
    return float(cfg.router_prior_start + progress * (cfg.router_prior_end - cfg.router_prior_start))


def _get_observation_indices(env: CityLearnMAPPOEnv) -> Dict[str, int]:
    names = getattr(env._env, "observation_names", None)
    if names is None:
        return {}
    try:
        first_names = list(names[0])
    except Exception:
        return {}
    wanted = ["electrical_storage_soc", "non_shiftable_load", "heating_demand"]
    return {name: first_names.index(name) for name in wanted if name in first_names}


def _read_router_static_features(cluster_dir: Path, n_buildings: int) -> Tuple[np.ndarray, np.ndarray]:
    assignment_path = Path(cluster_dir) / "building_cluster_assignment.csv"
    capacity = np.zeros(n_buildings, dtype=np.float32)
    hvac = np.zeros(n_buildings, dtype=np.float32)
    if not assignment_path.exists():
        print(f"[warn] router static feature file not found: {assignment_path}")
        return capacity, hvac

    with assignment_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                i = int(row["building_idx"])
            except Exception:
                continue
            if 0 <= i < n_buildings:
                capacity[i] = float(row.get("bes_capacity_kwh") or 0.0)
                hvac[i] = float(row.get("hvac_total_kw") or 0.0)

    cap_scale = float(np.nanmax(capacity)) if np.isfinite(capacity).any() else 0.0
    hvac_scale = float(np.nanmax(hvac)) if np.isfinite(hvac).any() else 0.0
    if cap_scale > 1e-8:
        capacity = capacity / cap_scale
    if hvac_scale > 1e-8:
        hvac = hvac / hvac_scale
    return capacity.astype(np.float32), hvac.astype(np.float32)


def _train_soft_router_actor(
    cfg: Config,
    controller: SoftRouterGlobalCommActorController,
    actor_optimizer: torch.optim.Optimizer,
    trainers: List[R_MAPPO],
    buffers: List[GroupedSharedReplayBuffer],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    controller.train()
    device = next(controller.parameters()).device
    advantages_by_group = [
        _compute_advantages(trainer, buffer)
        for trainer, buffer in zip(trainers, buffers)
    ]

    train_info: Dict[str, float] = {
        "policy_loss": 0.0,
        "dist_entropy": 0.0,
        "actor_grad_norm": 0.0,
        "ratio": 0.0,
        "router_alpha": 0.0,
        "gate_entropy": 0.0,
        "gate_max": 0.0,
        "gate_static_weight": 0.0,
    }
    for k in range(len(buffers)):
        train_info[f"gate_expert_{k}_mean"] = 0.0

    per_group_info = [
        {"policy_loss": 0.0, "dist_entropy": 0.0, "ratio": 0.0}
        for _ in buffers
    ]

    for _ in range(cfg.ppo_epoch):
        for group_batches in _joint_actor_generator(buffers, advantages_by_group, cfg.num_mini_batch):
            tensor_batches: List[Dict[str, torch.Tensor]] = []
            for batch in group_batches:
                tensor_batch: Dict[str, torch.Tensor] = {
                    "obs": torch.as_tensor(batch["obs"], dtype=torch.float32, device=device),
                    "rnn_states": torch.as_tensor(batch["rnn_states"], dtype=torch.float32, device=device),
                    "action": torch.as_tensor(batch["action"], dtype=torch.float32, device=device),
                    "masks": torch.as_tensor(batch["masks"], dtype=torch.float32, device=device),
                    "active_masks": torch.as_tensor(batch["active_masks"], dtype=torch.float32, device=device),
                    "old_log_probs": torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=device),
                    "advantages": torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device),
                    "available_actions": None,
                }
                if batch["available_actions"] is not None:
                    tensor_batch["available_actions"] = torch.as_tensor(
                        batch["available_actions"], dtype=torch.float32, device=device
                    )
                tensor_batches.append(tensor_batch)

            current_log_probs, entropies = controller.evaluate_actions(tensor_batches)

            policy_losses: List[torch.Tensor] = []
            entropy_terms: List[torch.Tensor] = []
            ratios: List[torch.Tensor] = []
            for k, (curr_logp, entropy_k, batch) in enumerate(zip(current_log_probs, entropies, tensor_batches)):
                imp_weights = torch.exp(curr_logp - batch["old_log_probs"])
                surr1 = imp_weights * batch["advantages"]
                surr2 = torch.clamp(
                    imp_weights,
                    1.0 - cfg.clip_param,
                    1.0 + cfg.clip_param,
                ) * batch["advantages"]
                policy_action_loss = (
                    -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * batch["active_masks"]
                ).sum() / batch["active_masks"].sum()

                policy_losses.append(policy_action_loss)
                entropy_terms.append(entropy_k)
                ratios.append(imp_weights.mean())

                per_group_info[k]["policy_loss"] += float(policy_action_loss.item())
                per_group_info[k]["dist_entropy"] += float(entropy_k.item())
                per_group_info[k]["ratio"] += float(imp_weights.mean().item())

            total_policy_loss = torch.stack(policy_losses).mean()
            total_entropy = torch.stack(entropy_terms).mean()
            total_ratio = torch.stack(ratios).mean()

            actor_optimizer.zero_grad()
            (total_policy_loss - total_entropy * cfg.entropy_coef).backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(controller.parameters(), cfg.max_grad_norm)
            actor_optimizer.step()

            train_info["policy_loss"] += float(total_policy_loss.item())
            train_info["dist_entropy"] += float(total_entropy.item())
            train_info["actor_grad_norm"] += float(actor_grad_norm)
            train_info["ratio"] += float(total_ratio.item())
            for key, value in controller.last_gate_stats.items():
                if key in train_info:
                    train_info[key] += float(value)

    num_updates = cfg.ppo_epoch * cfg.num_mini_batch
    for key in train_info:
        train_info[key] /= num_updates
    for info in per_group_info:
        for key in info:
            info[key] /= num_updates

    return train_info, per_group_info


def _apply_checkpoint_model_config(cfg: Config, ckpt_meta: Dict[str, Any]) -> None:
    saved_cfg = ckpt_meta.get("cfg")
    if not isinstance(saved_cfg, dict):
        return
    for field_name in [
        "hidden_size",
        "layer_N",
        "lr",
        "critic_lr",
        "comm_method",
        "comm_hidden_dim",
        "comm_rounds",
        "comm_use_residual",
        "comm_dropout",
        "comm_key_dim",
        "comm_value_dim",
        "comm_fusion_mode",
        "grouping_method",
        "grouping_feature_set",
        "grouping_feature_month",
        "grouping_feature_columns",
        "router_hidden_dim",
        "router_temperature",
        "router_prior_start",
        "router_prior_end",
        "router_warmup_episodes",
        "router_entropy_scale",
        "router_use_capacity_features",
        "router_eval_hard",
    ]:
        if field_name in saved_cfg:
            setattr(cfg, field_name, saved_cfg[field_name])


def evaluate_on_test(
    controller: SoftRouterGlobalCommActorController,
    cfg: Config,
    test_start: int,
    test_end: int,
    use_wandb: bool,
    iteration: Optional[int] = None,
    log_per_step: bool = False,
) -> Dict[str, Any]:
    month_name = _MONTH_NAMES.get(cfg.test_month, str(cfg.test_month))
    print("\n" + "=" * 65)
    print(
        f"TEST EVALUATION (grouped TarMAC soft-router) | {cfg.climate} | {month_name} "
        f"(steps {test_start}-{test_end})"
    )
    print("=" * 65)

    test_env = CityLearnMAPPOEnv(
        climate=cfg.climate,
        n_buildings=cfg.n_buildings,
        start_step=test_start,
        end_step=test_end,
    )

    n_agents = test_env.n_agents
    recurrent_N = 1
    hidden_size = controller.hidden_size

    obs, _ = test_env.reset(seed=cfg.seed)
    rnn_a = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
    masks = np.ones((n_agents, 1), dtype=np.float32)

    step_portfolio_rewards: List[float] = []
    step_portfolio_loads: List[float] = []
    per_building_rewards: Dict[str, float] = {f"building_{i}": 0.0 for i in range(n_agents)}
    cumulative_reward = 0.0
    test_step = 0

    controller.eval()
    done = False
    while not done:
        with torch.no_grad():
            actions_t, _, rnn_a_new_t = controller.act(
                obs=obs,
                rnn_states=rnn_a,
                masks=masks,
                deterministic=True,
            )
        all_actions = np.clip(actions_t.cpu().numpy(), -1.0, 1.0)
        rnn_a = rnn_a_new_t.cpu().numpy().reshape(n_agents, recurrent_N, hidden_size)

        next_obs, _, rewards, done, _ = test_env.step(all_actions)
        step_rew = float(rewards.sum())
        cumulative_reward += step_rew
        step_portfolio_rewards.append(step_rew)
        for i in range(n_agents):
            per_building_rewards[f"building_{i}"] += float(rewards[i])

        try:
            net_load = sum(
                float(b.net_electricity_consumption[-1])
                for b in test_env.base_env.buildings
            )
            step_portfolio_loads.append(net_load)
        except Exception:
            step_portfolio_loads.append(float("nan"))

        if log_per_step and use_wandb:
            wandb.log(
                _filter_wandb(
                    {
                        "test_step": test_step,
                        "test/step_reward": step_rew,
                        "test/cumulative_reward": cumulative_reward,
                    }
                )
            )

        obs = next_obs
        masks = np.zeros((n_agents, 1), dtype=np.float32) if done else np.ones((n_agents, 1), dtype=np.float32)
        test_step += 1

    test_kpis = extract_episode_kpis(test_env.base_env)
    test_soc = get_soc_stats(test_env.base_env)
    primary_metrics, daily_primary_df, comfort_building_df = compute_primary_metric_tables(
        test_env.base_env,
        cfg.test_month,
    )
    portfolio_reward = float(sum(per_building_rewards.values()))

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

    result: Dict[str, Any] = {
        "test/portfolio/reward_sum": portfolio_reward,
        "test/portfolio/reward_mean": float(np.mean(list(per_building_rewards.values()))),
        "test/step_reward_mean": float(np.mean(step_portfolio_rewards)) if step_portfolio_rewards else 0.0,
        "test/step_reward_std": float(np.std(step_portfolio_rewards)) if step_portfolio_rewards else 0.0,
        "test/portfolio/load_mean": float(np.nanmean(step_portfolio_loads)) if step_portfolio_loads else 0.0,
        "test/portfolio/load_std": float(np.nanstd(step_portfolio_loads)) if step_portfolio_loads else 0.0,
        **{f"test/{k}": v for k, v in primary_metrics.items()},
        **{f"test/{k}": v for k, v in test_kpis.items()},
        **{f"test/{k}": v for k, v in test_soc.items()},
        "_step_portfolio_loads": step_portfolio_loads,
        "_step_portfolio_loads_baseline": resolve_reference_baseline_series(test_env.base_env)[: len(step_portfolio_loads)].tolist(),
        "_daily_primary_metrics": daily_primary_df.to_dict(orient="records"),
        "_building_comfort_metrics": comfort_building_df.to_dict(orient="records"),
        "_building_temperature_timeseries": collect_building_temperature_timeseries(test_env.base_env).to_dict(orient="records"),
    }
    for bid, reward in per_building_rewards.items():
        result[f"test/{bid}/reward"] = reward
    if iteration is not None:
        result["episode"] = iteration

    if use_wandb and not log_per_step:
        wandb.log(_filter_wandb(result), step=iteration)
    elif use_wandb and log_per_step:
        summary = {k: v for k, v in result.items() if not k.startswith("test/building_")}
        summary["test_step"] = test_step - 1
        wandb.log(_filter_wandb(summary))

    return result


def train(cfg: Config) -> None:
    defaults = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    train_month = cfg.train_month or defaults.get("train_month")
    test_month = cfg.test_month or defaults.get("test_month")
    if train_month is None or train_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid train_month={train_month}. Must be 1-12.")
    if cfg.do_test and (test_month is None or test_month not in _MONTH_STARTS):
        raise ValueError(f"Invalid test_month={test_month}. Must be 1-12.")

    train_start = _MONTH_STARTS[train_month]
    train_end = _MONTH_ENDS[train_month]
    test_start = _MONTH_STARTS[test_month] if test_month else None
    test_end = _MONTH_ENDS[test_month] if test_month else None
    cfg.train_month = train_month
    cfg.test_month = test_month
    if cfg.grouping_feature_month is None:
        cfg.grouping_feature_month = train_month

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = Path(cfg.cluster_artifact_dir).resolve() if cfg.cluster_artifact_dir else save_dir
    cluster_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, save_dir)

    group_assignments, cluster_result = run_clustering(
        climate=cfg.climate,
        save_dir=cluster_dir,
        n_buildings=cfg.n_buildings,
        k_candidates=cfg.group_k_candidates,
        cluster_seed=cfg.cluster_seed,
        retries=cfg.cluster_retries,
        repo_dir=REPO_DIR,
        grouping_method=cfg.grouping_method,
        grouping_feature_set=cfg.grouping_feature_set,
        grouping_feature_month=cfg.grouping_feature_month,
        grouping_feature_columns=cfg.grouping_feature_columns,
    )
    K = int(group_assignments.max()) + 1

    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = vars(cfg).copy()
        cfg_dict.update(
            {
                "train_start_step": train_start,
                "train_end_step": train_end,
                "test_start_step": test_start,
                "test_end_step": test_end,
                "algorithm": "mappo_grouped_tarmac_soft_router",
                "comm_scope": "global",
                "n_groups": K,
                "group_sizes": cluster_result["sizes"],
                "router_type": "state_conditioned_soft_mixture",
                "grouping_method": cfg.grouping_method,
                "grouping_feature_set": cfg.grouping_feature_set,
                "grouping_feature_columns": cfg.grouping_feature_columns,
            }
        )
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg_dict)
        wandb.define_metric("episode")
        wandb.define_metric("train/*", step_metric="episode")
        wandb.define_metric("test/*", step_metric="episode")

    train_env = CityLearnMAPPOEnv(
        climate=cfg.climate,
        n_buildings=cfg.n_buildings,
        start_step=train_start,
        end_step=train_end,
    )
    router_obs_indices = _get_observation_indices(train_env)
    capacity_norm, hvac_norm = _read_router_static_features(cluster_dir, cfg.n_buildings)
    (
        mappo_args,
        policies,
        trainers,
        buffers,
        group_indices,
        controller,
        actor_optimizer,
    ) = _build_grouped_components(
        cfg,
        train_env,
        group_assignments,
        device,
        router_obs_indices=router_obs_indices,
        building_capacity_norm=capacity_norm,
        building_hvac_norm=hvac_norm,
    )

    episode_length = train_env._base_env.episode_time_steps
    n_agents = train_env.n_agents
    act_dim = train_env.act_dim
    cent_obs_dim = train_env.cent_obs_dim
    recurrent_N = mappo_args.recurrent_N
    hidden_size = mappo_args.hidden_size
    group_sizes = [len(idx) for idx in group_indices]

    print("\nGrouped MAPPO TarMAC soft-router components built:")
    print(f"  n_agents={n_agents} K={K} group_sizes={group_sizes}")
    print(
        f"  comm_method={cfg.comm_method} comm_scope=global "
        f"comm_rounds={cfg.comm_rounds} fusion_mode={cfg.comm_fusion_mode}"
    )
    print(
        f"  grouping_method={cfg.grouping_method} "
        f"grouping_feature_set={cfg.grouping_feature_set} "
        f"grouping_feature_columns={cfg.grouping_feature_columns}"
    )
    print(
        f"  router_hidden_dim={cfg.router_hidden_dim} "
        f"router_alpha={cfg.router_prior_start}->{cfg.router_prior_end} "
        f"warmup={cfg.router_warmup_episodes} "
        f"capacity_features={cfg.router_use_capacity_features} "
        f"eval_hard={cfg.router_eval_hard} "
        f"obs_indices={router_obs_indices}"
    )

    if use_wandb:
        wandb.log(
            {
                "train/comm/method": cfg.comm_method,
                "train/comm/scope": cfg.comm_scope,
                "train/comm/rounds": cfg.comm_rounds,
                "train/comm/hidden_dim": cfg.comm_hidden_dim,
                "train/comm/use_residual": int(cfg.comm_use_residual),
                "train/comm/fusion_mode": cfg.comm_fusion_mode,
                "train/grouping/method": cfg.grouping_method,
                "train/grouping/feature_set": cfg.grouping_feature_set,
                "train/grouping/feature_columns": (
                    ",".join(cfg.grouping_feature_columns) if cfg.grouping_feature_columns else ""
                ),
                "train/router/hidden_dim": cfg.router_hidden_dim,
                "train/router/temperature": cfg.router_temperature,
                "train/router/prior_start": cfg.router_prior_start,
                "train/router/prior_end": cfg.router_prior_end,
                "train/router/warmup_episodes": cfg.router_warmup_episodes,
                "train/router/entropy_scale": cfg.router_entropy_scale,
                "train/router/use_capacity_features": int(cfg.router_use_capacity_features),
                "train/router/eval_hard": int(cfg.router_eval_hard),
            },
            step=0,
        )

    all_rewards: List[float] = []
    all_primary_metrics: List[Dict[str, Any]] = []
    save_every = max(cfg.n_episodes // 10, 1)

    for episode in range(1, cfg.n_episodes + 1):
        controller.set_router_alpha(_router_alpha_for_episode(cfg, episode))
        obs, share_obs = train_env.reset(seed=cfg.seed + episode)

        for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
            n_k = group_sizes[k]
            buf.step = 0
            buf.share_obs[0] = share_obs[idx_k][np.newaxis]
            buf.obs[0] = obs[idx_k][np.newaxis]
            buf.masks[0] = np.ones((1, n_k, 1), dtype=np.float32)
            buf.rnn_states[0] = np.zeros((1, n_k, recurrent_N, hidden_size), dtype=np.float32)
            buf.rnn_states_critic[0] = np.zeros_like(buf.rnn_states[0])

        rnn_a = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
        rnn_c = np.zeros((n_agents, recurrent_N, hidden_size), dtype=np.float32)
        masks = np.ones((n_agents, 1), dtype=np.float32)
        episode_reward = 0.0

        controller.eval()
        for trainer in trainers:
            trainer.prep_rollout()

        for _step in range(episode_length):
            all_values = np.zeros((n_agents, 1), dtype=np.float32)

            with torch.no_grad():
                actions_t, logp_t, rnn_a_new_t = controller.act(
                    obs=obs,
                    rnn_states=rnn_a,
                    masks=masks,
                    deterministic=False,
                )
                rnn_a_new = rnn_a_new_t.cpu().numpy().reshape(n_agents, recurrent_N, hidden_size)

                for k, idx_k in enumerate(group_indices):
                    vals_k = policies[k].get_values(
                        share_obs[idx_k],
                        rnn_c[idx_k],
                        masks[idx_k],
                    )
                    all_values[idx_k] = vals_k.cpu().numpy()

            all_actions = actions_t.cpu().numpy()
            all_log_probs = logp_t.cpu().numpy()
            actions_clipped = np.clip(all_actions, -1.0, 1.0)
            next_obs, next_share_obs, rewards, done, _ = train_env.step(actions_clipped)
            episode_reward += float(rewards.sum())

            masks_np = np.zeros((n_agents, 1), dtype=np.float32) if done else np.ones((n_agents, 1), dtype=np.float32)
            rnn_c_new = rnn_c.copy()

            for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
                n_k = group_sizes[k]
                buf.insert(
                    share_obs=next_share_obs[idx_k][np.newaxis],
                    obs=next_obs[idx_k][np.newaxis],
                    rnn_states_actor=rnn_a_new[idx_k].reshape(1, n_k, recurrent_N, hidden_size),
                    rnn_states_critic=rnn_c_new[idx_k].reshape(1, n_k, recurrent_N, hidden_size),
                    actions=all_actions[idx_k].reshape(1, n_k, act_dim),
                    action_log_probs=all_log_probs[idx_k].reshape(1, n_k, 1),
                    value_preds=all_values[idx_k].reshape(1, n_k, 1),
                    rewards=rewards[idx_k].reshape(1, n_k, 1),
                    masks=masks_np[idx_k].reshape(1, n_k, 1),
                )

            obs = next_obs
            share_obs = next_share_obs
            rnn_a = rnn_a_new
            rnn_c = rnn_c_new
            masks = masks_np

            if done:
                break

        with torch.no_grad():
            for k, (buf, idx_k) in enumerate(zip(buffers, group_indices)):
                n_k = group_sizes[k]
                share_last = buf.share_obs[-1].reshape(n_k, cent_obs_dim)
                masks_last = buf.masks[-1].reshape(n_k, 1)
                rnn_c_last = buf.rnn_states_critic[-1].reshape(n_k, recurrent_N, hidden_size)
                next_values = policies[k].get_values(share_last, rnn_c_last, masks_last)
                buf.compute_returns(
                    next_values.cpu().numpy().reshape(1, n_k, 1),
                    value_normalizer=trainers[0].value_normalizer,
                )

        actor_train_info, actor_group_infos = _train_soft_router_actor(
            cfg=cfg,
            controller=controller,
            actor_optimizer=actor_optimizer,
            trainers=trainers,
            buffers=buffers,
        )

        critic_group_infos: List[Dict[str, Any]] = []
        for k in range(K):
            trainers[k].prep_training()
            info_k = trainers[k].train(buffers[k], update_actor=False)
            critic_group_infos.append(info_k)

        for k in range(K):
            trainers[k].prep_rollout()
            buffers[k].after_update()

        def _avg_critic(key: str) -> float:
            vals = [float(info.get(key, 0.0)) for info in critic_group_infos]
            return float(np.mean(vals)) if vals else 0.0

        train_info = {
            "value_loss": _avg_critic("value_loss"),
            "policy_loss": actor_train_info["policy_loss"],
            "dist_entropy": actor_train_info["dist_entropy"],
            "actor_grad_norm": actor_train_info["actor_grad_norm"],
            "critic_grad_norm": _avg_critic("critic_grad_norm"),
            "ratio": actor_train_info["ratio"],
            "router_alpha": actor_train_info["router_alpha"],
            "gate_entropy": actor_train_info["gate_entropy"],
            "gate_max": actor_train_info["gate_max"],
            "gate_static_weight": actor_train_info["gate_static_weight"],
        }

        all_rewards.append(episode_reward)
        kpis = extract_episode_kpis(train_env.base_env)
        primary_metrics, _daily_primary_df, _comfort_building_df = compute_primary_metric_tables(
            train_env.base_env,
            cfg.train_month,
        )
        all_primary_metrics.append(primary_metrics)

        if episode % 10 == 0 or episode == 1:
            nmbe = primary_metrics.get("primary/load_tracking/nmbe_pct")
            cv_rmse = primary_metrics.get("primary/load_tracking/cv_rmse_pct")
            print(
                f"[train] Ep {episode:4d} | ep_rew {episode_reward:9.2f} | "
                f"v_loss {train_info['value_loss']:.4f} | "
                f"p_loss {train_info['policy_loss']:.4f} | "
                f"entropy {train_info['dist_entropy']:.4f} | "
                f"gate_static {train_info['gate_static_weight']:.3f} | "
                f"NMBE {float(nmbe):.3f}% | CV-RMSE {float(cv_rmse):.3f}%"
            )

        if use_wandb:
            log: Dict[str, Any] = {
                "episode": episode,
                "train/portfolio/reward_sum": episode_reward,
                "train/loss/value_loss": train_info["value_loss"],
                "train/loss/policy_loss": train_info["policy_loss"],
                "train/loss/dist_entropy": train_info["dist_entropy"],
                "train/loss/actor_grad_norm": train_info["actor_grad_norm"],
                "train/loss/critic_grad_norm": train_info["critic_grad_norm"],
                "train/loss/ratio": train_info["ratio"],
                "train/router/alpha": train_info["router_alpha"],
                "train/router/gate_entropy": train_info["gate_entropy"],
                "train/router/gate_max": train_info["gate_max"],
                "train/router/gate_static_weight": train_info["gate_static_weight"],
            }
            for k in range(K):
                key = f"gate_expert_{k}_mean"
                log[f"train/router/expert_{k}_mean"] = actor_train_info.get(key, 0.0)
            for k, info_k in enumerate(actor_group_infos):
                log[f"train/group_{k}/policy_loss"] = info_k["policy_loss"]
                log[f"train/group_{k}/dist_entropy"] = info_k["dist_entropy"]
            for kk, value in primary_metrics.items():
                if value is not None:
                    log[f"train/{kk}"] = value
            for kk, value in kpis.items():
                if value is not None:
                    log[f"train/{kk}"] = value
            wandb.log(_filter_wandb(log), step=episode)

        if episode % save_every == 0 or episode == cfg.n_episodes:
            save_checkpoint(
                policies=policies,
                controller=controller,
                actor_optimizer=actor_optimizer,
                group_assignments=group_assignments,
                mappo_args=mappo_args,
                save_dir=save_dir,
                episode=episode,
                cfg=cfg,
            )
            save_plots(all_rewards, all_primary_metrics, save_dir)

    test_result: Optional[Dict[str, Any]] = None
    if cfg.do_test and test_start is not None:
        test_result = evaluate_on_test(
            controller=controller,
            cfg=cfg,
            test_start=test_start,
            test_end=test_end,
            use_wandb=use_wandb,
            iteration=cfg.n_episodes,
        )
        export_test_metrics(test_result, cfg, save_dir, test_start, test_end, test_month)
        _run_daily_pipeline(test_result, cfg, save_dir, use_wandb)

    save_plots(all_rewards, all_primary_metrics, save_dir)

    final_metrics: Dict[str, Any] = {
        "algorithm": "mappo_grouped_tarmac_soft_router",
        "comm_method": cfg.comm_method,
        "comm_scope": cfg.comm_scope,
        "comm_fusion_mode": cfg.comm_fusion_mode,
        "grouping_method": cfg.grouping_method,
        "grouping_feature_set": cfg.grouping_feature_set,
        "grouping_feature_month": cfg.grouping_feature_month,
        "grouping_feature_columns": cfg.grouping_feature_columns,
        "router": {
            "type": "state_conditioned_soft_mixture",
            "hidden_dim": cfg.router_hidden_dim,
            "temperature": cfg.router_temperature,
            "prior_start": cfg.router_prior_start,
            "prior_end": cfg.router_prior_end,
            "warmup_episodes": cfg.router_warmup_episodes,
            "entropy_scale": cfg.router_entropy_scale,
            "use_capacity_features": cfg.router_use_capacity_features,
            "eval_hard": cfg.router_eval_hard,
        },
        "n_groups": K,
        "group_sizes": group_sizes,
        "train": {
            "n_episodes": cfg.n_episodes,
            "last_ep_reward_sum": all_rewards[-1] if all_rewards else None,
            "train_month": train_month,
            "train_steps": f"{train_start}-{train_end}",
            **(all_primary_metrics[-1] if all_primary_metrics else {}),
        },
    }
    if test_result is not None:
        public_test = {k: v for k, v in test_result.items() if not k.startswith("_")}
        final_metrics["test"] = {
            "test_month": test_month,
            "test_steps": f"{test_start}-{test_end}",
            **public_test,
        }

    (save_dir / "latest_metrics.json").write_text(json.dumps(final_metrics, indent=2))
    print(f"  [metrics] latest_metrics.json -> {save_dir}/latest_metrics.json")

    backup_dir = (
        Path(cfg.backup_dir).resolve()
        if cfg.backup_dir
        else save_dir / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in list(save_dir.glob("*.json")) + list(save_dir.glob("*.png")) + list(save_dir.glob("*.csv")):
        shutil.copy2(f, backup_dir / f.name)

    if use_wandb:
        wandb.finish()

    print("\nGrouped MAPPO TarMAC soft-router training complete.")
    print(f"  Artifacts: {save_dir}/")


def run_test_only(cfg: Config) -> Dict[str, Any]:
    defaults = _CLIMATE_DEFAULTS.get(cfg.climate, {})
    test_month = cfg.test_month or defaults.get("test_month")
    train_month = cfg.train_month or defaults.get("train_month")
    if test_month is None or test_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid test_month={test_month}. Must be 1-12.")

    cfg.test_month = test_month
    cfg.train_month = train_month
    test_start = _MONTH_STARTS[test_month]
    test_end = _MONTH_ENDS[test_month]
    train_start = _MONTH_STARTS[train_month]
    train_end = _MONTH_ENDS[train_month]

    default_ckpt = Path(cfg.save_dir).resolve() / "checkpoint.pt"
    ckpt_dir = Path(cfg.checkpoint_dir).resolve() if cfg.checkpoint_dir else None
    ckpt_path = (
        ckpt_dir / "checkpoint.pt" if ckpt_dir and ckpt_dir.is_dir()
        else ckpt_dir if ckpt_dir and ckpt_dir.is_file()
        else default_ckpt
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_meta = torch.load(ckpt_path, map_location=device, weights_only=False)
    _apply_checkpoint_model_config(cfg, ckpt_meta)
    group_assignments = np.array(ckpt_meta.get("group_assignments", list(range(cfg.n_buildings))))

    probe_env = CityLearnMAPPOEnv(
        climate=cfg.climate,
        n_buildings=cfg.n_buildings,
        start_step=train_start,
        end_step=train_end,
    )
    router_obs_indices = _get_observation_indices(probe_env)
    router_feature_dir = (
        Path(cfg.cluster_artifact_dir).resolve()
        if cfg.cluster_artifact_dir
        else ckpt_path.parent
    )
    capacity_norm, hvac_norm = _read_router_static_features(router_feature_dir, cfg.n_buildings)
    (
        _mappo_args,
        policies,
        _trainers,
        _buffers,
        _group_indices,
        controller,
        actor_optimizer,
    ) = _build_grouped_components(
        cfg,
        probe_env,
        group_assignments,
        device,
        router_obs_indices=router_obs_indices,
        building_capacity_norm=capacity_norm,
        building_hvac_norm=hvac_norm,
    )
    load_checkpoint(policies, controller, actor_optimizer, ckpt_path, device)

    use_wandb = _WANDB_OK
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.wandb_name}-test-only",
            config=vars(cfg),
        )
        wandb.define_metric("test_step")
        wandb.define_metric("test/*", step_metric="test_step")

    test_result = evaluate_on_test(
        controller=controller,
        cfg=cfg,
        test_start=test_start,
        test_end=test_end,
        use_wandb=use_wandb,
        iteration=None,
        log_per_step=True,
    )

    test_save_dir = Path(cfg.test_save_dir or str(ckpt_path.parent)).resolve()
    test_save_dir.mkdir(parents=True, exist_ok=True)
    export_test_metrics(test_result, cfg, test_save_dir, test_start, test_end, test_month)
    _run_daily_pipeline(test_result, cfg, test_save_dir, use_wandb)

    if use_wandb:
        wandb.finish()

    return test_result


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Grouped MAPPO with hybrid TarMAC-style global communication and soft actor routing."
    )
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)

    parser.add_argument("--group_k_candidates", type=int, nargs="+", default=[4, 5], metavar="K")
    parser.add_argument("--cluster_seed", type=int, default=0)
    parser.add_argument("--cluster_retries", type=int, default=10)
    parser.add_argument("--cluster_artifact_dir", default=None)
    parser.add_argument(
        "--grouping_method",
        default="agglomerative",
        choices=["kmeans", "gmm", "agglomerative"],
    )
    parser.add_argument(
        "--grouping_feature_set",
        default="control_profile",
        choices=[
            "legacy_capacity_power",
            "static_extended",
            "operational_profile",
            "static_operational",
            "control_profile",
        ],
    )
    parser.add_argument("--grouping_feature_month", type=int, default=None)
    parser.add_argument(
        "--grouping_feature_columns",
        nargs="+",
        default=[
            "bes_capacity_kwh",
            "hvac_total_kw",
            "heating_mean",
            "nsl_mean",
            "comfort_lower_excess_mean",
        ],
        help=(
            "Feature columns for the static expert prior. Defaults to the compact "
            "capacity/load 5-feature set."
        ),
    )

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--layer_N", type=int, default=2)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--ppo_epoch", type=int, default=10)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--value_loss_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name", default="mappo-grouped-tarmac-soft-router")
    parser.add_argument("--save_dir", default="results/mappo_grouped_tarmac_soft_router")
    parser.add_argument("--backup_dir", default=None)

    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--test_save_dir", default=None)

    parser.add_argument("--comm_hidden_dim", type=int, default=64)
    parser.add_argument("--comm_rounds", type=int, default=1)
    parser.add_argument("--comm_key_dim", type=int, default=32)
    parser.add_argument("--comm_value_dim", type=int, default=64)
    parser.add_argument(
        "--comm_fusion_mode",
        default="linear",
        choices=["relu", "linear", "gated"],
        help=(
            "Communication fusion ablation: relu=local Linear+ReLU concat, "
            "linear=local Linear-only concat, gated=context projection with "
            "a learned residual gate."
        ),
    )
    parser.add_argument("--no_comm_residual", action="store_true", default=False)
    parser.add_argument("--comm_dropout", type=float, default=0.0)

    parser.add_argument("--router_hidden_dim", type=int, default=64)
    parser.add_argument("--router_temperature", type=float, default=0.7)
    parser.add_argument("--router_prior_start", type=float, default=0.0)
    parser.add_argument("--router_prior_end", type=float, default=0.7)
    parser.add_argument("--router_warmup_episodes", type=int, default=100)
    parser.add_argument("--router_entropy_scale", type=float, default=0.02)
    parser.add_argument("--no_router_capacity_features", action="store_true", default=False)
    parser.add_argument("--router_eval_soft", action="store_true", default=False)

    args = parser.parse_args()
    return Config(
        climate=args.climate,
        n_buildings=args.n_buildings,
        group_k_candidates=args.group_k_candidates,
        cluster_seed=args.cluster_seed,
        cluster_retries=args.cluster_retries,
        cluster_artifact_dir=args.cluster_artifact_dir,
        grouping_method=args.grouping_method,
        grouping_feature_set=args.grouping_feature_set,
        grouping_feature_month=args.grouping_feature_month,
        grouping_feature_columns=args.grouping_feature_columns,
        hidden_size=args.hidden_size,
        layer_N=args.layer_N,
        lr=args.lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_episodes=args.n_episodes,
        train_month=args.train_month,
        test_month=args.test_month,
        do_test=not args.no_test,
        test_only=args.test_only,
        checkpoint_dir=args.checkpoint_dir,
        test_save_dir=args.test_save_dir,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        save_dir=args.save_dir,
        backup_dir=args.backup_dir,
        comm_hidden_dim=args.comm_hidden_dim,
        comm_rounds=args.comm_rounds,
        comm_use_residual=not args.no_comm_residual,
        comm_dropout=args.comm_dropout,
        comm_key_dim=args.comm_key_dim,
        comm_value_dim=args.comm_value_dim,
        comm_fusion_mode=args.comm_fusion_mode,
        router_hidden_dim=args.router_hidden_dim,
        router_temperature=args.router_temperature,
        router_prior_start=args.router_prior_start,
        router_prior_end=args.router_prior_end,
        router_warmup_episodes=args.router_warmup_episodes,
        router_entropy_scale=args.router_entropy_scale,
        router_use_capacity_features=not args.no_router_capacity_features,
        router_eval_hard=not args.router_eval_soft,
    )


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.test_only:
        run_test_only(cfg)
    else:
        train(cfg)
