from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

from independent_sac.train import (
    _CLIMATE_DEFAULTS,
    _MONTH_ENDS,
    _MONTH_STARTS,
    _export_test_metrics,
    _filter_log_values,
    _make_backup_dir,
    build_env,
    print_repro_metadata,
    seed_everything,
)
from sb3_independent_common import (
    _extract_logger_means,
    _prefix_public_metrics,
    _run_daily_pipeline,
    evaluate_portfolio_models,
    save_plots,
    save_sb3_checkpoint,
)
from training_progress import ProgressTimer

try:
    import wandb

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False
    wandb = None


@dataclass
class Config:
    climate: str = "VT"
    n_buildings: int = 25

    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 10
    batch_size: int = 168

    n_episodes: int = 100
    train_month: Optional[int] = None
    test_month: Optional[int] = None
    episode_time_steps: Optional[int] = None
    do_test: bool = True

    save_every: int = 10
    seed: int = 42
    wandb_project: str = "annex96-ce1"
    wandb_name: str = "sb3-shared-env-independent-ppo"
    save_dir: str = "results/sb3_independent_ppo_shared_env"
    backup_dir: Optional[str] = None


class SingleBuildingDummyEnv(gym.Env):
    """Minimal per-building env used only to initialize SB3 PPO models."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, 0.0, True, False, {}


def _resolved_rollout_steps(cfg: Config, episode_steps: int) -> int:
    return cfg.episode_time_steps if cfg.episode_time_steps is not None else episode_steps


def build_model(cfg: Config, env: gym.Env, building_idx: int, rollout_steps: int) -> PPO:
    batch_size = min(cfg.batch_size, rollout_steps)
    while batch_size > 1 and rollout_steps % batch_size != 0:
        batch_size -= 1

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.lr,
        n_steps=rollout_steps,
        batch_size=batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs={"net_arch": [cfg.hidden_dim, cfg.hidden_dim]},
        verbose=0,
        seed=cfg.seed + building_idx,
        device="auto",
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def _maybe_update_progress(model: PPO, total_timesteps: int) -> None:
    if total_timesteps <= 0:
        return
    if hasattr(model, "_update_current_progress_remaining"):
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
    else:
        progress = 1.0 - min(float(model.num_timesteps) / float(total_timesteps), 1.0)
        setattr(model, "_current_progress_remaining", progress)


def _policy_step(model: PPO, obs: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    obs_batch = obs.reshape((1,) + obs.shape)
    with torch.no_grad():
        obs_tensor = obs_as_tensor(obs_batch, model.device)
        actions, values, log_probs = model.policy(obs_tensor)
    action_np = actions.cpu().numpy()
    if isinstance(model.action_space, gym.spaces.Box):
        if getattr(model.policy, "squash_output", False):
            action_np = model.policy.unscale_action(action_np)
        else:
            action_np = np.clip(action_np, model.action_space.low, model.action_space.high)
    return np.asarray(action_np[0], dtype=np.float32), values, log_probs


def _last_values(model: PPO, obs: np.ndarray) -> torch.Tensor:
    obs_batch = obs.reshape((1,) + obs.shape)
    with torch.no_grad():
        return model.policy.predict_values(obs_as_tensor(obs_batch, model.device))


def _train_shared_env_ppo(
    cfg: Config,
    models: Sequence[PPO],
    train_start: int,
    train_end: int,
    total_timesteps: int,
) -> None:
    env, base_env = build_env(cfg, start_step=train_start, end_step=train_end)
    try:
        obs_list, _ = env.reset(seed=cfg.seed)
        rollout_starts = [np.ones((1,), dtype=np.float32) for _ in models]
        for model in models:
            model.rollout_buffer.reset()
            model.policy.set_training_mode(False)

        while not base_env.terminated:
            obs_arrays = [np.asarray(obs, dtype=np.float32) for obs in obs_list]
            action_arrays: List[np.ndarray] = []
            values_list: List[torch.Tensor] = []
            log_probs_list: List[torch.Tensor] = []

            for i, model in enumerate(models):
                action_np, values, log_probs = _policy_step(model, obs_arrays[i])
                action_arrays.append(action_np)
                values_list.append(values)
                log_probs_list.append(log_probs)

            env_actions = [action.tolist() for action in action_arrays]
            next_obs_list, rewards, terminated, truncated, _ = env.step(env_actions)
            done = bool(terminated or truncated)
            next_obs_arrays = [np.asarray(obs, dtype=np.float32) for obs in next_obs_list]

            for i, model in enumerate(models):
                model.rollout_buffer.add(
                    obs_arrays[i].reshape((1,) + obs_arrays[i].shape),
                    action_arrays[i].reshape((1,) + action_arrays[i].shape),
                    np.array([float(rewards[i])], dtype=np.float32),
                    rollout_starts[i],
                    values_list[i],
                    log_probs_list[i],
                )
                model.num_timesteps += 1
                rollout_starts[i] = np.array([done], dtype=np.float32)
                _maybe_update_progress(model, total_timesteps)

            obs_list = next_obs_list
            if done:
                break

        for i, model in enumerate(models):
            last_values = _last_values(model, next_obs_arrays[i])
            model.rollout_buffer.compute_returns_and_advantage(
                last_values=last_values,
                dones=np.array([done], dtype=np.float32),
            )
            model.train()
    finally:
        try:
            env.close()
        except Exception:
            pass


def train(cfg: Config) -> Sequence[PPO]:
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
    rollout_steps = _resolved_rollout_steps(cfg, episode_steps)
    total_timesteps = cfg.n_episodes * rollout_steps

    cfg.train_month = train_month
    cfg.test_month = test_month

    save_dir = Path(cfg.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed)
    repro_meta = print_repro_metadata(
        cfg,
        torch.device(device),
        train_start,
        train_end,
        test_start,
        test_end,
    )

    probe_env, _ = build_env(cfg, start_step=train_start, end_step=train_end)
    try:
        dummy_envs: List[SingleBuildingDummyEnv] = [
            SingleBuildingDummyEnv(
                observation_space=probe_env.observation_space[i],
                action_space=probe_env.action_space[i],
            )
            for i in range(cfg.n_buildings)
        ]
    finally:
        try:
            probe_env.close()
        except Exception:
            pass

    use_wandb = _WANDB_OK
    if use_wandb:
        cfg_dict = dict(vars(cfg))
        cfg_dict.update(
            {
                "algorithm": "SB3 Shared-Env Independent PPO",
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

    models: List[PPO] = [
        build_model(cfg, dummy_envs[i], i, rollout_steps)
        for i in range(cfg.n_buildings)
    ]

    print("=" * 65)
    print(
        "SB3 Shared-Env Independent PPO | "
        f"Climate: {cfg.climate} | buildings: {cfg.n_buildings} | "
        f"episodes: {cfg.n_episodes}"
    )
    print("=" * 65)

    all_rewards: List[float] = []
    all_primary_metrics: List[Dict[str, Any]] = []
    progress = ProgressTimer(cfg.n_episodes, unit="Ep", label="train")

    for episode in range(1, cfg.n_episodes + 1):
        _train_shared_env_ppo(
            cfg=cfg,
            models=models,
            train_start=train_start,
            train_end=train_end,
            total_timesteps=total_timesteps,
        )

        train_metrics = evaluate_portfolio_models(
            models=models,
            cfg=cfg,
            start_step=train_start,
            end_step=train_end,
            month=train_month,
            deterministic=True,
        )

        loss_metrics = _extract_logger_means(
            models,
            {
                "train/loss": "total_loss",
                "train/policy_gradient_loss": "policy_gradient_loss",
                "train/value_loss": "value_loss",
                "train/entropy_loss": "entropy_loss",
                "train/approx_kl": "approx_kl",
                "train/clip_fraction": "clip_fraction",
                "train/explained_variance": "explained_variance",
            },
        )
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
            save_plots(all_rewards, all_primary_metrics, save_dir, "SB3 Shared-Env Independent PPO")
            save_sb3_checkpoint(models, save_dir, cfg)

        progress.step(episode)

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
            "SB3 Shared-Env Independent PPO",
        )

    save_plots(all_rewards, all_primary_metrics, save_dir, "SB3 Shared-Env Independent PPO")

    final_metrics: Dict[str, Any] = {
        "algorithm": "SB3 Shared-Env Independent PPO",
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

    if use_wandb:
        wandb.finish()

    print("\nTraining complete.")
    print(f"  Artifacts: {save_dir}/")
    print(f"  Backup:    {backup_dir}/")
    return models


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="SB3 shared-environment independent PPO baseline for Annex96 CE1"
    )
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=168)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month", type=int, default=None)
    parser.add_argument("--episode_time_steps", type=int, default=None)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name", default="sb3-shared-env-independent-ppo")
    parser.add_argument("--save_dir", default="results/sb3_independent_ppo_shared_env")
    parser.add_argument("--backup_dir", default=None)
    args = parser.parse_args()
    return Config(
        climate=args.climate,
        n_buildings=args.n_buildings,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_episodes=args.n_episodes,
        train_month=args.train_month,
        test_month=args.test_month,
        episode_time_steps=args.episode_time_steps,
        do_test=not args.no_test,
        save_every=args.save_every,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        save_dir=args.save_dir,
        backup_dir=args.backup_dir,
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
