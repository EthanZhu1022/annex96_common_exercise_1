from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import PPO

from sb3_independent_common import train_sb3_independent


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
    wandb_name: str = "sb3-independent-ppo"
    save_dir: str = "results/sb3_independent_ppo"
    backup_dir: Optional[str] = None


def build_model(cfg: Config, env, building_idx: int, episode_steps: int) -> PPO:
    batch_size = min(cfg.batch_size, episode_steps)
    while batch_size > 1 and episode_steps % batch_size != 0:
        batch_size -= 1

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.lr,
        n_steps=episode_steps,
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


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="SB3 independent PPO baseline for Annex96 CE1")
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
    parser.add_argument("--wandb_name", default="sb3-independent-ppo")
    parser.add_argument("--save_dir", default="results/sb3_independent_ppo")
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
    cfg = parse_args()
    train_sb3_independent(
        cfg=cfg,
        algorithm_label="SB3 Independent PPO",
        model_builder=build_model,
        loss_key_map={
            "train/loss": "total_loss",
            "train/policy_gradient_loss": "policy_gradient_loss",
            "train/value_loss": "value_loss",
            "train/entropy_loss": "entropy_loss",
            "train/approx_kl": "approx_kl",
            "train/clip_fraction": "clip_fraction",
            "train/explained_variance": "explained_variance",
        },
    )


if __name__ == "__main__":
    main()
