from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import SAC

from sb3_independent_common import train_sb3_independent


@dataclass
class Config:
    climate: str = "VT"
    n_buildings: int = 25

    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    learning_starts: int = 500
    buffer_size: int = 100_000
    batch_size: int = 256
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"

    n_episodes: int = 100
    train_month: Optional[int] = None
    test_month: Optional[int] = None
    episode_time_steps: Optional[int] = None
    do_test: bool = True

    save_every: int = 10
    seed: int = 42
    wandb_project: str = "annex96-ce1"
    wandb_name: str = "sb3-independent-sac"
    save_dir: str = "results/sb3_independent_sac"
    backup_dir: Optional[str] = None


def build_model(cfg: Config, env, building_idx: int, episode_steps: int) -> SAC:
    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=cfg.lr,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        policy_kwargs={"net_arch": [cfg.hidden_dim, cfg.hidden_dim]},
        verbose=0,
        seed=cfg.seed + building_idx,
        device="auto",
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="SB3 independent SAC baseline for Annex96 CE1")
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--learning_starts", type=int, default=500)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--ent_coef", default="auto")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_month", type=int, default=None)
    parser.add_argument("--test_month", type=int, default=None)
    parser.add_argument("--episode_time_steps", type=int, default=None)
    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name", default="sb3-independent-sac")
    parser.add_argument("--save_dir", default="results/sb3_independent_sac")
    parser.add_argument("--backup_dir", default=None)
    args = parser.parse_args()
    return Config(
        climate=args.climate,
        n_buildings=args.n_buildings,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
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
        algorithm_label="SB3 Independent SAC",
        model_builder=build_model,
        loss_key_map={
            "train/actor_loss": "actor_loss",
            "train/critic_loss": "critic_loss",
            "train/ent_coef": "ent_coef",
            "train/ent_coef_loss": "ent_coef_loss",
        },
    )


if __name__ == "__main__":
    main()
