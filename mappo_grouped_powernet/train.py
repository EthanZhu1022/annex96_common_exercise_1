"""
Grouped MAPPO with PowerNet-style neighbor communication.

This entrypoint reuses the grouped communication pipeline from
`mappo_grouped_comm.train`, so metrics, W&B logging, testing, plots, and result
artifacts stay aligned with the existing grouped communication experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

from mappo_grouped_comm import train as base_train


@dataclass
class Config:
    climate: str = "VT"
    n_buildings: int = 25
    plot_label: str = "Grouped MAPPO (PowerNet)"

    group_k_candidates: List[int] = field(default_factory=lambda: [4, 5])
    cluster_seed: int = 0
    cluster_retries: int = 10
    cluster_artifact_dir: Optional[str] = None

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
    wandb_name: str = "mappo-grouped-powernet"
    save_dir: str = "results/mappo_grouped_powernet"
    backup_dir: Optional[str] = None

    comm_method: str = "powernet"
    comm_hidden_dim: int = 64
    comm_rounds: int = 1
    comm_share_within_group_only: bool = True
    comm_use_residual: bool = True
    comm_dropout: float = 0.0
    comm_topology: str = "ring"
    comm_neighbors: int = 1


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Grouped MAPPO with PowerNet-style neighbor communication."
    )
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)

    parser.add_argument("--group_k_candidates", type=int, nargs="+", default=[4, 5], metavar="K")
    parser.add_argument("--cluster_seed", type=int, default=0)
    parser.add_argument("--cluster_retries", type=int, default=10)
    parser.add_argument("--cluster_artifact_dir", default=None)

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
    parser.add_argument("--wandb_name", default="mappo-grouped-powernet")
    parser.add_argument("--save_dir", default="results/mappo_grouped_powernet")
    parser.add_argument("--backup_dir", default=None)

    parser.add_argument("--no_test", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--test_save_dir", default=None)

    parser.add_argument("--comm_hidden_dim", type=int, default=64)
    parser.add_argument("--comm_rounds", type=int, default=1)
    parser.add_argument("--comm_topology", default="ring", choices=["ring", "chain", "full"])
    parser.add_argument("--comm_neighbors", type=int, default=1)
    parser.add_argument("--no_comm_residual", action="store_true", default=False)
    parser.add_argument("--comm_dropout", type=float, default=0.0)

    args = parser.parse_args()
    return Config(
        climate=args.climate,
        n_buildings=args.n_buildings,
        group_k_candidates=args.group_k_candidates,
        cluster_seed=args.cluster_seed,
        cluster_retries=args.cluster_retries,
        cluster_artifact_dir=args.cluster_artifact_dir,
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
        comm_topology=args.comm_topology,
        comm_neighbors=args.comm_neighbors,
    )


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.test_only:
        base_train.run_test_only(cfg)
    else:
        base_train.train(cfg)
