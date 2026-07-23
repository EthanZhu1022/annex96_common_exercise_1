"""Train a fresh grouped TarMAC model from policy-induced SOC clusters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import mappo_grouped_tarmac_hybrid_grouping.train as grouped_train
from mappo_grouped_tarmac_hybrid_grouping.train import Config as GroupedTarMACConfig

from .cluster import run_clustering as run_soc_clustering
from .features import GROUPING_MODES


@dataclass
class Config(GroupedTarMACConfig):
    soc_statistics_path: str = ""
    soc_grouping_mode: str = "soc6f"


def _soc_feature_columns(mode: str) -> List[str]:
    if mode not in GROUPING_MODES:
        raise ValueError(f"Unknown soc_grouping_mode={mode!r}.")
    return list(GROUPING_MODES[mode])


def train(cfg: Config) -> None:
    """Inject SOC clustering into the proven grouped TarMAC training loop."""

    statistics_path = Path(cfg.soc_statistics_path).expanduser().resolve()
    if not statistics_path.exists():
        raise FileNotFoundError(f"SOC statistics file not found: {statistics_path}")
    if cfg.soc_grouping_mode not in GROUPING_MODES:
        raise ValueError(
            f"Unknown soc_grouping_mode={cfg.soc_grouping_mode!r}. "
            f"Choices: {sorted(GROUPING_MODES)}"
        )

    cfg.soc_statistics_path = str(statistics_path)
    cfg.grouping_feature_set = f"policy_soc_{cfg.soc_grouping_mode}"
    cfg.grouping_feature_columns = _soc_feature_columns(cfg.soc_grouping_mode)
    if cfg.grouping_feature_month is None:
        cfg.grouping_feature_month = cfg.train_month

    original_run_clustering = grouped_train.run_clustering

    def _run_soc_clustering_adapter(**kwargs):
        return run_soc_clustering(
            soc_statistics_path=statistics_path,
            grouping_mode=cfg.soc_grouping_mode,
            **kwargs,
        )

    grouped_train.run_clustering = _run_soc_clustering_adapter
    try:
        # This constructs new policies and optimizers. The source checkpoint is
        # deliberately not accepted here, so stage 2 always trains from scratch.
        grouped_train.train(cfg)
    finally:
        grouped_train.run_clustering = original_run_clustering


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train grouped TarMAC from policy-induced SOC regrouping features."
    )
    parser.add_argument("--soc_statistics_path", required=True)
    parser.add_argument("--soc_grouping_mode", choices=sorted(GROUPING_MODES), required=True)
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)
    parser.add_argument("--group_k_candidates", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--cluster_seed", type=int, default=0)
    parser.add_argument("--cluster_retries", type=int, default=10)
    parser.add_argument("--cluster_artifact_dir", default=None)
    parser.add_argument(
        "--grouping_method",
        default="agglomerative",
        choices=["kmeans", "gmm", "agglomerative"],
    )
    parser.add_argument("--grouping_feature_month", type=int, default=None)

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

    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--train_month", type=int, default=1)
    parser.add_argument("--test_month", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", default="annex96-ce1")
    parser.add_argument("--wandb_name", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--backup_dir", default=None)
    parser.add_argument("--no_test", action="store_true")

    parser.add_argument("--comm_hidden_dim", type=int, default=64)
    parser.add_argument("--comm_rounds", type=int, default=1)
    parser.add_argument("--comm_key_dim", type=int, default=32)
    parser.add_argument("--comm_value_dim", type=int, default=64)
    parser.add_argument(
        "--comm_fusion_mode",
        default="linear",
        choices=["relu", "linear", "gated"],
    )
    parser.add_argument("--no_comm_residual", action="store_true")
    parser.add_argument("--comm_dropout", type=float, default=0.0)

    args = parser.parse_args()
    return Config(
        climate=args.climate,
        n_buildings=args.n_buildings,
        group_k_candidates=args.group_k_candidates,
        cluster_seed=args.cluster_seed,
        cluster_retries=args.cluster_retries,
        cluster_artifact_dir=args.cluster_artifact_dir,
        grouping_method=args.grouping_method,
        grouping_feature_set=f"policy_soc_{args.soc_grouping_mode}",
        grouping_feature_month=args.grouping_feature_month,
        grouping_feature_columns=_soc_feature_columns(args.soc_grouping_mode),
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
        soc_statistics_path=args.soc_statistics_path,
        soc_grouping_mode=args.soc_grouping_mode,
    )


if __name__ == "__main__":
    train(parse_args())
