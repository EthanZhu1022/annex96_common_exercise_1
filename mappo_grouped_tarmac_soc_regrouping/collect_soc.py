"""Collect January electrical-storage SOC from a pretrained grouped TarMAC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from mappo_grouped_comm.train import _MONTH_ENDS, _MONTH_STARTS, seed_everything
from mappo_grouped_tarmac_hybrid_grouping.train import (
    Config as SourceModelConfig,
    _apply_checkpoint_model_config,
    _build_grouped_components,
)

from .env import CityLearnMAPPOEnv
from .features import compute_soc_statistics


EXPECTED_SOURCE_FEATURES = [
    "bes_capacity_kwh",
    "heating_mean",
    "nsl_mean",
]


def _resolve_checkpoint(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "checkpoint.pt"
    if not candidate.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {candidate}")
    return candidate


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_name)


def _validate_source_checkpoint(
    checkpoint: Dict,
    climate: str,
    n_buildings: int,
    collection_month: int,
) -> None:
    saved = checkpoint.get("cfg", {})
    if not isinstance(saved, dict):
        raise ValueError("Source checkpoint does not contain a cfg mapping.")

    expected_pairs = {
        "climate": climate,
        "n_buildings": n_buildings,
        "train_month": collection_month,
        "grouping_method": "agglomerative",
        "comm_fusion_mode": "linear",
    }
    mismatches = {
        key: {"expected": expected, "actual": saved.get(key)}
        for key, expected in expected_pairs.items()
        if saved.get(key) != expected
    }
    if saved.get("grouping_feature_columns") != EXPECTED_SOURCE_FEATURES:
        mismatches["grouping_feature_columns"] = {
            "expected": EXPECTED_SOURCE_FEATURES,
            "actual": saved.get("grouping_feature_columns"),
        }
    if int(checkpoint.get("episode", -1)) != 500:
        mismatches["episode"] = {"expected": 500, "actual": checkpoint.get("episode")}
    if mismatches:
        raise ValueError(
            "The source checkpoint is not the requested final 3f model:\n"
            + json.dumps(mismatches, indent=2)
        )


def _record_soc_snapshot(
    env: CityLearnMAPPOEnv,
    sample_index: int,
    absolute_start_step: int,
    rows: List[Dict[str, object]],
) -> None:
    for building_idx, building in enumerate(env.base_env.buildings):
        time_step = int(building.time_step)
        soc = float(building.electrical_storage.soc[time_step])
        rows.append(
            {
                "building_idx": building_idx,
                "building_name": str(getattr(building, "name", f"building_{building_idx}")),
                "sample_index": sample_index,
                "absolute_time_step": absolute_start_step + sample_index,
                "day_index": sample_index // 24,
                "day_of_month": sample_index // 24 + 1,
                "hour_of_day": sample_index % 24,
                "electrical_storage_soc": soc,
            }
        )


def collect_soc(
    checkpoint_path: str,
    output_dir: str,
    climate: str = "VT",
    n_buildings: int = 25,
    collection_month: int = 1,
    seed: int = 42,
    device_name: str = "auto",
) -> Dict[str, Path]:
    """Run deterministic inference and save hourly SOC and per-building statistics."""

    if collection_month not in _MONTH_STARTS:
        raise ValueError(f"Invalid collection_month={collection_month}.")

    checkpoint_file = _resolve_checkpoint(checkpoint_path)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(device_name)
    seed_everything(seed)

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    _validate_source_checkpoint(checkpoint, climate, n_buildings, collection_month)

    source_cfg = SourceModelConfig(
        climate=climate,
        n_buildings=n_buildings,
        train_month=collection_month,
        test_month=2,
        seed=seed,
    )
    _apply_checkpoint_model_config(source_cfg, checkpoint)
    group_assignments = np.asarray(checkpoint["group_assignments"], dtype=np.int64)
    if group_assignments.shape != (n_buildings,):
        raise ValueError(
            f"Expected {n_buildings} source group assignments, got {group_assignments.shape}."
        )

    start_step = _MONTH_STARTS[collection_month]
    end_step = _MONTH_ENDS[collection_month]
    env = CityLearnMAPPOEnv(
        climate=climate,
        n_buildings=n_buildings,
        start_step=start_step,
        end_step=end_step,
    )
    (
        mappo_args,
        _policies,
        _trainers,
        _buffers,
        _group_indices,
        controller,
        _actor_optimizer,
    ) = _build_grouped_components(source_cfg, env, group_assignments, device)
    controller.load_state_dict(checkpoint["global_actor_state_dict"])
    controller.eval()

    obs, _ = env.reset(seed=seed)
    rnn_states = np.zeros(
        (n_buildings, mappo_args.recurrent_N, mappo_args.hidden_size),
        dtype=np.float32,
    )
    masks = np.ones((n_buildings, 1), dtype=np.float32)
    trajectory_rows: List[Dict[str, object]] = []

    # CityLearn reset exposes hour 0. Each step advances to the next hour, so
    # one 744-hour month contains the reset snapshot plus 743 transitions.
    _record_soc_snapshot(env, 0, start_step, trajectory_rows)
    done = False
    action_steps = 0
    while not done:
        with torch.no_grad():
            actions, _, new_rnn_states = controller.act(
                obs=obs,
                rnn_states=rnn_states,
                masks=masks,
                deterministic=True,
            )
        clipped_actions = np.clip(actions.cpu().numpy(), -1.0, 1.0)
        obs, _, _, done, _ = env.step(clipped_actions)
        action_steps += 1
        rnn_states = new_rnn_states.cpu().numpy().reshape(
            n_buildings, mappo_args.recurrent_N, mappo_args.hidden_size
        )
        masks = (
            np.zeros((n_buildings, 1), dtype=np.float32)
            if done
            else np.ones((n_buildings, 1), dtype=np.float32)
        )
        _record_soc_snapshot(env, action_steps, start_step, trajectory_rows)

    expected_samples = end_step - start_step + 1
    if action_steps + 1 != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} hourly SOC samples, got {action_steps + 1}."
        )

    trajectory = pd.DataFrame(trajectory_rows)
    counts = trajectory.groupby("building_idx").size()
    if not (counts == expected_samples).all():
        raise RuntimeError(
            "Not every building has the expected number of SOC samples: "
            f"{counts.to_dict()}"
        )
    statistics = compute_soc_statistics(trajectory)

    trajectory_file = output_path / "soc_hourly_trajectory.csv"
    statistics_file = output_path / "soc_statistics.csv"
    metadata_file = output_path / "soc_collection_metadata.json"
    trajectory.to_csv(trajectory_file, index=False)
    statistics.to_csv(statistics_file, index=False)
    metadata = {
        "source_checkpoint": str(checkpoint_file),
        "source_checkpoint_episode": int(checkpoint.get("episode", -1)),
        "source_group_assignments": group_assignments.tolist(),
        "source_grouping_feature_columns": EXPECTED_SOURCE_FEATURES,
        "deterministic": True,
        "climate": climate,
        "n_buildings": n_buildings,
        "collection_month": collection_month,
        "absolute_start_step": start_step,
        "absolute_end_step": end_step,
        "hourly_soc_samples_per_building": expected_samples,
        "policy_action_steps": action_steps,
        "seed": seed,
        "device": str(device),
        "soc_units": "normalized_fraction_0_to_1",
        "statistics": {
            "soc_mean": "mean of 744 hourly normalized SOC samples",
            "soc_std": "population standard deviation (ddof=0)",
            "soc_q10": "10th percentile",
            "soc_low_fraction": "fraction with SOC < 0.1",
            "soc_high_fraction": "fraction with SOC > 0.9",
            "soc_daily_range_mean": "mean over 31 daily (max SOC - min SOC) ranges",
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))

    print("\nDeterministic SOC collection complete.")
    print(f"  device: {device}")
    print(f"  source checkpoint: {checkpoint_file}")
    print(f"  hourly samples/building: {expected_samples}")
    print(f"  policy action steps: {action_steps}")
    print(f"  trajectory: {trajectory_file}")
    print(f"  statistics: {statistics_file}")
    print(f"  metadata: {metadata_file}")
    return {
        "trajectory": trajectory_file,
        "statistics": statistics_file,
        "metadata": metadata_file,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect policy-induced electrical-storage SOC from a grouped TarMAC checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--climate", default="VT", choices=["VT", "TX"])
    parser.add_argument("--n_buildings", type=int, default=25)
    parser.add_argument("--collection_month", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_soc(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        climate=args.climate,
        n_buildings=args.n_buildings,
        collection_month=args.collection_month,
        seed=args.seed,
        device_name=args.device,
    )
