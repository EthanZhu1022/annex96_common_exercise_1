"""Clustering utilities for policy-induced battery SOC regrouping."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from mappo_grouping_variants.cluster import (
    extract_grouping_features,
    save_grouping_artifacts,
    select_best_grouping,
)

from .features import ENERGY_4F_FEATURES, GROUPING_MODES, SOC_6F_FEATURES


REPO_DIR = Path(__file__).resolve().parent.parent


def _load_soc_statistics(path: Path, n_buildings: int) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"SOC statistics file not found: {path}")

    frame = pd.read_csv(path)
    required = {"building_idx", *SOC_6F_FEATURES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"SOC statistics file is missing columns: {missing}")

    frame["building_idx"] = pd.to_numeric(frame["building_idx"], errors="raise").astype(int)
    expected = list(range(n_buildings))
    actual = sorted(frame["building_idx"].tolist())
    if actual != expected:
        raise ValueError(
            "SOC statistics must contain exactly one row for every building. "
            f"Expected indices {expected}, got {actual}."
        )
    for column in SOC_6F_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"SOC statistics column {column!r} contains non-finite values.")
    return frame.sort_values("building_idx").reset_index(drop=True)


def build_regrouping_features(
    soc_statistics_path: Path,
    grouping_mode: str,
    climate: str,
    n_buildings: int,
    feature_month: int,
    repo_dir: Path = REPO_DIR,
) -> pd.DataFrame:
    """Build the exact feature matrix used for SOC-induced regrouping."""

    if grouping_mode not in GROUPING_MODES:
        raise ValueError(
            f"Unknown grouping_mode={grouping_mode!r}. Choices: {sorted(GROUPING_MODES)}"
        )

    soc_frame = _load_soc_statistics(soc_statistics_path, n_buildings)
    if grouping_mode == "soc6f":
        return soc_frame[["building_idx", *SOC_6F_FEATURES]].copy()

    profile_frame = extract_grouping_features(
        climate=climate,
        n_buildings=n_buildings,
        repo_dir=repo_dir,
        feature_set="static_operational",
        feature_month=feature_month,
        feature_columns=["bes_capacity_kwh", "heating_mean", "nsl_mean"],
    )
    merged = profile_frame.merge(
        soc_frame[["building_idx", "soc_q10"]],
        on="building_idx",
        how="inner",
        validate="one_to_one",
    )
    return merged[["building_idx", *ENERGY_4F_FEATURES]].copy()


def run_clustering(
    *,
    soc_statistics_path: Path,
    grouping_mode: str,
    climate: str,
    save_dir: Path,
    n_buildings: int = 25,
    k_candidates: Optional[List[int]] = None,
    cluster_seed: int = 0,
    retries: int = 10,
    repo_dir: Path = REPO_DIR,
    grouping_method: str = "agglomerative",
    grouping_feature_month: Optional[int] = None,
    **_unused,
) -> Tuple[np.ndarray, Dict]:
    """Cluster buildings from a pretrained policy's SOC behavior."""

    if k_candidates is None:
        k_candidates = [4, 5]
    if grouping_feature_month is None:
        raise ValueError("grouping_feature_month is required for SOC regrouping.")

    feature_frame = build_regrouping_features(
        soc_statistics_path=soc_statistics_path,
        grouping_mode=grouping_mode,
        climate=climate,
        n_buildings=n_buildings,
        feature_month=grouping_feature_month,
        repo_dir=repo_dir,
    )
    feature_columns = [column for column in feature_frame.columns if column != "building_idx"]

    print(
        f"\n[soc-cluster] method={grouping_method} mode={grouping_mode} "
        f"month={grouping_feature_month} climate={climate}"
    )
    print(f"  source statistics: {Path(soc_statistics_path).resolve()}")
    print(f"  features ({len(feature_columns)}): {', '.join(feature_columns)}")

    result = select_best_grouping(
        feature_df=feature_frame,
        method=grouping_method,
        k_candidates=k_candidates,
        cluster_seed=cluster_seed,
        retries=retries,
    )
    result.update(
        {
            "grouping_method": grouping_method,
            "grouping_feature_set": f"policy_soc_{grouping_mode}",
            "grouping_feature_month": grouping_feature_month,
            "grouping_feature_columns": feature_columns,
            "soc_statistics_path": str(Path(soc_statistics_path).resolve()),
            "soc_grouping_mode": grouping_mode,
        }
    )

    print(
        f"[soc-cluster] Best: method={grouping_method} K={result['k']} "
        f"seed={result['seed']} balance={result['balance']:.3f}"
    )
    for cluster_id, size in enumerate(result["sizes"]):
        indices = np.where(result["assignments"] == cluster_id)[0].tolist()
        print(f"  Group {cluster_id}: {size} buildings -> {indices}")

    save_dir = Path(save_dir).resolve()
    save_grouping_artifacts(
        result=result,
        feature_df=feature_frame,
        save_dir=save_dir,
        method=grouping_method,
        feature_set=f"policy_soc_{grouping_mode}",
        feature_month=grouping_feature_month,
        grouping_feature_columns=feature_columns,
    )

    summary_path = save_dir / "cluster_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {
            "soc_grouping_mode": grouping_mode,
            "soc_statistics_path": str(Path(soc_statistics_path).resolve()),
            "policy_induced_features": True,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    source_path = Path(soc_statistics_path).resolve()
    copied_path = save_dir / "source_soc_statistics.csv"
    if source_path != copied_path.resolve():
        shutil.copy2(source_path, copied_path)
    source_metadata = source_path.parent / "soc_collection_metadata.json"
    if source_metadata.exists():
        shutil.copy2(
            source_metadata,
            save_dir / "source_soc_collection_metadata.json",
        )

    return result["assignments"], result
