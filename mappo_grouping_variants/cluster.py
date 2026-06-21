"""
Alternative building grouping methods for grouped MAPPO experiments.

This module keeps the same public run_clustering API used by the existing
grouped MAPPO train scripts, but adds two controls:

  grouping_method: kmeans, gmm, agglomerative
  grouping_feature_set: legacy_capacity_power, static_extended,
                        operational_profile, static_operational

The saved artifact names match mappo_grouped.cluster so downstream reporting
continues to work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from mappo_grouped.cluster import extract_building_features as extract_legacy_features


REPO_DIR = Path(__file__).resolve().parent.parent

GROUPING_METHODS = ("kmeans", "gmm", "agglomerative")
GROUPING_FEATURE_SETS = (
    "legacy_capacity_power",
    "static_extended",
    "operational_profile",
    "static_operational",
)


def _schema_path(climate: str, repo_dir: Path) -> Path:
    dataset_name = f"annex96_ce1_{climate.lower()}_neighborhood"
    path = repo_dir / "data" / "datasets" / dataset_name / "schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return path


def _load_schema(climate: str, repo_dir: Path) -> Tuple[Path, Dict]:
    path = _schema_path(climate, repo_dir)
    return path, json.loads(path.read_text())


def _building_items(schema: Dict, n_buildings: int) -> List[Tuple[str, Dict]]:
    return list(schema["buildings"].items())[:n_buildings]


def _nested_float(data: Dict, keys: Iterable[str], default: float = 0.0) -> float:
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return float(default)
        value = value[key]
    try:
        return float(value)
    except Exception:
        return float(default)


def _series_stats(df: pd.DataFrame, column: str, prefix: str) -> Dict[str, float]:
    if column not in df.columns:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_sum": 0.0,
        }
    values = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_sum": float(values.sum()),
    }


def _month_slice(df: pd.DataFrame, feature_month: Optional[int]) -> pd.DataFrame:
    if feature_month is None or "month" not in df.columns:
        return df
    month_values = pd.to_numeric(df["month"], errors="coerce")
    sliced = df.loc[month_values == int(feature_month)]
    return sliced if len(sliced) > 0 else df


def _extract_static_extended_features(
    climate: str,
    n_buildings: int,
    repo_dir: Path,
) -> pd.DataFrame:
    legacy = extract_legacy_features(climate, n_buildings, repo_dir)
    _schema_file, schema = _load_schema(climate, repo_dir)
    rows: List[Dict[str, float]] = []
    for i, (_name, building_cfg) in enumerate(_building_items(schema, n_buildings)):
        pv_attrs = building_cfg.get("pv", {}) or {}
        pv_autosize = pv_attrs.get("autosize_attributes", {}) or {}
        rows.append(
            {
                "building_idx": i,
                "bes_capacity_kwh": float(legacy[i, 0]),
                "hvac_total_kw": float(legacy[i, 1]),
                "pv_sizing_nominal_power_kw": _nested_float(
                    pv_attrs, ("nominal_power",), _nested_float(pv_autosize, ("pv_sizing_nominal_power",))
                ),
                "pv_roof_area_m2": _nested_float(pv_autosize, ("roof_area",)),
                "has_dhw_storage": float(building_cfg.get("dhw_storage") is not None),
                "has_cooling_storage": float(building_cfg.get("cooling_storage") is not None),
                "has_heating_storage": float(building_cfg.get("heating_storage") is not None),
            }
        )
    return pd.DataFrame(rows)


def _extract_operational_profile_features(
    climate: str,
    n_buildings: int,
    repo_dir: Path,
    feature_month: Optional[int],
) -> pd.DataFrame:
    schema_file, schema = _load_schema(climate, repo_dir)
    dataset_dir = schema_file.parent
    rows: List[Dict[str, float]] = []
    stat_columns = [
        ("non_shiftable_load", "nsl"),
        ("cooling_demand", "cooling"),
        ("heating_demand", "heating"),
        ("dhw_demand", "dhw"),
        ("solar_generation", "solar"),
        ("occupant_count", "occupants"),
        ("indoor_dry_bulb_temperature", "indoor_temp"),
    ]

    for i, (_name, building_cfg) in enumerate(_building_items(schema, n_buildings)):
        csv_name = building_cfg.get("energy_simulation")
        if not csv_name:
            raise ValueError(f"Building {i} has no energy_simulation CSV in schema.")
        df = pd.read_csv(dataset_dir / csv_name)
        df = _month_slice(df, feature_month)

        row: Dict[str, float] = {"building_idx": i}
        for column, prefix in stat_columns:
            row.update(_series_stats(df, column, prefix))

        if {
            "indoor_dry_bulb_temperature",
            "indoor_dry_bulb_temperature_cooling_set_point",
            "indoor_dry_bulb_temperature_heating_set_point",
        }.issubset(df.columns):
            temp = pd.to_numeric(df["indoor_dry_bulb_temperature"], errors="coerce").fillna(0.0)
            cooling_sp = pd.to_numeric(
                df["indoor_dry_bulb_temperature_cooling_set_point"], errors="coerce"
            ).fillna(24.0)
            heating_sp = pd.to_numeric(
                df["indoor_dry_bulb_temperature_heating_set_point"], errors="coerce"
            ).fillna(20.0)
            row["comfort_upper_excess_mean"] = float(np.maximum(temp - cooling_sp, 0.0).mean())
            row["comfort_lower_excess_mean"] = float(np.maximum(heating_sp - temp, 0.0).mean())
        else:
            row["comfort_upper_excess_mean"] = 0.0
            row["comfort_lower_excess_mean"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def extract_grouping_features(
    climate: str,
    n_buildings: int = 25,
    repo_dir: Path = REPO_DIR,
    feature_set: str = "legacy_capacity_power",
    feature_month: Optional[int] = None,
) -> pd.DataFrame:
    if feature_set not in GROUPING_FEATURE_SETS:
        raise ValueError(f"Unknown feature_set={feature_set}. Choices: {GROUPING_FEATURE_SETS}")

    if feature_set == "legacy_capacity_power":
        legacy = extract_legacy_features(climate, n_buildings, repo_dir)
        return pd.DataFrame(
            {
                "building_idx": np.arange(n_buildings),
                "bes_capacity_kwh": legacy[:, 0],
                "hvac_total_kw": legacy[:, 1],
            }
        )

    static_df = _extract_static_extended_features(climate, n_buildings, repo_dir)
    if feature_set == "static_extended":
        return static_df

    operational_df = _extract_operational_profile_features(climate, n_buildings, repo_dir, feature_month)
    if feature_set == "operational_profile":
        return operational_df

    return static_df.merge(operational_df, on="building_idx", how="inner")


def _balance_score(labels: np.ndarray, k: int) -> float:
    sizes = np.array([int((labels == c).sum()) for c in range(k)])
    if sizes.min() == 0:
        return -1.0
    if sizes.min() == 1:
        return sizes.min() / sizes.max() - 0.5
    return sizes.min() / sizes.max()


def _fit_labels(method: str, x_scaled: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        return model.fit_predict(x_scaled)
    if method == "gmm":
        model = GaussianMixture(
            n_components=k,
            random_state=seed,
            n_init=3,
            covariance_type="full",
            reg_covar=1e-6,
        )
        return model.fit_predict(x_scaled)
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        return model.fit_predict(x_scaled)
    raise ValueError(f"Unknown grouping method={method}. Choices: {GROUPING_METHODS}")


def _cluster_centers(features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    centers = []
    for c in range(k):
        mask = labels == c
        centers.append(features[mask].mean(axis=0) if mask.any() else np.zeros(features.shape[1]))
    return np.vstack(centers)


def select_best_grouping(
    feature_df: pd.DataFrame,
    method: str,
    k_candidates: Optional[List[int]],
    cluster_seed: int,
    retries: int,
) -> Dict:
    if method not in GROUPING_METHODS:
        raise ValueError(f"Unknown grouping method={method}. Choices: {GROUPING_METHODS}")
    if k_candidates is None:
        k_candidates = [4, 5]

    feature_columns = [c for c in feature_df.columns if c != "building_idx"]
    raw_features = feature_df[feature_columns].to_numpy(dtype=np.float64)
    x_scaled = StandardScaler().fit_transform(raw_features)

    best: Optional[Dict] = None
    best_score = -np.inf
    all_candidates: List[Dict] = []
    method_retries = 1 if method == "agglomerative" else retries

    for k in k_candidates:
        for trial in range(method_retries):
            seed = cluster_seed + trial
            labels = _fit_labels(method, x_scaled, k, seed)
            score = _balance_score(labels, k)
            sizes = [int((labels == c).sum()) for c in range(k)]
            centers = _cluster_centers(raw_features, labels, k)
            candidate = {
                "assignments": labels.copy(),
                "centers_orig": centers,
                "k": k,
                "seed": seed if method != "agglomerative" else None,
                "balance": score,
                "sizes": sizes,
            }
            all_candidates.append(candidate)
            if score > best_score:
                best_score = score
                best = candidate

    assert best is not None
    sizes_str = ", ".join(f"group {c}: {s} buildings" for c, s in enumerate(best["sizes"]))
    best["feature_columns"] = feature_columns
    best["reason"] = (
        f"Chose method={method} K={best['k']} seed={best['seed']} "
        f"(balance={best['balance']:.3f}; {sizes_str}). "
        f"Tried {len(k_candidates)} K values x {method_retries} fits. "
        f"Balance = min_cluster_size / max_cluster_size."
    )
    return best


def save_grouping_artifacts(
    result: Dict,
    feature_df: pd.DataFrame,
    save_dir: Path,
    method: str,
    feature_set: str,
    feature_month: Optional[int],
) -> Dict[str, Path]:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assignments = result["assignments"]
    centers_orig = result["centers_orig"]
    k = result["k"]
    sizes = result["sizes"]
    feature_columns = result["feature_columns"]

    assign_df = feature_df.copy()
    assign_df.insert(1, "cluster", assignments.tolist())
    assign_path = save_dir / "building_cluster_assignment.csv"
    assign_df.to_csv(assign_path, index=False)

    centers_df = pd.DataFrame(centers_orig, columns=feature_columns)
    centers_df.insert(0, "cluster", list(range(k)))
    centers_df.insert(1, "n_buildings", sizes)
    centers_path = save_dir / "cluster_centers.csv"
    centers_df.to_csv(centers_path, index=False)

    summary = {
        "grouping_method": method,
        "grouping_feature_set": feature_set,
        "grouping_feature_month": feature_month,
        "k": k,
        "seed": result["seed"],
        "balance": float(result["balance"]),
        "cluster_sizes": sizes,
        "feature_columns": feature_columns,
        "cluster_centers_original_scale": {
            f"cluster_{c}": {
                **{feature: float(centers_orig[c, j]) for j, feature in enumerate(feature_columns)},
                "n_buildings": sizes[c],
            }
            for c in range(k)
        },
        "group_assignments": {
            f"building_{i}": int(assignments[i]) for i in range(len(assignments))
        },
        "selection_rationale": result["reason"],
    }
    summary_path = save_dir / "cluster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"  [cluster] assignment CSV -> {assign_path}")
    print(f"  [cluster] centers CSV    -> {centers_path}")
    print(f"  [cluster] summary JSON   -> {summary_path}")
    return {"assignment": assign_path, "centers": centers_path, "summary": summary_path}


def run_clustering(
    climate: str,
    save_dir: Path,
    n_buildings: int = 25,
    k_candidates: Optional[List[int]] = None,
    cluster_seed: int = 0,
    retries: int = 10,
    repo_dir: Path = REPO_DIR,
    grouping_method: str = "kmeans",
    grouping_feature_set: str = "legacy_capacity_power",
    grouping_feature_month: Optional[int] = None,
    **_unused,
) -> Tuple[np.ndarray, Dict]:
    if k_candidates is None:
        k_candidates = [4, 5]

    print(
        f"\n[cluster] method={grouping_method} feature_set={grouping_feature_set} "
        f"month={grouping_feature_month} climate={climate}"
    )
    feature_df = extract_grouping_features(
        climate=climate,
        n_buildings=n_buildings,
        repo_dir=repo_dir,
        feature_set=grouping_feature_set,
        feature_month=grouping_feature_month,
    )
    feature_columns = [c for c in feature_df.columns if c != "building_idx"]
    print(f"  features ({len(feature_columns)}): {', '.join(feature_columns)}")

    result = select_best_grouping(
        feature_df=feature_df,
        method=grouping_method,
        k_candidates=k_candidates,
        cluster_seed=cluster_seed,
        retries=retries,
    )
    result["grouping_method"] = grouping_method
    result["grouping_feature_set"] = grouping_feature_set
    result["grouping_feature_month"] = grouping_feature_month

    print(
        f"[cluster] Best: method={grouping_method} K={result['k']} "
        f"seed={result['seed']} balance={result['balance']:.3f}"
    )
    for c, s in enumerate(result["sizes"]):
        idx = np.where(result["assignments"] == c)[0].tolist()
        print(f"  Group {c}: {s} buildings -> {idx}")

    save_grouping_artifacts(
        result=result,
        feature_df=feature_df,
        save_dir=save_dir,
        method=grouping_method,
        feature_set=grouping_feature_set,
        feature_month=grouping_feature_month,
    )
    return result["assignments"], result

