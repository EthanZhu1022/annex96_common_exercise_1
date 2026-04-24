"""
Building clustering for mappo_grouped.
=======================================

Clusters 25 CityLearn buildings into K=4 or K=5 groups using K-means on
two normalised flexibility features:
  1. BES capacity (kWh)   — building.electrical_storage.capacity
  2. HVAC total power (kW) — cooling_device.nominal_power + heating_device.nominal_power

Clustering workflow
-------------------
1. Instantiate CityLearnEnv briefly and reset once to trigger device autosizing.
2. Extract [BES_cap, HVAC_total] for each of the 25 buildings.
3. Normalise with sklearn StandardScaler.
4. Try each K in {4, 5} × `retries` random seeds; score by balance metric:
       score = min_cluster_size / max_cluster_size  (higher = more balanced)
   Select the K/seed with the highest balance score.
   Reject any solution with a single-building cluster (size==1) if a better
   alternative exists.
5. Save cluster artifacts to `save_dir`:
   - building_cluster_assignment.csv
   - cluster_centers.csv            (centres in original feature scale)
   - cluster_summary.json

Public API
----------
  features  = extract_building_features(climate, n_buildings, repo_dir)
  result    = select_best_clustering(features, k_candidates, cluster_seed, retries)
  artifacts = save_cluster_artifacts(result, features, save_dir)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_DIR = Path(__file__).resolve().parent.parent
for _p in [str(REPO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from annex96_rewards import DEFAULT_CE1_REWARD_PATH, build_ce1_reward_kwargs
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_building_features(
    climate:     str,
    n_buildings: int = 25,
    repo_dir:    Path = REPO_DIR,
) -> np.ndarray:
    """Extract [BES_capacity, HVAC_total_capacity] for each building.

    Returns
    -------
    features : np.ndarray of shape (n_buildings, 2)
        Column 0: electrical_storage.capacity    [kWh]
        Column 1: cooling_device.nominal_power
                + heating_device.nominal_power   [kW]
    """
    dataset_name = f"annex96_ce1_{climate.lower()}_neighborhood"
    schema_path  = repo_dir / "data" / "datasets" / dataset_name / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    base_env = CityLearnEnv(
        schema          = str(schema_path),
        root_directory  = str(schema_path.parent),
        central_agent   = False,
        buildings       = list(range(n_buildings)),
        reward_function = DEFAULT_CE1_REWARD_PATH,
        reward_function_kwargs = build_ce1_reward_kwargs(),
    )
    # Reset once to trigger device autosizing
    NormalizedObservationWrapper(base_env).reset(seed=0)

    features = np.zeros((n_buildings, 2), dtype=np.float64)
    for i, building in enumerate(base_env.buildings):
        # BES capacity
        bes_cap = 0.0
        try:
            bes_cap = float(building.electrical_storage.capacity)
        except Exception:
            pass

        # HVAC total = cooling + heating nominal power
        hvac_total = 0.0
        try:
            if building.cooling_device is not None:
                hvac_total += float(building.cooling_device.nominal_power)
        except Exception:
            pass
        try:
            if building.heating_device is not None:
                hvac_total += float(building.heating_device.nominal_power)
        except Exception:
            pass

        features[i, 0] = bes_cap
        features[i, 1] = hvac_total

    return features


# ---------------------------------------------------------------------------
# Balance scoring
# ---------------------------------------------------------------------------

def _balance_score(labels: np.ndarray, k: int) -> float:
    """min_cluster_size / max_cluster_size — higher is more balanced."""
    sizes = np.array([int((labels == c).sum()) for c in range(k)])
    if sizes.min() == 0:
        return -1.0   # degenerate: empty cluster
    if sizes.min() == 1:
        return sizes.min() / sizes.max() - 0.5   # penalise singleton clusters
    return sizes.min() / sizes.max()


# ---------------------------------------------------------------------------
# Main clustering selector
# ---------------------------------------------------------------------------

def select_best_clustering(
    features:      np.ndarray,
    k_candidates:  List[int] = None,
    cluster_seed:  int = 0,
    retries:       int = 10,
) -> Dict:
    """Run K-means for each K candidate × seed, return the best balanced result.

    Parameters
    ----------
    features      : (n_buildings, 2) raw feature matrix
    k_candidates  : list of K values to try (default [4, 5])
    cluster_seed  : base seed; seeds tried are cluster_seed … cluster_seed+retries-1
    retries       : number of random seeds to try per K

    Returns
    -------
    dict with keys:
      assignments  — np.ndarray (n_buildings,) cluster labels [0, K-1]
      centers_orig — np.ndarray (K, 2) cluster centres in original scale
      k            — chosen K
      seed         — chosen seed
      balance      — min_size / max_size of chosen solution
      sizes        — list of per-cluster building counts
      reason       — human-readable selection rationale
    """
    if k_candidates is None:
        k_candidates = [4, 5]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    best: Optional[Dict] = None
    best_score = -np.inf
    all_candidates: List[Dict] = []

    for k in k_candidates:
        for trial in range(retries):
            seed = cluster_seed + trial
            km   = KMeans(n_clusters=k, random_state=seed, n_init=10)
            labels = km.fit_predict(X_scaled)
            score  = _balance_score(labels, k)
            centers_orig = scaler.inverse_transform(km.cluster_centers_)
            sizes  = [int((labels == c).sum()) for c in range(k)]

            candidate = dict(
                assignments  = labels.copy(),
                centers_orig = centers_orig,
                k            = k,
                seed         = seed,
                balance      = score,
                sizes        = sizes,
            )
            all_candidates.append(candidate)

            if score > best_score:
                best_score = score
                best       = candidate

    assert best is not None

    # Build rationale string
    sizes_str = ", ".join(f"group {c}: {s} buildings"
                          for c, s in enumerate(best["sizes"]))
    reason = (
        f"Chose K={best['k']} seed={best['seed']} "
        f"(balance={best['balance']:.3f}; {sizes_str}). "
        f"Tried {len(k_candidates)} K values × {retries} seeds = "
        f"{len(all_candidates)} total candidates. "
        f"Balance = min_cluster_size / max_cluster_size."
    )
    best["reason"] = reason

    return best


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_cluster_artifacts(
    result:      Dict,
    features:    np.ndarray,
    save_dir:    Path,
) -> Dict[str, Path]:
    """Write cluster artifacts to *save_dir*.

    Files created
    -------------
    building_cluster_assignment.csv  — one row per building
    cluster_centers.csv              — one row per cluster centre (original scale)
    cluster_summary.json             — full summary + rationale

    Returns mapping artifact_name → Path.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    assignments  = result["assignments"]
    centers_orig = result["centers_orig"]
    k            = result["k"]
    sizes        = result["sizes"]
    reason       = result["reason"]

    n_buildings = len(assignments)

    # 1. Building assignment CSV
    assign_df = pd.DataFrame({
        "building_idx": list(range(n_buildings)),
        "cluster":      assignments.tolist(),
        "bes_capacity_kwh":  features[:, 0].tolist(),
        "hvac_total_kw":     features[:, 1].tolist(),
    })
    assign_path = save_dir / "building_cluster_assignment.csv"
    assign_df.to_csv(assign_path, index=False)

    # 2. Cluster centres CSV
    centers_df = pd.DataFrame(
        centers_orig,
        columns=["bes_capacity_kwh", "hvac_total_kw"],
    )
    centers_df.insert(0, "cluster", list(range(k)))
    centers_df.insert(1, "n_buildings", sizes)
    centers_path = save_dir / "cluster_centers.csv"
    centers_df.to_csv(centers_path, index=False)

    # 3. Summary JSON
    summary = {
        "k":           k,
        "seed":        result["seed"],
        "balance":     float(result["balance"]),
        "cluster_sizes": sizes,
        "cluster_centers_original_scale": {
            f"cluster_{c}": {
                "bes_capacity_kwh": float(centers_orig[c, 0]),
                "hvac_total_kw":    float(centers_orig[c, 1]),
                "n_buildings":      sizes[c],
            }
            for c in range(k)
        },
        "group_assignments": {
            f"building_{i}": int(assignments[i]) for i in range(n_buildings)
        },
        "selection_rationale": reason,
    }
    summary_path = save_dir / "cluster_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"  [cluster] assignment CSV → {assign_path}")
    print(f"  [cluster] centers CSV    → {centers_path}")
    print(f"  [cluster] summary JSON   → {summary_path}")

    return {
        "assignment": assign_path,
        "centers":    centers_path,
        "summary":    summary_path,
    }


# ---------------------------------------------------------------------------
# Convenience: run clustering and save in one call
# ---------------------------------------------------------------------------

def run_clustering(
    climate:       str,
    save_dir:      Path,
    n_buildings:   int  = 25,
    k_candidates:  List[int] = None,
    cluster_seed:  int  = 0,
    retries:       int  = 10,
    repo_dir:      Path = REPO_DIR,
) -> Tuple[np.ndarray, Dict]:
    """Extract features, cluster, save artifacts, and return (assignments, result).

    Returns
    -------
    assignments : np.ndarray (n_buildings,)  cluster label per building [0, K-1]
    result      : full clustering result dict
    """
    if k_candidates is None:
        k_candidates = [4, 5]

    print(f"\n[cluster] Extracting building features ({climate}, {n_buildings} buildings)…")
    features = extract_building_features(climate, n_buildings, repo_dir)
    print(f"  BES  range: [{features[:, 0].min():.2f}, {features[:, 0].max():.2f}] kWh")
    print(f"  HVAC range: [{features[:, 1].min():.2f}, {features[:, 1].max():.2f}] kW")

    print(f"[cluster] Running K-means: K∈{k_candidates}, {retries} seeds each…")
    result = select_best_clustering(features, k_candidates, cluster_seed, retries)

    print(f"[cluster] Best: K={result['k']} seed={result['seed']} "
          f"balance={result['balance']:.3f}")
    for c, s in enumerate(result["sizes"]):
        idx = np.where(result["assignments"] == c)[0].tolist()
        print(f"  Group {c}: {s} buildings → {idx}")

    save_cluster_artifacts(result, features, save_dir)
    return result["assignments"], result
