import numpy as np
import pandas as pd

from mappo_grouping_variants.cluster import (
    _balanced_spectral_labels,
    select_best_grouping,
)
from tools.summarize_selected_experiment_metrics import _grouping_ablation_labels


def _synthetic_features(n_samples: int = 25) -> np.ndarray:
    rng = np.random.default_rng(7)
    centers = np.array(
        [
            [-3.0, -3.0],
            [-3.0, 3.0],
            [0.0, 0.0],
            [3.0, -3.0],
            [3.0, 3.0],
        ]
    )
    return np.vstack(
        [centers[index % len(centers)] + rng.normal(0.0, 0.15, 2) for index in range(n_samples)]
    )


def test_balanced_spectral_enforces_equal_sizes_for_25_buildings_and_5_groups():
    labels = _balanced_spectral_labels(_synthetic_features(), k=5, seed=0)

    assert sorted(np.bincount(labels, minlength=5).tolist()) == [5, 5, 5, 5, 5]


def test_balanced_spectral_is_reproducible_for_a_fixed_seed():
    features = _synthetic_features()

    first = _balanced_spectral_labels(features, k=5, seed=3)
    second = _balanced_spectral_labels(features, k=5, seed=3)

    np.testing.assert_array_equal(first, second)


def test_select_best_balanced_spectral_prefers_the_fully_balanced_k_candidate():
    features = _synthetic_features()
    frame = pd.DataFrame(
        {
            "building_idx": np.arange(len(features)),
            "feature_a": features[:, 0],
            "feature_b": features[:, 1],
        }
    )

    result = select_best_grouping(
        feature_df=frame,
        method="balanced_spectral",
        k_candidates=[4, 5],
        cluster_seed=0,
        retries=3,
    )

    assert result["k"] == 5
    assert result["sizes"] == [5, 5, 5, 5, 5]
    assert np.isclose(result["balance"], 1.0)
    assert result["method_details"]["affinity"] == "rbf_on_standardized_features"
    assert result["method_details"]["assignment"] == "capacity_constrained_linear_sum_assignment"


def test_summary_labels_balanced_spectral_as_the_capacity_load_5f_comparison():
    labels = _grouping_ablation_labels(
        "mappo_grouped_tarmac_hybrid_balanced_spectral_"
        "capacity_load_5f_linear_vt_500_seed42"
    )

    assert labels == {
        "architecture": "TarMAC hybrid",
        "grouping_method_short": "balanced_spectral",
        "grouping_feature_set_short": "capacity_load_5f",
        "feature_group": "compact fixed grouping",
    }
