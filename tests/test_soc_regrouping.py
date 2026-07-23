import numpy as np
import pandas as pd

from mappo_grouped_tarmac_soc_regrouping.features import compute_soc_statistics


def test_compute_soc_statistics_uses_normalized_soc_and_daily_ranges():
    values = np.concatenate(
        [
            np.linspace(0.0, 1.0, 24),
            np.linspace(0.2, 0.8, 24),
        ]
    )
    trajectory = pd.DataFrame(
        {
            "building_idx": np.zeros(values.size, dtype=int),
            "building_name": ["Building_0"] * values.size,
            "sample_index": np.arange(values.size),
            "electrical_storage_soc": values,
        }
    )

    row = compute_soc_statistics(trajectory).iloc[0]

    assert row["n_soc_samples"] == 48
    assert np.isclose(row["soc_mean"], values.mean())
    assert np.isclose(row["soc_std"], values.std(ddof=0))
    assert np.isclose(row["soc_q10"], np.quantile(values, 0.10))
    assert np.isclose(row["soc_low_fraction"], np.mean(values < 0.1))
    assert np.isclose(row["soc_high_fraction"], np.mean(values > 0.9))
    assert np.isclose(row["soc_daily_range_mean"], np.mean([1.0, 0.6]))
