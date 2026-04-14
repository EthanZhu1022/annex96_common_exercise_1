"""
CityLearn MAPPO environment adapter for mappo_grouped.
=======================================================

The environment interface is identical to mappo_standard; the only difference
is which actor network is used per building (handled in train.py, not here).

This module re-exports CityLearnMAPPOEnv and _build_citylearn directly from
mappo_standard.env to avoid code duplication.

Usage
-----
    from mappo_grouped_comm.env import CityLearnMAPPOEnv
    env = CityLearnMAPPOEnv(climate="VT", n_buildings=25,
                            start_step=0, end_step=743)
    obs, share_obs = env.reset(seed=42)
    # obs.shape       == (25, obs_dim)
    # share_obs.shape == (25, 25 * obs_dim)
    actions = np.zeros((25, act_dim))  # in (-1, 1)
    obs, share_obs, rewards, done, info = env.step(actions)
"""

from mappo_standard.env import CityLearnMAPPOEnv, _build_citylearn  # noqa: F401

__all__ = ["CityLearnMAPPOEnv", "_build_citylearn"]
