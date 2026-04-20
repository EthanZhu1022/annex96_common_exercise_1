"""
CityLearn → on-policy-main MAPPO adapter.
==========================================

Provides a thin numpy-based wrapper (NOT an RLlib env) that bridges CityLearnEnv
into the tensor/buffer format expected by on-policy-main's R_MAPPOPolicy and
SharedReplayBuffer.

Key design choices
------------------
- Action space convention : normalized (-1, 1), scaled to native CityLearn bounds
  inside step() — identical to the convention used across all other baselines
  (rllib_independent_ppo, rllib_sac, mappo).
  Formula: scaled = low + (raw + 1) * 0.5 * (high - low)

- Centralized observation : concatenation of all N=25 buildings' local observations.
  Each agent's share_obs row is the same full-portfolio concat vector.
  This is the standard MAPPO "full sharing" centralized critic input.

- No RLlib dependency : env.py uses only CityLearnEnv + NormalizedObservationWrapper.
  reset() / step() return numpy arrays directly for use with SharedReplayBuffer.

- gymnasium spaces  : obs_space, act_space, cent_obs_space properties are provided
  so that R_MAPPOPolicy can construct the correct actor/critic network dimensions.

Usage
-----
    from mappo_standard.env import CityLearnMAPPOEnv
    env = CityLearnMAPPOEnv(climate="VT", n_buildings=25,
                            start_step=0, end_step=743)
    obs, share_obs = env.reset(seed=42)
    # obs.shape       == (25, obs_dim)
    # share_obs.shape == (25, 25 * obs_dim)
    actions = np.zeros((25, act_dim))  # in (-1, 1)
    obs, share_obs, rewards, done, info = env.step(actions)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper


# ---------------------------------------------------------------------------
# Internal builder (mirrors rllib_independent_ppo/env.py::_build_citylearn)
# ---------------------------------------------------------------------------

def _build_citylearn(
    climate:     str,
    n_buildings: int,
    start_step:  Optional[int],
    end_step:    Optional[int],
    repo_dir:    Path = REPO_DIR,
) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate CityLearnEnv wrapped with NormalizedObservationWrapper."""
    dataset_name = f"annex96_ce1_{climate.lower()}_neighborhood"
    dataset_dir  = repo_dir / "data" / "datasets" / dataset_name
    schema_path  = dataset_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found: {schema_path}.\n"
            f"Available: {[d.name for d in (repo_dir / 'data' / 'datasets').iterdir()]}"
        )

    kwargs: Dict = dict(
        schema=str(schema_path),
        root_directory=str(dataset_dir),
        central_agent=False,
        buildings=list(range(n_buildings)),
    )
    if start_step is not None and end_step is not None:
        window_len = end_step - start_step + 1
        kwargs["simulation_start_time_step"] = start_step
        kwargs["simulation_end_time_step"]   = end_step
        kwargs["episode_time_steps"]         = window_len

    base_env = CityLearnEnv(**kwargs)
    env      = NormalizedObservationWrapper(base_env)
    return env, base_env


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class CityLearnMAPPOEnv:
    """Thin numpy adapter for on-policy-main MAPPO training.

    Returns numpy arrays directly — NOT an RLlib MultiAgentEnv.
    Designed to feed trajectories into on-policy-main's SharedReplayBuffer.

    Parameters
    ----------
    climate      : "VT" or "TX"
    n_buildings  : number of buildings (default 25)
    start_step   : simulation_start_time_step in the CityLearn schema
    end_step     : simulation_end_time_step in the CityLearn schema
    """

    def __init__(
        self,
        climate:     str,
        n_buildings: int      = 25,
        start_step:  Optional[int] = None,
        end_step:    Optional[int] = None,
        repo_dir:    Path     = REPO_DIR,
    ) -> None:
        self._env, self._base_env = _build_citylearn(
            climate, n_buildings, start_step, end_step, repo_dir
        )
        self.n_agents  = n_buildings
        self.climate   = climate

        # Dimension shortcuts
        self.obs_dim      = self._env.observation_space[0].shape[0]
        self.act_dim      = self._env.action_space[0].shape[0]
        self.cent_obs_dim = self.obs_dim * n_buildings

        # Native action bounds for scaling (stored per-agent)
        self._act_low  = [
            self._env.action_space[i].low.astype(np.float32)
            for i in range(n_buildings)
        ]
        self._act_high = [
            self._env.action_space[i].high.astype(np.float32)
            for i in range(n_buildings)
        ]

    # ------------------------------------------------------------------
    # gymnasium spaces  — passed to R_MAPPOPolicy for network construction
    # ------------------------------------------------------------------

    @property
    def obs_space(self) -> spaces.Box:
        """Local observation space for actor network."""
        return spaces.Box(
            low=-5.0, high=5.0, shape=(self.obs_dim,), dtype=np.float32
        )

    @property
    def act_space(self) -> spaces.Box:
        """Normalized action space (-1, 1) for actor output."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32
        )

    @property
    def cent_obs_space(self) -> spaces.Box:
        """Centralized observation space for MAPPO critic.

        Each dimension is the concat of all N buildings' local obs.
        """
        return spaces.Box(
            low=-5.0, high=5.0, shape=(self.cent_obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Underlying CityLearn base env (for KPI extraction)
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> CityLearnEnv:
        return self._base_env

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset environment and return initial observations.

        Returns
        -------
        obs       : np.ndarray of shape (n_agents, obs_dim)
                    Each row is one building's local observation, clipped to [-5, 5].
        share_obs : np.ndarray of shape (n_agents, cent_obs_dim)
                    Each row is the concatenation of ALL buildings' obs (centralized).
        """
        obs_list, _ = self._env.reset(seed=seed)
        obs = self._clip_obs(obs_list)          # (n_agents, obs_dim)
        share_obs = self._make_share_obs(obs)    # (n_agents, cent_obs_dim)
        return obs, share_obs

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """Step the environment with normalized actions.

        Parameters
        ----------
        actions : np.ndarray of shape (n_agents, act_dim), values in (-1, 1).

        Returns
        -------
        obs       : (n_agents, obs_dim)
        share_obs : (n_agents, cent_obs_dim)
        rewards   : (n_agents,)
        done      : bool — True when episode ends
        info      : dict (empty; CityLearn info is available via base_env)
        """
        env_actions: List[List[float]] = []
        for i in range(self.n_agents):
            raw = np.clip(actions[i], -1.0, 1.0)
            lo, hi = self._act_low[i], self._act_high[i]
            scaled = lo + (raw + 1.0) * 0.5 * (hi - lo)
            scaled = np.clip(scaled, lo, hi)
            env_actions.append(scaled.tolist())

        next_obs_list, rewards_list, terminated, truncated, info = \
            self._env.step(env_actions)

        done = bool(terminated or truncated)
        next_obs   = self._clip_obs(next_obs_list)       # (n_agents, obs_dim)
        share_obs  = self._make_share_obs(next_obs)       # (n_agents, cent_obs_dim)
        rewards    = np.array(rewards_list, dtype=np.float32)  # (n_agents,)

        return next_obs, share_obs, rewards, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clip_obs(self, obs_list) -> np.ndarray:
        return np.stack([
            np.clip(np.array(o, dtype=np.float32), -5.0, 5.0)
            for o in obs_list
        ])  # (n_agents, obs_dim)

    def _make_share_obs(self, obs: np.ndarray) -> np.ndarray:
        """Build centralized obs: full portfolio concat, one copy per agent.

        Parameters
        ----------
        obs : (n_agents, obs_dim)

        Returns
        -------
        share_obs : (n_agents, cent_obs_dim)   cent_obs_dim = n_agents * obs_dim
        """
        flat = obs.flatten()                    # (n_agents * obs_dim,)
        return np.tile(flat, (self.n_agents, 1))  # (n_agents, cent_obs_dim)
