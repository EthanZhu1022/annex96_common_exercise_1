"""
CityLearn → RLlib MultiAgentEnv adapter.
==========================================

RLlib's multi-agent SAC pipeline expects an environment that satisfies
ray.rllib.env.multi_agent_env.MultiAgentEnv.  This adapter wraps a
CityLearnEnv (with NormalizedObservationWrapper) so RLlib can drive it.

Key design choices
------------------
- Each building is a separate agent, identified by a string key "building_i".
- The adapter exposes per-building observation/action spaces so RLlib can
  instantiate one independent policy per building.
- No communication is performed: each agent only receives its own observation.
- The episode terminates when CityLearn's terminated flag is set (full month).

Usage
-----
    from rllib_sac.env import CityLearnMultiAgentEnv
    env = CityLearnMultiAgentEnv({
        "climate":     "TX",
        "n_buildings": 25,
        "start_step":  5088,   # August
        "end_step":    5831,
    })
    obs, info = env.reset()
    # obs = {"building_0": array, ..., "building_24": array}
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Ensure local CityLearn copy takes precedence
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

# Sentinel used by RLlib to signal episode termination for all agents
_DONE_ALL = "__all__"


def _build_citylearn(
    climate:     str,
    n_buildings: int,
    start_step:  Optional[int] = None,
    end_step:    Optional[int] = None,
    repo_dir:    Path = REPO_DIR,
) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate CityLearnEnv + NormalizedObservationWrapper.

    Mirrors the build_env() helper in mappo/train.py and
    independent_sac/train.py so time-window handling is identical
    across all baselines.
    """
    dataset_name = f"annex96_ce1_{climate.lower()}_neighborhood"
    dataset_dir  = repo_dir / "data" / "datasets" / dataset_name
    schema_path  = dataset_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found: {schema_path}. "
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


class CityLearnMultiAgentEnv(MultiAgentEnv):
    """RLlib MultiAgentEnv wrapper around CityLearnEnv.

    Config dict keys
    ----------------
    climate      : str  — "TX" or "VT" (default "TX")
    n_buildings  : int  — number of buildings to include (default 25)
    start_step   : int  — simulation_start_time_step (default: schema default)
    end_step     : int  — simulation_end_time_step
    seed         : int  — RNG seed (default 42)

    Each agent is named "building_i" where i is the 0-based building index.
    Agents that are done (terminal) in a step return True in the done dict;
    "__all__" is set True when the episode ends (CityLearn terminates).
    """

    def __init__(self, env_config: Dict[str, Any]) -> None:
        super().__init__()

        self._climate     = env_config.get("climate",     "TX")
        self._n_buildings = env_config.get("n_buildings", 25)
        self._start_step  = env_config.get("start_step",  None)
        self._end_step    = env_config.get("end_step",    None)
        self._seed        = env_config.get("seed",        42)

        self._env, self._base_env = _build_citylearn(
            climate     = self._climate,
            n_buildings = self._n_buildings,
            start_step  = self._start_step,
            end_step    = self._end_step,
        )

        n = self._n_buildings
        self._agent_ids: List[str] = [f"building_{i}" for i in range(n)]

        # Per-agent observation and action spaces.
        # NormalizedObservationWrapper declares obs bounds as [0, 1] but
        # normalized values can go slightly outside (e.g. negative).
        # RLlib's preprocessor strictly enforces the declared bounds, so we
        # widen to [-5, 5] — large enough to contain all real observations
        # while still informing RLlib of the correct dimensionality.
        def _unbounded_obs(base_space: spaces.Box) -> spaces.Box:
            shape = base_space.shape
            return spaces.Box(
                low=-5.0, high=5.0, shape=shape, dtype=np.float32
            )

        self.observation_space = spaces.Dict({
            aid: _unbounded_obs(self._env.observation_space[i])
            for i, aid in enumerate(sorted(
                [f"building_{j}" for j in range(self._n_buildings)],
                key=lambda s: int(s.split("_")[1])
            ))
        })
        self.action_space = spaces.Dict({
            aid: self._env.action_space[i]
            for i, aid in enumerate(self._agent_ids)
        })

        # RLlib requires _agent_ids as a set
        self._agent_ids = set(self._agent_ids)

        # Sorted list for deterministic ordering when building env_actions
        self._sorted_agents = sorted(self._agent_ids,
                                     key=lambda s: int(s.split("_")[1]))

        self._obs_list: Optional[List] = None

    # ------------------------------------------------------------------
    # MultiAgentEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        use_seed = seed if seed is not None else self._seed
        obs_list, _ = self._env.reset(seed=use_seed)
        self._obs_list = obs_list
        obs_dict = {
            f"building_{i}": np.clip(
                np.array(obs, dtype=np.float32), -5.0, 5.0
            )
            for i, obs in enumerate(obs_list)
        }
        return obs_dict, {}

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],   # obs
        Dict[str, float],        # rewards
        Dict[str, bool],         # terminated
        Dict[str, bool],         # truncated
        Dict[str, Any],          # info
    ]:
        # Build CityLearn-format actions: List[List[float]], ordered by building index
        env_actions: List[List[float]] = []
        for i, agent_id in enumerate(self._sorted_agents):
            raw = action_dict.get(agent_id, np.zeros(
                self._env.action_space[i].shape, dtype=np.float32
            ))
            # Scale from tanh range (-1,1) → actual action-space bounds
            lo  = self._env.action_space[i].low.astype(np.float32)
            hi  = self._env.action_space[i].high.astype(np.float32)
            scaled = lo + (raw + 1.0) * 0.5 * (hi - lo)
            scaled = np.clip(scaled, lo, hi)
            env_actions.append(scaled.tolist())

        next_obs_list, rewards, terminated, truncated, info = self._env.step(env_actions)
        self._obs_list = next_obs_list

        episode_done = bool(terminated or truncated)

        obs_d: Dict[str, np.ndarray] = {}
        rew_d: Dict[str, float]      = {}
        ter_d: Dict[str, bool]       = {}
        tru_d: Dict[str, bool]       = {}

        for i, agent_id in enumerate(self._sorted_agents):
            obs_d[agent_id] = np.clip(
                np.array(next_obs_list[i], dtype=np.float32), -5.0, 5.0
            )
            rew_d[agent_id] = float(rewards[i])
            ter_d[agent_id] = episode_done
            tru_d[agent_id] = False

        ter_d[_DONE_ALL] = episode_done
        tru_d[_DONE_ALL] = False

        return obs_d, rew_d, ter_d, tru_d, {}

    # ------------------------------------------------------------------
    # Accessors used by train.py for KPI extraction after each episode
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> CityLearnEnv:
        """Underlying CityLearnEnv (unwrapped) for KPI extraction."""
        return self._base_env

    def get_agent_ids(self):
        return self._agent_ids
