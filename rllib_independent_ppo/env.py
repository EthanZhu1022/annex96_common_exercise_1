"""
CityLearn → RLlib MultiAgentEnv adapter for PPO.
==================================================

Identical to rllib_sac/env.py with one key difference: the action_space
exposed to RLlib is Box(-1, 1, ...) rather than the native CityLearn bounds.

Rationale
---------
All baselines in this repo use the same action-scaling convention:
    scaled = low + (raw + 1.0) * 0.5 * (high - low)
where `raw` ∈ (-1, 1) is the policy output.  For RLlib SAC this is natural
because SAC uses tanh-squashing.  For PPO we expose a (-1, 1) action space
so that RLlib's Gaussian policy operates over the same normalised range and
the same scaling logic applies unchanged.

Usage
-----
    from rllib_independent_ppo.env import CityLearnPPOEnv
    env = CityLearnPPOEnv({
        "climate":     "TX",
        "n_buildings": 25,
        "start_step":  5088,
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

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))

from annex96_rewards import DEFAULT_CE1_REWARD_PATH, build_ce1_reward_kwargs
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper

_DONE_ALL = "__all__"


def _build_citylearn(
    climate:     str,
    n_buildings: int,
    start_step:  Optional[int] = None,
    end_step:    Optional[int] = None,
    reward_function: Optional[str] = None,
    reward_function_kwargs: Optional[Dict[str, Any]] = None,
    repo_dir:    Path = REPO_DIR,
) -> Tuple[NormalizedObservationWrapper, CityLearnEnv]:
    """Instantiate CityLearnEnv + NormalizedObservationWrapper."""
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
        reward_function=reward_function or DEFAULT_CE1_REWARD_PATH,
        reward_function_kwargs=reward_function_kwargs or build_ce1_reward_kwargs(),
    )

    if start_step is not None and end_step is not None:
        window_len = end_step - start_step + 1
        kwargs["simulation_start_time_step"] = start_step
        kwargs["simulation_end_time_step"]   = end_step
        kwargs["episode_time_steps"]         = window_len

    base_env = CityLearnEnv(**kwargs)
    env      = NormalizedObservationWrapper(base_env)
    return env, base_env


class CityLearnPPOEnv(MultiAgentEnv):
    """RLlib MultiAgentEnv wrapper for PPO-based independent per-building agents.

    Exposes action_space as Box(-1, 1, action_dim) so RLlib's Gaussian policy
    operates in the normalised range expected by the shared action-scaling
    convention used across all baselines.

    Config dict keys
    ----------------
    climate      : str  — "TX" or "VT" (default "TX")
    n_buildings  : int  — number of buildings (default 25)
    start_step   : int  — simulation_start_time_step
    end_step     : int  — simulation_end_time_step
    seed         : int  — RNG seed (default 42)
    """

    def __init__(self, env_config: Dict[str, Any]) -> None:
        super().__init__()

        self._climate     = env_config.get("climate",     "TX")
        self._n_buildings = env_config.get("n_buildings", 25)
        self._start_step  = env_config.get("start_step",  None)
        self._end_step    = env_config.get("end_step",    None)
        self._seed        = env_config.get("seed",        42)
        self._reward_function = env_config.get("reward_function", DEFAULT_CE1_REWARD_PATH)
        self._reward_function_kwargs = env_config.get(
            "reward_function_kwargs",
            build_ce1_reward_kwargs(),
        )

        self._env, self._base_env = _build_citylearn(
            climate     = self._climate,
            n_buildings = self._n_buildings,
            start_step  = self._start_step,
            end_step    = self._end_step,
            reward_function=self._reward_function,
            reward_function_kwargs=self._reward_function_kwargs,
        )

        n = self._n_buildings
        sorted_agents = [f"building_{i}" for i in range(n)]

        # Observation space: widen to [-5, 5] to accommodate normalised values
        # that may slightly exceed [0, 1].
        def _unbounded_obs(base_space: spaces.Box) -> spaces.Box:
            return spaces.Box(
                low=-5.0, high=5.0, shape=base_space.shape, dtype=np.float32
            )

        # Action space: always (-1, 1) so PPO's Gaussian policy outputs are in
        # the range expected by the baseline action-scaling convention.
        def _normalised_act(base_space: spaces.Box) -> spaces.Box:
            return spaces.Box(
                low=-1.0, high=1.0, shape=base_space.shape, dtype=np.float32
            )

        self.observation_space = spaces.Dict({
            aid: _unbounded_obs(self._env.observation_space[i])
            for i, aid in enumerate(sorted_agents)
        })
        self.action_space = spaces.Dict({
            aid: _normalised_act(self._env.action_space[i])
            for i, aid in enumerate(sorted_agents)
        })

        # Store native action bounds for scaling inside step()
        self._act_low  = [self._env.action_space[i].low.astype(np.float32)  for i in range(n)]
        self._act_high = [self._env.action_space[i].high.astype(np.float32) for i in range(n)]

        self._agent_ids    = set(sorted_agents)
        self._sorted_agents = sorted_agents
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
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        # Scale from (-1, 1) → native CityLearn action bounds
        env_actions: List[List[float]] = []
        for i, agent_id in enumerate(self._sorted_agents):
            raw = action_dict.get(
                agent_id,
                np.zeros(self._env.action_space[i].shape, dtype=np.float32),
            )
            raw = np.clip(raw, -1.0, 1.0)
            lo, hi  = self._act_low[i], self._act_high[i]
            scaled  = lo + (raw + 1.0) * 0.5 * (hi - lo)
            scaled  = np.clip(scaled, lo, hi)
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
    # Accessors for KPI extraction after each episode
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> CityLearnEnv:
        return self._base_env

    def get_agent_ids(self):
        return self._agent_ids
