"""
Utilities for Hierarchical MAPPO.

Contents:
  RolloutBuffer  — stores one episode of trajectories for all cluster agents.
  compute_gae    — Generalized Advantage Estimation (GAE-λ).
  extract_episode_kpis — parse the DataFrame returned by env.evaluate().
  get_soc_stats  — battery state-of-charge statistics across all buildings.
"""

from typing import Dict, Iterator, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """On-policy trajectory storage for K cluster agents.

    All cluster agents share the same global reward and the same centralized
    critic, so advantages and returns are stored only once (not per-agent).
    Per-agent data (actor inputs, actions, log probs) are stored in parallel
    lists indexed by agent index.

    Typical usage
    -------------
    buf = RolloutBuffer(n_agents=5)
    while not done:
        buf.add(actor_inputs, actions, log_probs, global_obs, reward, done, value)
    buf.compute_returns_and_advantages(last_value, gamma, gae_lambda)
    for batch in buf.get_minibatches(batch_size, device):
        ...  # PPO update
    buf.clear()
    """

    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self.clear()

    def clear(self) -> None:
        # Per-agent lists: shape [T] of np.ndarray
        self.actor_inputs: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.actions:      List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.log_probs:    List[List[float]]       = [[] for _ in range(self.n_agents)]
        # Shared across all agents
        self.global_obs: List[np.ndarray] = []
        self.rewards:    List[float]       = []
        self.dones:      List[bool]        = []
        self.values:     List[float]       = []
        # Filled by compute_returns_and_advantages
        self.advantages: Optional[np.ndarray] = None
        self.returns:    Optional[np.ndarray] = None

    def add(
        self,
        actor_inputs: List[np.ndarray],  # one per agent, shape (actor_input_dim_i,)
        actions:      List[np.ndarray],  # one per agent, shape (action_dim_i,)
        log_probs:    List[float],        # one per agent, scalar
        global_obs:   np.ndarray,         # shape (global_obs_dim,)
        reward:       float,              # global cooperative reward
        done:         bool,
        value:        float,              # V(global_obs) from centralized critic
    ) -> None:
        for i in range(self.n_agents):
            self.actor_inputs[i].append(actor_inputs[i].copy())
            self.actions[i].append(actions[i].copy())
            self.log_probs[i].append(log_probs[i])
        self.global_obs.append(global_obs.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE-λ advantages and discounted returns in-place.

        GAE(γ, λ) trades off bias vs. variance in the advantage estimate:
          δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
          A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}

        Returns are then R_t = A_t + V(s_t), used as critic regression targets.

        Args:
            last_value: Bootstrap value V(s_T) for the state after the last step.
                        Set to 0 if the episode terminated naturally.
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)

        # Append bootstrap value for the step after the rollout ends
        values_ext = np.array(self.values + [last_value], dtype=np.float32)
        rewards    = np.array(self.rewards, dtype=np.float32)
        dones      = np.array(self.dones,   dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            delta    = rewards[t] + gamma * values_ext[t + 1] * not_done - values_ext[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns    = advantages + values_ext[:T]   # R_t = A_t + V(s_t)

    def get_minibatches(
        self, batch_size: int, device: torch.device
    ) -> Iterator[Dict]:
        """Yield shuffled minibatches as dicts of tensors.

        Advantages are normalized (zero mean, unit std) within the full
        rollout before batching — a standard PPO stability trick.

        Yields dicts with keys:
          actor_inputs  : List[Tensor] of shape (B, actor_input_dim_i) per agent
          actions       : List[Tensor] of shape (B, action_dim_i) per agent
          old_log_probs : List[Tensor] of shape (B,) per agent
          global_obs    : Tensor of shape (B, global_obs_dim)
          advantages    : Tensor of shape (B,) — normalized
          returns       : Tensor of shape (B,) — critic regression targets
        """
        assert self.advantages is not None, "Call compute_returns_and_advantages first."
        T       = len(self.rewards)
        indices = np.random.permutation(T)

        # Normalize advantages over the full rollout for variance reduction
        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

        # Pre-convert to numpy arrays for fast indexing
        global_obs_arr    = np.array(self.global_obs,  dtype=np.float32)   # (T, G)
        returns_arr       = np.array(self.returns,      dtype=np.float32)   # (T,)
        actor_inputs_arr  = [np.array(self.actor_inputs[i], dtype=np.float32)
                             for i in range(self.n_agents)]                 # [(T, D_i)]
        actions_arr       = [np.array(self.actions[i],      dtype=np.float32)
                             for i in range(self.n_agents)]
        log_probs_arr     = [np.array(self.log_probs[i],    dtype=np.float32)
                             for i in range(self.n_agents)]

        for start in range(0, T, batch_size):
            idx = indices[start: start + batch_size]
            yield {
                "actor_inputs":  [torch.from_numpy(actor_inputs_arr[i][idx]).to(device)
                                   for i in range(self.n_agents)],
                "actions":       [torch.from_numpy(actions_arr[i][idx]).to(device)
                                   for i in range(self.n_agents)],
                "old_log_probs": [torch.from_numpy(log_probs_arr[i][idx]).to(device)
                                   for i in range(self.n_agents)],
                "global_obs":    torch.from_numpy(global_obs_arr[idx]).to(device),
                "advantages":    torch.from_numpy(adv_norm[idx]).to(device),
                "returns":       torch.from_numpy(returns_arr[idx]).to(device),
            }


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------

def extract_episode_kpis(env_unwrapped) -> Dict[str, Optional[float]]:
    """Extract scalar KPIs from the DataFrame returned by env.evaluate().

    env.evaluate() returns a DataFrame with columns:
        cost_function | value | name | level
    where level is 'district' or 'building'.

    We extract district-level aggregates (normalized w.r.t. the no-control
    baseline, so values < 1 indicate improvement over doing nothing).

    Returns a flat dict suitable for wandb.log().
    """
    try:
        df       = env_unwrapped.evaluate()
        district = (df[df["level"] == "district"]
                    .set_index("cost_function")["value"])

        def _get(key: str) -> Optional[float]:
            if key not in district.index:
                return None
            val = district[key]
            if val is None:
                return None
            val_f = float(val)
            return val_f if np.isfinite(val_f) else None

        return {
            "kpi/electricity_consumption": _get("electricity_consumption_total"),
            "kpi/carbon_emissions":        _get("carbon_emissions_total"),
            "kpi/cost":                    _get("cost_total"),
            "kpi/ramping":                 _get("ramping_average"),
            "kpi/daily_peak":              _get("daily_peak_average"),
            "kpi/all_time_peak":           _get("all_time_peak_average"),
            "kpi/load_factor":             _get("daily_one_minus_load_factor_average"),
        }
    except Exception as exc:
        print(f"[warn] KPI extraction failed: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Battery SOC statistics
# ---------------------------------------------------------------------------

def get_soc_stats(env_unwrapped) -> Dict[str, float]:
    """Return mean/min/max battery SOC across all buildings at current step."""
    soc_values = []
    for b in env_unwrapped.buildings:
        t = b.time_step
        try:
            soc = b.electrical_storage.soc[t]
            if soc is not None:
                soc_values.append(float(soc))
        except (IndexError, AttributeError, TypeError):
            pass

    if not soc_values:
        return {}
    return {
        "soc/mean": float(np.mean(soc_values)),
        "soc/min":  float(np.min(soc_values)),
        "soc/max":  float(np.max(soc_values)),
    }
