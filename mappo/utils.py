"""
Utilities for Hierarchical MAPPO.

Contents:
  RolloutBuffer - stores one episode of trajectories for all cluster agents.
  extract_episode_kpis - parse evaluation outputs and Annex96 reporting metrics.
  get_soc_stats - battery state-of-charge statistics across all buildings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch


class RolloutBuffer:
    """On-policy trajectory storage for K cluster agents."""

    def __init__(self, n_agents: int) -> None:
        self.n_agents = n_agents
        self.clear()

    def clear(self) -> None:
        self.actor_inputs: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.actions: List[List[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.log_probs: List[List[float]] = [[] for _ in range(self.n_agents)]
        self.global_obs: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(
        self,
        actor_inputs: List[np.ndarray],
        actions: List[np.ndarray],
        log_probs: List[float],
        global_obs: np.ndarray,
        reward: float,
        done: bool,
        value: float,
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
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        t_steps = len(self.rewards)
        advantages = np.zeros(t_steps, dtype=np.float32)

        values_ext = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(t_steps)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * values_ext[t + 1] * not_done - values_ext[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values_ext[:t_steps]

    def get_minibatches(self, batch_size: int, device: torch.device) -> Iterator[Dict[str, Any]]:
        assert self.advantages is not None, "Call compute_returns_and_advantages first."
        t_steps = len(self.rewards)
        indices = np.random.permutation(t_steps)

        adv_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        global_obs_arr = np.array(self.global_obs, dtype=np.float32)
        returns_arr = np.array(self.returns, dtype=np.float32)
        actor_inputs_arr = [
            np.array(self.actor_inputs[i], dtype=np.float32) for i in range(self.n_agents)
        ]
        actions_arr = [np.array(self.actions[i], dtype=np.float32) for i in range(self.n_agents)]
        log_probs_arr = [np.array(self.log_probs[i], dtype=np.float32) for i in range(self.n_agents)]

        for start in range(0, t_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "actor_inputs": [
                    torch.from_numpy(actor_inputs_arr[i][idx]).to(device) for i in range(self.n_agents)
                ],
                "actions": [
                    torch.from_numpy(actions_arr[i][idx]).to(device) for i in range(self.n_agents)
                ],
                "old_log_probs": [
                    torch.from_numpy(log_probs_arr[i][idx]).to(device) for i in range(self.n_agents)
                ],
                "global_obs": torch.from_numpy(global_obs_arr[idx]).to(device),
                "advantages": torch.from_numpy(adv_norm[idx]).to(device),
                "returns": torch.from_numpy(returns_arr[idx]).to(device),
            }


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _safe_percent_change(ratio_value: Optional[float]) -> Optional[float]:
    ratio_f = _safe_float(ratio_value)
    if ratio_f is None:
        return None
    return float((ratio_f - 1.0) * 100.0)


def _safe_percent_savings(ratio_value: Optional[float]) -> Optional[float]:
    ratio_f = _safe_float(ratio_value)
    if ratio_f is None:
        return None
    return float((1.0 - ratio_f) * 100.0)


def _safe_ratio_to_pct(ratio_value: Optional[float]) -> Optional[float]:
    ratio_f = _safe_float(ratio_value)
    if ratio_f is None:
        return None
    return float(ratio_f * 100.0)


def _safe_ratio_from_totals(actual: Optional[float], baseline: Optional[float]) -> Optional[float]:
    actual_f = _safe_float(actual)
    baseline_f = _safe_float(baseline)
    if actual_f is None or baseline_f is None or abs(baseline_f) <= 1e-12:
        return None
    return float(actual_f / baseline_f)


def _resolve_series(entity: Any, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for attr in candidates:
        series = getattr(entity, attr, None)
        if series is None:
            continue
        arr = np.asarray(series, dtype=float)
        if arr.size > 0:
            return arr
    return None


def resolve_reference_baseline_series(entity: Any) -> np.ndarray:
    arr = _resolve_series(
        entity,
        [
            "net_electricity_consumption_without_storage_and_partial_load_and_pv",
            "net_electricity_consumption_without_storage_and_pv",
            "net_electricity_consumption_without_storage_and_partial_load",
            "net_electricity_consumption_without_storage",
        ],
    )
    if arr is None:
        raise AttributeError("Unable to resolve baseline load series.")
    return arr


def resolve_reference_cost_series(entity: Any) -> Optional[np.ndarray]:
    return _resolve_series(
        entity,
        [
            "net_electricity_consumption_cost_without_storage_and_partial_load_and_pv",
            "net_electricity_consumption_cost_without_storage_and_pv",
            "net_electricity_consumption_cost_without_storage_and_partial_load",
            "net_electricity_consumption_cost_without_storage",
        ],
    )


def resolve_reference_emission_series(entity: Any) -> Optional[np.ndarray]:
    return _resolve_series(
        entity,
        [
            "net_electricity_consumption_emission_without_storage_and_partial_load_and_pv",
            "net_electricity_consumption_emission_without_storage_and_pv",
            "net_electricity_consumption_emission_without_storage_and_partial_load",
            "net_electricity_consumption_emission_without_storage",
        ],
    )


def get_episode_time_resolution(env_unwrapped: Any) -> Tuple[int, float]:
    seconds_per_step = float(getattr(env_unwrapped, "seconds_per_time_step", 3600) or 3600)
    steps_per_day = max(int(round(24 * 3600 / seconds_per_step)), 1)
    step_hours = seconds_per_step / 3600.0
    return steps_per_day, step_hours


def compute_daily_power_metrics(step_loads: Sequence[float], steps_per_day: int) -> Any:
    import pandas as pd

    arr = np.asarray(step_loads, dtype=float)
    n_days = len(arr) // steps_per_day
    rows: List[Dict[str, float]] = []
    for day_idx in range(n_days):
        start = day_idx * steps_per_day
        end = start + steps_per_day
        day = arr[start:end]
        peak = float(np.nanmax(day))
        low = float(np.nanmin(day))
        mean = float(np.nanmean(day))
        diffs = np.abs(np.diff(day))
        rows.append(
            {
                "day": day_idx + 1,
                "ramping": float(np.nansum(diffs)) if diffs.size > 0 else 0.0,
                "daily_peak": peak,
                "daily_min": low,
                "load_factor": mean / peak if peak > 1e-9 else float("nan"),
                "pvr": peak / low if low > 1e-9 else float("nan"),
                "energy": float(np.nansum(day)),
            }
        )
    return pd.DataFrame(rows)


def _compute_gini(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr)
    return float((n + 1.0 - 2.0 * float(np.sum(cum)) / total) / n)


def _compute_normalized_entropy(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return 1.0
    shares = arr / total
    shares = shares[shares > 0]
    if shares.size <= 1:
        return 1.0
    entropy = -float(np.sum(shares * np.log(shares)))
    return float(entropy / np.log(len(arr)))


def _format_peak_occurrence(env_unwrapped: Any, index: int, steps_per_day: int, step_hours: float) -> str:
    month_value: Optional[int] = None
    try:
        building = env_unwrapped.buildings[0]
        month_series = getattr(getattr(building, "energy_simulation", None), "month", None)
        if month_series is not None:
            episode_start = int(
                getattr(getattr(env_unwrapped, "episode_tracker", None), "episode_start_time_step", 0) or 0
            )
            month_arr = np.asarray(month_series, dtype=int)
            if episode_start + index < month_arr.size:
                month_value = int(month_arr[episode_start + index])
    except Exception:
        month_value = None

    day_index = int(index // steps_per_day) + 1 if steps_per_day > 0 else 1
    minutes_per_step = int(round(step_hours * 60.0))
    total_minutes = int((index % steps_per_day) * minutes_per_step)
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    if month_value is not None:
        return f"M{month_value:02d}-D{day_index:02d} {hour:02d}:{minute:02d}"
    return f"D{day_index:02d} {hour:02d}:{minute:02d}"


def _daily_summary(prefix: str, daily_df: Any) -> Dict[str, Optional[float]]:
    if getattr(daily_df, "empty", True):
        return {}
    return {
        f"{prefix}/ramping_mean": _safe_float(daily_df["ramping"].mean()),
        f"{prefix}/daily_peak_mean": _safe_float(daily_df["daily_peak"].mean()),
        f"{prefix}/load_factor_mean": _safe_float(daily_df["load_factor"].mean()),
        f"{prefix}/pvr_mean": _safe_float(daily_df["pvr"].mean()),
        f"{prefix}/energy_mean": _safe_float(daily_df["energy"].mean()),
    }


def extract_episode_kpis(env_unwrapped: Any) -> Dict[str, Any]:
    """Extract scalar KPIs and README-aligned secondary metrics."""
    try:
        df = env_unwrapped.evaluate()
        district = df[df["level"] == "district"].set_index("cost_function")["value"]

        def _get(key: str) -> Optional[float]:
            if key not in district.index:
                return None
            return _safe_float(district[key])

        actual_load = np.asarray(getattr(env_unwrapped, "net_electricity_consumption", []), dtype=float)
        baseline_load = resolve_reference_baseline_series(env_unwrapped)
        n_steps = min(len(actual_load), len(baseline_load))
        actual_load = actual_load[:n_steps]
        baseline_load = baseline_load[:n_steps]
        steps_per_day, step_hours = get_episode_time_resolution(env_unwrapped)

        actual_daily_df = compute_daily_power_metrics(actual_load, steps_per_day)
        baseline_daily_df = compute_daily_power_metrics(baseline_load, steps_per_day)

        actual_cost = np.asarray(getattr(env_unwrapped, "net_electricity_consumption_cost", []), dtype=float)
        baseline_cost = resolve_reference_cost_series(env_unwrapped)
        if baseline_cost is not None:
            cost_steps = min(len(actual_cost), len(baseline_cost))
            actual_cost = actual_cost[:cost_steps]
            baseline_cost = baseline_cost[:cost_steps]

        actual_emission = np.asarray(getattr(env_unwrapped, "net_electricity_consumption_emission", []), dtype=float)
        baseline_emission = resolve_reference_emission_series(env_unwrapped)
        if baseline_emission is not None:
            emission_steps = min(len(actual_emission), len(baseline_emission))
            actual_emission = actual_emission[:emission_steps]
            baseline_emission = baseline_emission[:emission_steps]

        peak_idx = int(np.nanargmax(actual_load)) if actual_load.size else 0
        baseline_peak_idx = int(np.nanargmax(baseline_load)) if baseline_load.size else 0

        flexibility_contrib: List[float] = []
        for building in getattr(env_unwrapped, "buildings", []):
            actual_building = np.asarray(getattr(building, "net_electricity_consumption", []), dtype=float)
            baseline_building = resolve_reference_baseline_series(building)
            building_steps = min(len(actual_building), len(baseline_building))
            if building_steps == 0:
                continue
            flexibility_contrib.append(
                float(np.nansum(np.abs(actual_building[:building_steps] - baseline_building[:building_steps])))
            )

        baseline_peak = _safe_float(np.nanmax(baseline_load)) if baseline_load.size else None
        actual_peak = _safe_float(np.nanmax(actual_load)) if actual_load.size else None
        peak_ratio = (
            float(actual_peak / baseline_peak)
            if actual_peak is not None and baseline_peak is not None and baseline_peak > 1e-9
            else None
        )

        result: Dict[str, Any] = {
            "kpi/electricity_consumption": _get("electricity_consumption_total"),
            "kpi/carbon_emissions": _get("carbon_emissions_total"),
            "kpi/cost": _get("cost_total"),
            "kpi/ramping": _get("ramping_average"),
            "kpi/daily_peak": _get("daily_peak_average"),
            "kpi/all_time_peak": _get("all_time_peak_average"),
            "kpi/load_factor": _get("daily_one_minus_load_factor_average"),
            "secondary/cost/normalized_ratio": _get("cost_total"),
            "secondary/cost/change_pct": _safe_percent_change(_get("cost_total")),
            "secondary/cost/absolute": _safe_float(np.nansum(actual_cost)) if actual_cost.size else None,
            "secondary/cost/baseline": _safe_float(np.nansum(baseline_cost)) if baseline_cost is not None else None,
            "secondary/carbon_emissions/normalized_ratio": _get("carbon_emissions_total"),
            "secondary/carbon_emissions/change_pct": _safe_percent_change(_get("carbon_emissions_total")),
            "secondary/carbon_emissions/absolute": (
                _safe_float(np.nansum(actual_emission)) if actual_emission.size else None
            ),
            "secondary/carbon_emissions/baseline": (
                _safe_float(np.nansum(baseline_emission)) if baseline_emission is not None else None
            ),
            "secondary/site_energy/normalized_ratio": _get("electricity_consumption_total"),
            "secondary/site_energy/change_pct": _safe_percent_change(_get("electricity_consumption_total")),
            "secondary/site_energy/absolute": _safe_float(np.nansum(actual_load)) if actual_load.size else None,
            "secondary/site_energy/baseline": _safe_float(np.nansum(baseline_load)) if baseline_load.size else None,
            "secondary/peak/flexible": actual_peak,
            "secondary/peak/baseline": baseline_peak,
            "secondary/peak/change_pct": _safe_percent_change(peak_ratio),
            "secondary/peak/flexible_time": (
                _format_peak_occurrence(env_unwrapped, peak_idx, steps_per_day, step_hours)
                if actual_load.size
                else None
            ),
            "secondary/peak/baseline_time": (
                _format_peak_occurrence(env_unwrapped, baseline_peak_idx, steps_per_day, step_hours)
                if baseline_load.size
                else None
            ),
            "secondary/fairness/flexibility_gini": _compute_gini(flexibility_contrib),
            "secondary/fairness/flexibility_entropy": _compute_normalized_entropy(flexibility_contrib),
            "secondary/fairness/max_share_pct": (
                float(100.0 * np.max(flexibility_contrib) / np.sum(flexibility_contrib))
                if flexibility_contrib and float(np.sum(flexibility_contrib)) > 1e-12
                else None
            ),
            **_daily_summary("secondary/daily/flexible", actual_daily_df),
            **_daily_summary("secondary/daily/baseline", baseline_daily_df),
        }
        cost_ratio = result["secondary/cost/normalized_ratio"]
        if cost_ratio is None:
            cost_ratio = _safe_ratio_from_totals(
                result["secondary/cost/absolute"],
                result["secondary/cost/baseline"],
            )
        carbon_ratio = result["secondary/carbon_emissions/normalized_ratio"]
        if carbon_ratio is None:
            carbon_ratio = _safe_ratio_from_totals(
                result["secondary/carbon_emissions/absolute"],
                result["secondary/carbon_emissions/baseline"],
            )
        site_energy_ratio = result["secondary/site_energy/normalized_ratio"]
        if site_energy_ratio is None:
            site_energy_ratio = _safe_ratio_from_totals(
                result["secondary/site_energy/absolute"],
                result["secondary/site_energy/baseline"],
            )
        result.update(
            {
                # README Secondary Metrics aliases. Legacy secondary/* paths above
                # are kept for backward compatibility with existing W&B panels.
                "secondary/cost_changes_pct": _safe_percent_savings(cost_ratio),
                "secondary/cost_change_pct": _safe_percent_change(cost_ratio),
                "secondary/cost_flexible": result["secondary/cost/absolute"],
                "secondary/cost_baseline": result["secondary/cost/baseline"],
                "secondary/carbon_emissions_kgco2e": result["secondary/carbon_emissions/absolute"],
                "secondary/carbon_emissions_baseline_kgco2e": result["secondary/carbon_emissions/baseline"],
                "secondary/carbon_emissions_change_pct": _safe_percent_change(carbon_ratio),
                "secondary/site_total_energy_change_pct": _safe_percent_change(site_energy_ratio),
                "secondary/site_total_energy_kwh": result["secondary/site_energy/absolute"],
                "secondary/site_total_energy_baseline_kwh": result["secondary/site_energy/baseline"],
                "secondary/peak_demand_kw": result["secondary/peak/flexible"],
                "secondary/peak_demand_baseline_kw": result["secondary/peak/baseline"],
                "secondary/peak_demand_change_pct": result["secondary/peak/change_pct"],
                "secondary/peak_demand_time": result["secondary/peak/flexible_time"],
                "secondary/peak_demand_baseline_time": result["secondary/peak/baseline_time"],
                "secondary/peak_to_valley_ratio_pct": _safe_ratio_to_pct(
                    result.get("secondary/daily/flexible/pvr_mean")
                ),
                "secondary/peak_to_valley_ratio_baseline_pct": _safe_ratio_to_pct(
                    result.get("secondary/daily/baseline/pvr_mean")
                ),
                "secondary/load_factor_pct": _safe_ratio_to_pct(
                    result.get("secondary/daily/flexible/load_factor_mean")
                ),
                "secondary/load_factor_baseline_pct": _safe_ratio_to_pct(
                    result.get("secondary/daily/baseline/load_factor_mean")
                ),
                "secondary/system_ramping_kw": result.get("secondary/daily/flexible/ramping_mean"),
                "secondary/system_ramping_baseline_kw": result.get("secondary/daily/baseline/ramping_mean"),
                "secondary/fairness_flexibility_gini": result["secondary/fairness/flexibility_gini"],
                "secondary/fairness_flexibility_entropy": result["secondary/fairness/flexibility_entropy"],
                "secondary/fairness_max_share_pct": result["secondary/fairness/max_share_pct"],
            }
        )
        return result
    except Exception as exc:
        print(f"[warn] KPI extraction failed: {exc}")
        return {}


def get_soc_stats(env_unwrapped: Any) -> Dict[str, float]:
    """Return mean/min/max battery SOC across all buildings at current step."""
    soc_values: List[float] = []
    for building in env_unwrapped.buildings:
        t_step = building.time_step
        try:
            soc = building.electrical_storage.soc[t_step]
            if soc is not None:
                soc_values.append(float(soc))
        except (IndexError, AttributeError, TypeError):
            pass

    if not soc_values:
        return {}
    return {
        "soc/mean": float(np.mean(soc_values)),
        "soc/min": float(np.min(soc_values)),
        "soc/max": float(np.max(soc_values)),
    }
