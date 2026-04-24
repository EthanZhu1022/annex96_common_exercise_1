from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from citylearn.reward_function import RewardFunction

DEFAULT_CE1_REWARD_PATH = "annex96_rewards.ce1.CE1ThreeMetricReward"
_EPSILON = 1e-9


def build_ce1_reward_kwargs(
    weight_nmbe: float = 1.0,
    weight_cv_rmse: float = 1.0,
    weight_comfort: float = 0.8,
    comfort_binary_weight: float = 1.3,
    comfort_degree_weight: float = 0.3,
    return_metadata: bool = True,
) -> Dict[str, Any]:
    """Return default kwargs for the CE1 three-metric reward."""

    return {
        "weight_nmbe": weight_nmbe,
        "weight_cv_rmse": weight_cv_rmse,
        "weight_comfort": weight_comfort,
        "comfort_binary_weight": comfort_binary_weight,
        "comfort_degree_weight": comfort_degree_weight,
        "return_metadata": return_metadata,
    }


def get_ce1_reward_config(**overrides: Any) -> Tuple[str, Dict[str, Any]]:
    """Return a CityLearn-compatible reward path and kwargs pair."""

    kwargs = build_ce1_reward_kwargs()
    kwargs.update(overrides)
    return DEFAULT_CE1_REWARD_PATH, kwargs


class CE1ThreeMetricReward(RewardFunction):
    """Reward that blends the three CE1 primary metrics into a trainable surrogate.

    Components
    ----------
    1. |NMBE| surrogate:
       running absolute portfolio bias normalized by cumulative target load.
    2. CV-RMSE surrogate:
       running root-mean-square portfolio tracking error normalized by mean target load.
    3. Thermal comfort surrogate:
       per-building exceedance outside the CE1 seasonal comfort band.

    The tracking terms are portfolio-level and shared equally across buildings so
    every controller sees the same district objective. The comfort term stays
    building-local to preserve some credit assignment for independent agents.
    """

    def __init__(
        self,
        env_metadata: Mapping[str, Any],
        weight_nmbe: float = 1.0,
        weight_cv_rmse: float = 1.0,
        weight_comfort: float = 0.8,
        comfort_binary_weight: float = 1.3,
        comfort_degree_weight: float = 0.3,
        return_metadata: bool = True,
    ):
        super().__init__(env_metadata)
        self.weight_nmbe = float(weight_nmbe)
        self.weight_cv_rmse = float(weight_cv_rmse)
        self.weight_comfort = float(weight_comfort)
        self.comfort_binary_weight = float(comfort_binary_weight)
        self.comfort_degree_weight = float(comfort_degree_weight)
        self.return_metadata = bool(return_metadata)
        self.reset()

    def reset(self):
        self._step_count = 0
        self._cumulative_error = 0.0
        self._cumulative_squared_error = 0.0
        self._cumulative_reference = 0.0

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        if abs(float(denominator)) < _EPSILON:
            return 0.0
        return float(numerator / denominator)

    @staticmethod
    def _get_season_comfort_bounds(month: Optional[Union[int, float]]) -> Tuple[float, float]:
        if month is not None:
            try:
                month_i = int(round(float(month)))
            except (TypeError, ValueError):
                month_i = 1
        else:
            month_i = 1

        if month_i in {5, 6, 7, 8, 9}:
            return 22.0, 26.0

        return 20.0, 24.0

    def _resolve_reference_load(self, observations: List[Mapping[str, Union[int, float]]]) -> float:
        reference = observations[0].get("district_load_target")

        if reference is None:
            raise KeyError(
                "CE1ThreeMetricReward requires `district_load_target` in the observation dictionary."
            )

        return float(reference)

    def _compute_comfort_penalty(
        self,
        observations: List[Mapping[str, Union[int, float]]],
    ) -> Tuple[np.ndarray, float, float]:
        month = observations[0].get("month")
        lower_bound_c, upper_bound_c = self._get_season_comfort_bounds(month)
        penalties: List[float] = []
        exceed_flags: List[float] = []

        for obs in observations:
            indoor_temp = float(obs["indoor_dry_bulb_temperature"])
            below = max(lower_bound_c - indoor_temp, 0.0)
            above = max(indoor_temp - upper_bound_c, 0.0)
            exceed_degrees = below + above
            exceed_flag = 1.0 if exceed_degrees > 0.0 else 0.0
            penalty = self.weight_comfort * (
                self.comfort_binary_weight * exceed_flag
                + self.comfort_degree_weight * exceed_degrees
            )
            penalties.append(penalty)
            exceed_flags.append(exceed_flag)

        comfort_penalties = np.asarray(penalties, dtype=float)
        portfolio_exceedance_pct = float(np.mean(exceed_flags) * 100.0) if exceed_flags else 0.0
        mean_comfort_penalty = float(np.mean(comfort_penalties)) if comfort_penalties.size else 0.0
        return comfort_penalties, portfolio_exceedance_pct, mean_comfort_penalty

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> Union[List[float], Tuple[Any, ...]]:
        n_buildings = max(len(observations), 1)
        district_actual_load = float(
            sum(float(obs["net_electricity_consumption"]) for obs in observations)
        )
        district_reference_load = self._resolve_reference_load(observations)
        tracking_error = district_actual_load - district_reference_load

        self._step_count += 1
        self._cumulative_error += tracking_error
        self._cumulative_squared_error += tracking_error**2
        self._cumulative_reference += district_reference_load

        running_reference_mean = self._safe_divide(self._cumulative_reference, self._step_count)
        running_nmbe_ratio = self._safe_divide(
            self._cumulative_error,
            self._cumulative_reference,
        )
        running_cv_rmse_ratio = self._safe_divide(
            np.sqrt(self._cumulative_squared_error / self._step_count),
            running_reference_mean,
        )

        tracking_penalty = (
            self.weight_nmbe * abs(running_nmbe_ratio)
            + self.weight_cv_rmse * running_cv_rmse_ratio
        )
        shared_tracking_penalty = tracking_penalty / n_buildings
        comfort_penalties, portfolio_exceedance_pct, mean_comfort_penalty = (
            self._compute_comfort_penalty(observations)
        )
        shared_tracking_penalties = np.full(n_buildings, shared_tracking_penalty, dtype=float)
        reward_list = -(shared_tracking_penalties + comfort_penalties)

        if self.central_agent:
            reward = [float(np.sum(reward_list))]
        else:
            reward = reward_list.tolist()

        if not self.return_metadata:
            return reward

        reward_info = {
            "portfolio_actual_load": district_actual_load,
            "portfolio_reference_load": district_reference_load,
            "portfolio_tracking_error": tracking_error,
            "running_nmbe_pct": running_nmbe_ratio * 100.0,
            "running_cv_rmse_pct": running_cv_rmse_ratio * 100.0,
            "portfolio_exceedance_pct": portfolio_exceedance_pct,
            "tracking_penalty_total": float(tracking_penalty),
            "comfort_penalty_mean": mean_comfort_penalty,
        }

        return (
            reward,
            (-shared_tracking_penalties).tolist(),
            (-comfort_penalties).tolist(),
            reward_info,
        )
