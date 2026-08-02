from annex96_rewards.ce1 import CE1ThreeMetricReward, build_ce1_reward_kwargs


def _observation(*, temperature: float, month: int = 8):
    return {
        "month": month,
        "indoor_dry_bulb_temperature": temperature,
        "indoor_dry_bulb_temperature_heating_set_point": 22.0,
        "indoor_dry_bulb_temperature_cooling_set_point": 22.0,
        "comfort_band": 0.0,
        "net_electricity_consumption": 0.0,
        "district_load_target": 0.0,
    }


def _reward() -> CE1ThreeMetricReward:
    return CE1ThreeMetricReward(
        {"central_agent": False},
        weight_nmbe=0.0,
        weight_cv_rmse=0.0,
        weight_comfort=1.0,
        comfort_binary_weight=1.0,
        comfort_degree_weight=1.0,
        return_metadata=False,
    )


def test_seasonal_mode_uses_tx_22_to_26_band():
    assert _reward().calculate([_observation(temperature=25.0)]) == [0.0]


def test_tx_temperature_above_26_is_penalized():
    assert _reward().calculate([_observation(temperature=27.0)]) == [-2.0]


def test_reward_kwargs_do_not_expose_dynamic_bounds_mode():
    assert "comfort_bounds_mode" not in build_ce1_reward_kwargs()


def test_default_reward_weights_match_degree_hours_followup():
    kwargs = build_ce1_reward_kwargs()
    assert kwargs["weight_nmbe"] == 1.0
    assert kwargs["weight_cv_rmse"] == 1.0
    assert kwargs["weight_comfort"] == 1.5
    assert kwargs["comfort_binary_weight"] == 3.0
    assert kwargs["comfort_degree_weight"] == 1.5
