import pytest

from cs_wdro_grid.experiments.stress import load_bias, scale_forecast_error, stress_conditions


def test_stress_transformations_leave_optimizer_information_out_of_scope() -> None:
    assert load_bias(1.0, 0.1) == pytest.approx(1.1)
    load, pv = scale_forecast_error(1.1, 0.2, 1.0, 0.4, 2.0)
    assert load == pytest.approx(1.2)
    assert pv == pytest.approx(0.0)
    conditions = stress_conditions(1.1, 0.2, 1.0, 0.4, [0.0, 0.1], [1.0, 2.0])
    assert set(conditions) == {
        "s0_nominal",
        "s1_load_bias_0.10",
        "s2_error_scale_2.00",
        "s3_pv_overprediction_0.10",
    }
