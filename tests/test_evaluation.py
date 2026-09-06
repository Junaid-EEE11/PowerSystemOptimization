import numpy as np
import pandas as pd
import pytest

from cs_wdro_grid.evaluation.comparisons import paired_method_comparisons
from cs_wdro_grid.evaluation.forecast import forecast_metrics, pinball_loss
from cs_wdro_grid.evaluation.statistics import paired_block_bootstrap_ci


def test_forecast_metrics_are_numerically_defined() -> None:
    actual = np.array([0.0, 1.0, 2.0])
    point = np.array([0.1, 0.9, 2.1])
    lower = np.array([-0.2, 0.5, 1.5])
    upper = np.array([0.2, 1.5, 2.5])
    metrics = forecast_metrics(actual, point, lower, upper)
    assert metrics["mae"] == pytest.approx(0.1)
    assert metrics["coverage"] == 1.0
    assert pinball_loss(actual, point, 0.5) >= 0


def test_paired_block_bootstrap_is_reproducible_and_paired() -> None:
    first = np.arange(20, dtype=float)
    second = first - 2.0
    a = paired_block_bootstrap_ci(first, second, block_length=4, replicates=100, seed=7)
    b = paired_block_bootstrap_ci(first, second, block_length=4, replicates=100, seed=7)
    assert a == b
    assert a["mean_difference"] == pytest.approx(2.0)
    assert a["ci_lower"] == pytest.approx(2.0)
    assert a["ci_upper"] == pytest.approx(2.0)


def test_method_comparison_counts_failure_as_unconditional_bad_event() -> None:
    rows = []
    for timestamp in range(4):
        rows.append(
            {
                "timestamp": timestamp,
                "condition": "s0_nominal",
                "pv_penetration": 0.4,
                "method": "m1",
                "successful": True,
                "violation_event": 0.0,
                "bus_time_violation_rate": 0.0,
            }
        )
        rows.append(
            {
                "timestamp": timestamp,
                "condition": "s0_nominal",
                "pv_penetration": 0.4,
                "method": "m5",
                "successful": timestamp != 0,
                "violation_event": 1.0 if timestamp == 0 else 0.0,
                "bus_time_violation_rate": np.nan if timestamp == 0 else 0.0,
            }
        )
    result = paired_method_comparisons(pd.DataFrame(rows), replicates=50, seed=1)
    unconditional = result[result["endpoint"] == "failure_or_voltage_violation_event"].iloc[0]
    conditional = result[
        result["endpoint"] == "bus_time_violation_rate_conditional_on_success"
    ].iloc[0]
    assert unconditional["mean_difference"] == pytest.approx(0.25)
    assert conditional["paired_observations"] == 3
