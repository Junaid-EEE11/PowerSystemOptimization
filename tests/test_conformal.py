import numpy as np
import pytest

from cs_wdro_grid.uncertainty.conformal import (
    conformalize,
    finite_sample_quantile,
    nonconformity_scores,
)
from cs_wdro_grid.uncertainty.scaling import predictive_scale, standardized_errors


def test_cqr_score_calculation_including_negative_scores() -> None:
    actual = np.array([0.0, 2.0, 5.0])
    lower = np.array([-1.0, 0.0, 3.0])
    upper = np.array([1.0, 1.0, 7.0])
    scores = nonconformity_scores(actual, lower, upper)
    np.testing.assert_allclose(scores, [-1.0, 1.0, -2.0])


def test_finite_sample_quantile_uses_ceiling_order_statistic() -> None:
    # n=9, alpha=.2 -> ceil(10*.8)=8, hence the eighth sorted value is 7.
    assert finite_sample_quantile(np.arange(9), miscoverage=0.2) == 7.0


def test_conformal_interval_and_standardization() -> None:
    lower, upper = conformalize(np.array([1.0]), np.array([3.0]), 0.5)
    np.testing.assert_allclose(lower, [0.5])
    np.testing.assert_allclose(upper, [3.5])
    scale = predictive_scale(lower, upper, floor=0.1)
    np.testing.assert_allclose(scale, [1.5])
    np.testing.assert_allclose(standardized_errors(np.array([3.5]), np.array([2.0]), scale), [1.0])


def test_wider_interval_increases_scale_and_floor_prevents_zero() -> None:
    narrow = predictive_scale(np.array([0.9, 1.0]), np.array([1.1, 1.0]), floor=0.05)
    wide = predictive_scale(np.array([0.5, 1.0]), np.array([1.5, 1.0]), floor=0.05)
    assert wide[0] > narrow[0]
    assert narrow[1] == pytest.approx(0.05)
