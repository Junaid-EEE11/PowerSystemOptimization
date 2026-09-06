import numpy as np
import pytest

from cs_wdro_grid.optimization.wdro import affine_worst_case_cvar


def test_worst_case_cvar_is_monotone_in_epsilon() -> None:
    residuals = np.array([[-1.0, 0.2], [0.0, -0.1], [1.0, 0.3], [2.0, -0.2]])
    coefficient = np.array([0.1, -0.4])
    values = [
        affine_worst_case_cvar(-0.2, coefficient, residuals, 0.75, radius)
        for radius in (0, 0.1, 0.5)
    ]
    assert values[0] <= values[1] <= values[2]


def test_zero_coefficient_has_no_wasserstein_radius_penalty() -> None:
    residuals = np.arange(10, dtype=float).reshape(5, 2)
    first = affine_worst_case_cvar(-0.3, np.zeros(2), residuals, 0.9, 0.0)
    second = affine_worst_case_cvar(-0.3, np.zeros(2), residuals, 0.9, 100.0)
    assert first == pytest.approx(second, abs=1e-5)
