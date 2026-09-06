import numpy as np
import pytest

pytest.importorskip("cvxpy")
pytest.importorskip("clarabel")

from cs_wdro_grid.networks.ieee33 import load_ieee33
from cs_wdro_grid.networks.profiles import create_pv_fleet
from cs_wdro_grid.optimization.opf import solve_volt_var

WEIGHTS = {"loss": 1.0, "voltage": 20.0, "curtailment": 100.0, "reactive": 0.1}


def test_deterministic_opf_is_feasible_and_respects_inverter_capability() -> None:
    network = load_ieee33()
    fleet = create_pv_fleet(network, [6, 13, 18, 25, 30], 0.4, 1.1)
    result = solve_volt_var(network, fleet, 0.65, 0.5, "m1", WEIGHTS)
    assert result.successful, result.status
    assert np.min(result.predicted_voltage_squared_pu) >= 0.95**2 - 1e-6
    q_limit = np.sqrt(np.square(fleet.inverter_rating_pu) - np.square(fleet.capacity_pu))
    assert np.all(np.abs(result.q_injection_pu) <= q_limit + 1e-7)
    assert np.allclose(result.curtailment_pu, 0.0, atol=1e-7)


def test_zero_pv_penetration_removes_pv_control_capacity() -> None:
    network = load_ieee33()
    fleet = create_pv_fleet(network, [6, 13], 0.0, 1.1)
    result = solve_volt_var(network, fleet, 0.5, 0.8, "m1", WEIGHTS)
    assert result.successful
    assert np.allclose(result.q_injection_pu, 0.0, atol=1e-7)


def test_zero_error_empirical_cvar_matches_deterministic_solution() -> None:
    network = load_ieee33()
    fleet = create_pv_fleet(network, [6, 13, 18, 25, 30], 0.4, 1.1)
    deterministic = solve_volt_var(network, fleet, 0.65, 0.5, "m1", WEIGHTS)
    empirical_cvar = solve_volt_var(
        network,
        fleet,
        0.65,
        0.5,
        "m4",
        WEIGHTS,
        scale=np.array([0.1, 0.1]),
        residuals=np.zeros((20, 2)),
    )
    assert deterministic.successful and empirical_cvar.successful
    np.testing.assert_allclose(
        empirical_cvar.q_injection_pu,
        deterministic.q_injection_pu,
        atol=1e-7,
    )
    assert empirical_cvar.objective == pytest.approx(deterministic.objective, abs=1e-7)
