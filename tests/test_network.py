import numpy as np
import pytest

from cs_wdro_grid.networks.ieee33 import load_ieee33
from cs_wdro_grid.networks.lindistflow import (
    assert_power_balance,
    solve_lindistflow,
    voltage_sensitivity,
)
from cs_wdro_grid.networks.profiles import create_pv_fleet, operating_state


def test_ieee33_topology_units_and_base_totals() -> None:
    network = load_ieee33()
    assert network.n_bus == 33
    assert network.n_branch == 32
    network.validate_topology()
    assert network.z_base_ohm == pytest.approx(12.66**2 / 10.0)
    assert network.base_load_p_pu.sum() * network.base_mva == pytest.approx(3.715)
    assert network.base_load_q_pu.sum() * network.base_mva == pytest.approx(2.30)


def test_downstream_flows_obey_power_balance_and_root_voltage_is_one() -> None:
    network = load_ieee33()
    result = solve_lindistflow(network, network.base_load_p_pu, network.base_load_q_pu)
    assert_power_balance(network, network.base_load_p_pu, result.p_branch_pu)
    assert_power_balance(network, network.base_load_q_pu, result.q_branch_pu)
    assert result.voltage_squared_pu[network.root] == pytest.approx(1.0)
    assert np.min(result.voltage_squared_pu) < 1.0


def test_voltage_sensitivity_has_expected_sign_and_dimensions() -> None:
    network = load_ieee33()
    dp, dq = voltage_sensitivity(network)
    assert dp.shape == dq.shape == (33, 33)
    assert np.all(dp <= 1e-14)
    assert np.all(dq <= 1e-14)
    assert np.allclose(dp[0], 0)


def test_pv_penetration_definition_and_sign_convention() -> None:
    network = load_ieee33()
    fleet = create_pv_fleet(network, [6, 13, 18, 25, 30], 0.4)
    assert fleet.capacity_pu.sum() == pytest.approx(0.4 * network.base_load_p_pu.sum())
    no_pv, _ = operating_state(network, fleet, 1.0, 0.0)
    full_pv, _ = operating_state(network, fleet, 1.0, 1.0)
    assert full_pv.sum() < no_pv.sum()
