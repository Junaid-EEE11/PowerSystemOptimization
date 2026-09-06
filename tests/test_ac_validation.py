import numpy as np
import pytest

from cs_wdro_grid.networks.ieee33 import load_ieee33
from cs_wdro_grid.validation.ac import solve_ac_power_flow
from cs_wdro_grid.validation.metrics import voltage_metrics


def test_ieee33_base_case_nonlinear_ac_converges() -> None:
    network = load_ieee33()
    result = solve_ac_power_flow(network, network.base_load_p_pu, network.base_load_q_pu)
    assert result.converged
    assert result.iterations < 100
    # Independently cross-checked against pandapower case33bw with BFSW.
    assert np.min(result.voltage_magnitude_pu) == pytest.approx(0.91309048, abs=2e-7)
    assert result.loss_pu * network.base_mva == pytest.approx(0.20267713, abs=2e-7)


def test_voltage_metrics_report_all_required_reliability_views() -> None:
    voltages = np.array([[1.0, 0.94, 1.06], [1.0, 0.96, 1.04]])
    metrics = voltage_metrics(voltages)
    assert metrics["violation_event_rate"] == 0.5
    assert metrics["bus_time_violation_rate"] == 2 / 6
    assert metrics["worst_violation_magnitude"] == pytest.approx(0.01)
