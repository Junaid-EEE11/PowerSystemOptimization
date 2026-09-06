"""Loss-approximated radial LinDistFlow calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs_wdro_grid.networks.model import RadialNetwork


@dataclass(frozen=True)
class LinearPowerFlowResult:
    p_branch_pu: np.ndarray
    q_branch_pu: np.ndarray
    voltage_squared_pu: np.ndarray
    loss_pu: float


def voltage_sensitivity(network: RadialNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Return dv/dp and dv/dq for positive nodal net consumption."""
    downstream = network.downstream_matrix()
    path = network.path_matrix()
    dp = -2.0 * path @ (network.resistance_pu[:, None] * downstream)
    dq = -2.0 * path @ (network.reactance_pu[:, None] * downstream)
    return dp, dq


def solve_lindistflow(
    network: RadialNetwork, net_p_pu: np.ndarray, net_q_pu: np.ndarray
) -> LinearPowerFlowResult:
    """Compute lossless branch balance, squared voltage, and quadratic loss proxy."""
    p = np.asarray(net_p_pu, dtype=float)
    q = np.asarray(net_q_pu, dtype=float)
    if p.shape != (network.n_bus,) or q.shape != (network.n_bus,):
        raise ValueError(f"Nodal vectors must have shape ({network.n_bus},)")
    downstream = network.downstream_matrix()
    p_branch = downstream @ p
    q_branch = downstream @ q
    drop = 2.0 * (network.resistance_pu * p_branch + network.reactance_pu * q_branch)
    voltage_squared = np.ones(network.n_bus) - network.path_matrix() @ drop
    loss = float(np.sum(network.resistance_pu * (np.square(p_branch) + np.square(q_branch))))
    return LinearPowerFlowResult(p_branch, q_branch, voltage_squared, loss)


def assert_power_balance(
    network: RadialNetwork, nodal: np.ndarray, branch_flows: np.ndarray, atol: float = 1e-10
) -> None:
    """Raise if lossless branch flows violate a nodal downstream balance identity."""
    values = np.asarray(nodal, dtype=float)
    flows = np.asarray(branch_flows, dtype=float)
    for edge in range(network.n_branch):
        child = int(network.branch_to[edge])
        outgoing = [i for i in range(network.n_branch) if int(network.branch_from[i]) == child]
        expected = values[child] + float(np.sum(flows[outgoing]))
        if not np.isclose(flows[edge], expected, atol=atol, rtol=0):
            raise AssertionError(
                f"Power balance failed on branch {edge}: {flows[edge]} != {expected}"
            )
