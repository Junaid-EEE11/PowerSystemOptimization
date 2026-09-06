"""Balanced radial nonlinear AC power flow using backward/forward sweep."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs_wdro_grid.networks.model import RadialNetwork


@dataclass(frozen=True)
class ACPowerFlowResult:
    converged: bool
    iterations: int
    voltage_complex_pu: np.ndarray
    branch_current_pu: np.ndarray
    loss_pu: float

    @property
    def voltage_magnitude_pu(self) -> np.ndarray:
        return np.abs(self.voltage_complex_pu)


def solve_ac_power_flow(
    network: RadialNetwork,
    net_p_pu: np.ndarray,
    net_q_pu: np.ndarray,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> ACPowerFlowResult:
    """Solve constant-power radial AC equations; positive power denotes consumption."""
    p = np.asarray(net_p_pu, dtype=float)
    q = np.asarray(net_q_pu, dtype=float)
    if p.shape != (network.n_bus,) or q.shape != (network.n_bus,):
        raise ValueError(f"Nodal vectors must have shape ({network.n_bus},)")
    voltage = np.ones(network.n_bus, dtype=complex)
    current_branch = np.zeros(network.n_branch, dtype=complex)
    ordered = network.topological_branches()
    children_edges: list[list[int]] = [[] for _ in range(network.n_bus)]
    for edge in range(network.n_branch):
        children_edges[int(network.branch_from[edge])].append(edge)
    demand = p + 1j * q
    converged = False
    iteration_count = 0
    for iteration in range(1, max_iterations + 1):
        iteration_count = iteration
        previous = voltage.copy()
        safe_voltage = np.where(np.abs(voltage) > 1e-8, voltage, 1.0 + 0j)
        current_node = np.conj(demand / safe_voltage)
        for edge in reversed(ordered):
            child = int(network.branch_to[edge])
            current_branch[edge] = current_node[child] + sum(
                current_branch[item] for item in children_edges[child]
            )
        voltage[network.root] = 1.0 + 0j
        for edge in ordered:
            parent = int(network.branch_from[edge])
            child = int(network.branch_to[edge])
            impedance = network.resistance_pu[edge] + 1j * network.reactance_pu[edge]
            voltage[child] = voltage[parent] - impedance * current_branch[edge]
        if not np.isfinite(voltage).all() or np.min(np.abs(voltage)) < 0.2:
            break
        if np.max(np.abs(voltage - previous)) <= tolerance:
            converged = True
            break
    loss = float(np.sum(network.resistance_pu * np.square(np.abs(current_branch))))
    return ACPowerFlowResult(converged, iteration_count, voltage, current_branch.copy(), loss)
