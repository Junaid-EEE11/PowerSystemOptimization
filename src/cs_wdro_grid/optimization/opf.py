"""M0–M5 here-and-now continuous inverter Volt/VAR formulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
from typing import Any

import cvxpy as cp
import numpy as np

from cs_wdro_grid.networks.lindistflow import voltage_sensitivity
from cs_wdro_grid.networks.model import RadialNetwork
from cs_wdro_grid.networks.profiles import PVFleet

SUCCESS_STATUS = "optimal"


@dataclass(frozen=True)
class OPFResult:
    method: str
    status: str
    q_injection_pu: np.ndarray
    curtailment_pu: np.ndarray
    objective: float
    solve_time_seconds: float
    predicted_voltage_squared_pu: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        """Only a fully optimal result counts as successful."""
        return self.status == SUCCESS_STATUS


def _selection_matrix(n_bus: int, buses: np.ndarray) -> np.ndarray:
    matrix = np.zeros((n_bus, len(buses)))
    matrix[buses, np.arange(len(buses))] = 1.0
    return matrix


def _affine_state(
    network: RadialNetwork,
    fleet: PVFleet,
    load_factor: float,
    pv_factor: float,
    q_injection: cp.Expression,
    curtailment: cp.Expression,
) -> tuple[cp.Expression, cp.Expression, cp.Expression, cp.Expression, cp.Expression]:
    selection = _selection_matrix(network.n_bus, fleet.buses)
    pv_nodal = selection @ (fleet.capacity_pu * float(pv_factor))
    p_net = network.base_load_p_pu * float(load_factor) - pv_nodal + selection @ curtailment
    q_net = network.base_load_q_pu * float(load_factor) - selection @ q_injection
    downstream = network.downstream_matrix()
    path = network.path_matrix()
    p_branch = downstream @ p_net
    q_branch = downstream @ q_net
    voltage = 1.0 - 2.0 * path @ (
        cp.multiply(network.resistance_pu, p_branch) + cp.multiply(network.reactance_pu, q_branch)
    )
    return p_net, q_net, p_branch, q_branch, voltage


def _voltage_uncertainty_coefficients(
    network: RadialNetwork, fleet: PVFleet, load_scale: float, pv_scale: float
) -> np.ndarray:
    """Return bus-by-2 squared-voltage coefficients for standardized load/PV errors."""
    dp, dq = voltage_sensitivity(network)
    pv_capacity = np.zeros(network.n_bus)
    pv_capacity[fleet.buses] = fleet.capacity_pu
    load_coefficient = (dp @ network.base_load_p_pu + dq @ network.base_load_q_pu) * load_scale
    pv_coefficient = (dp @ (-pv_capacity)) * pv_scale
    return np.column_stack((load_coefficient, pv_coefficient))


def _scenario_factors(
    point_load: float,
    point_pv: float,
    scale: np.ndarray,
    residuals: np.ndarray,
) -> list[tuple[float, float]]:
    values = np.column_stack(
        (np.full(len(residuals), point_load), np.full(len(residuals), point_pv))
    )
    values = values + np.asarray(residuals, dtype=float) * np.asarray(scale, dtype=float)
    values[:, 0] = np.maximum(values[:, 0], 0.0)
    values[:, 1] = np.clip(values[:, 1], 0.0, 1.0)
    return [(float(load), float(pv)) for load, pv in values]


def solve_volt_var(
    network: RadialNetwork,
    fleet: PVFleet,
    point_load: float,
    point_pv: float,
    method: str,
    weights: dict[str, float],
    v_min: float = 0.95,
    v_max: float = 1.05,
    scale: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    alpha: float = 0.95,
    epsilon: float = 0.0,
    solver: str = "CLARABEL",
    scenario_count: int | None = None,
    allow_curtailment: bool = False,
) -> OPFResult:
    """Solve M0–M5 with common here-and-now controls and objective weights.

    Methods: ``m0`` reference, ``m1`` point deterministic, ``m2`` scenario stochastic,
    ``m3`` box vertices, ``m4`` empirical CVaR (epsilon zero), and ``m5`` CS-WDRO.
    """
    method = method.lower()
    if method not in {"m0", "m1", "m2", "m3", "m4", "m5"}:
        raise ValueError(f"Unknown decision method: {method}")
    n_pv = len(fleet.buses)
    zero = np.zeros(n_pv)
    _, _, _, _, no_control_v = _numeric_state(network, fleet, point_load, point_pv, zero, zero)
    if method == "m0":
        return OPFResult("m0", SUCCESS_STATUS, zero, zero, 0.0, 0.0, no_control_v)
    if method in {"m2", "m3", "m4", "m5"}:
        if scale is None or residuals is None:
            raise ValueError(f"{method} requires scale and a standardized residual bank")
        scale_array = np.asarray(scale, dtype=float)
        sample_array = np.asarray(residuals, dtype=float)
        if scale_array.shape != (2,) or sample_array.ndim != 2 or sample_array.shape[1] != 2:
            raise ValueError("Scale must have shape (2,) and residuals shape (n, 2)")
        if np.any(scale_array <= 0) or len(sample_array) == 0:
            raise ValueError("Uncertainty scales must be positive and residual bank nonempty")
    else:
        scale_array = np.ones(2)
        sample_array = np.zeros((1, 2))

    q = cp.Variable(n_pv, name="q_inv")
    curtailment = cp.Variable(n_pv, nonneg=True, name="p_curt")
    nominal_available = fleet.capacity_pu * float(np.clip(point_pv, 0.0, 1.0))
    constraints: list[cp.Constraint] = [curtailment <= nominal_available]
    if not allow_curtailment:
        constraints.append(curtailment == 0)
    # Exact nominal SOC capability. Realized capability is audited during AC validation.
    for i in range(n_pv):
        constraints.append(
            cp.norm(cp.hstack((q[i], nominal_available[i] - curtailment[i])), 2)
            <= fleet.inverter_rating_pu[i]
        )
        # This supplemental constant limit preserves capability for every realized PV
        # availability in [0, capacity]. It is conservative and documented as such.
        robust_q_limit = np.sqrt(
            max(fleet.inverter_rating_pu[i] ** 2 - fleet.capacity_pu[i] ** 2, 0.0)
        )
        constraints.extend((q[i] <= robust_q_limit, q[i] >= -robust_q_limit))

    _, _, p_nom, q_nom, v_nom = _affine_state(network, fleet, point_load, point_pv, q, curtailment)
    objective_terms = [
        float(weights["loss"])
        * cp.sum(cp.multiply(network.resistance_pu, cp.square(p_nom) + cp.square(q_nom))),
        float(weights["voltage"]) * cp.sum_squares(v_nom - 1.0),
        float(weights["curtailment"]) * cp.sum(curtailment),
        float(weights["reactive"]) * cp.sum_squares(q),
    ]
    objective = cp.sum(objective_terms)
    v_min_sq, v_max_sq = v_min**2, v_max**2

    scenario_total = 0
    if method == "m1":
        constraints.extend((v_nom >= v_min_sq, v_nom <= v_max_sq))
    elif method in {"m2", "m3"}:
        if method == "m2":
            chosen = sample_array[:scenario_count] if scenario_count else sample_array
        else:
            bounds = [
                (float(np.min(sample_array[:, j])), float(np.max(sample_array[:, j])))
                for j in range(2)
            ]
            chosen = np.asarray(list(product(*bounds)), dtype=float)
        factors = _scenario_factors(point_load, point_pv, scale_array, chosen)
        scenario_total = len(factors)
        scenario_objectives: list[cp.Expression] = []
        for scenario_load, scenario_pv in factors:
            _, _, p_flow, q_flow, voltage = _affine_state(
                network, fleet, scenario_load, scenario_pv, q, curtailment
            )
            constraints.extend((voltage >= v_min_sq, voltage <= v_max_sq))
            scenario_objectives.append(
                float(weights["loss"])
                * cp.sum(cp.multiply(network.resistance_pu, cp.square(p_flow) + cp.square(q_flow)))
                + float(weights["voltage"]) * cp.sum_squares(voltage - 1.0)
            )
        objective = (
            cp.sum(scenario_objectives) / scenario_total
            + float(weights["curtailment"]) * cp.sum(curtailment)
            + float(weights["reactive"]) * cp.sum_squares(q)
        )
    else:
        radius = 0.0 if method == "m4" else float(epsilon)
        if radius < 0 or not 0 < alpha < 1:
            raise ValueError("Require epsilon >= 0 and alpha in (0, 1)")
        coefficient = _voltage_uncertainty_coefficients(
            network, fleet, float(scale_array[0]), float(scale_array[1])
        )
        for bus in range(1, network.n_bus):
            for sign, boundary in ((1.0, v_max_sq), (-1.0, -v_min_sq)):
                # g = sign * v + boundary_offset; g <= 0 is safe.
                intercept = sign * v_nom[bus] - boundary
                beta = sign * coefficient[bus]
                tau = cp.Variable(name=f"tau_{bus}_{'upper' if sign > 0 else 'lower'}")
                sample_loss = intercept + sample_array @ beta
                wasserstein_penalty = radius * float(np.linalg.norm(beta, ord=np.inf))
                worst_cvar = tau + (
                    cp.sum(cp.pos(sample_loss - tau)) / len(sample_array) + wasserstein_penalty
                ) / (1.0 - alpha)
                constraints.append(worst_cvar <= 0)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    start = perf_counter()
    try:
        problem.solve(solver=solver, verbose=False)
        status = str(problem.status)
    except cp.error.SolverError as exc:
        status = f"failed:{type(exc).__name__}"
    elapsed = perf_counter() - start
    if status != SUCCESS_STATUS or q.value is None or curtailment.value is None:
        return OPFResult(
            method,
            status,
            np.full(n_pv, np.nan),
            np.full(n_pv, np.nan),
            np.nan,
            elapsed,
            np.full(network.n_bus, np.nan),
            {"epsilon": epsilon, "scenario_count": scenario_total},
        )
    q_value = np.asarray(q.value, dtype=float)
    curt_value = np.maximum(np.asarray(curtailment.value, dtype=float), 0.0)
    _, _, _, _, voltage_value = _numeric_state(
        network, fleet, point_load, point_pv, q_value, curt_value
    )
    return OPFResult(
        method=method,
        status=status,
        q_injection_pu=q_value,
        curtailment_pu=curt_value,
        objective=float(problem.value),
        solve_time_seconds=elapsed,
        predicted_voltage_squared_pu=voltage_value,
        metadata={"epsilon": float(epsilon), "scenario_count": scenario_total, "alpha": alpha},
    )


def _numeric_state(
    network: RadialNetwork,
    fleet: PVFleet,
    load_factor: float,
    pv_factor: float,
    q_injection: np.ndarray,
    curtailment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selection = _selection_matrix(network.n_bus, fleet.buses)
    p_net = (
        network.base_load_p_pu * float(load_factor)
        - selection @ (fleet.capacity_pu * float(pv_factor))
        + selection @ curtailment
    )
    q_net = network.base_load_q_pu * float(load_factor) - selection @ q_injection
    downstream = network.downstream_matrix()
    p_branch, q_branch = downstream @ p_net, downstream @ q_net
    voltage = 1.0 - 2.0 * network.path_matrix() @ (
        network.resistance_pu * p_branch + network.reactance_pu * q_branch
    )
    return p_net, q_net, p_branch, q_branch, voltage
