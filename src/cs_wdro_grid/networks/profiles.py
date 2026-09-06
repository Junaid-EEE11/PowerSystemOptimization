"""Mapping of scalar load/PV factors to the feeder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs_wdro_grid.networks.model import RadialNetwork


@dataclass(frozen=True)
class PVFleet:
    buses: np.ndarray
    capacity_pu: np.ndarray
    inverter_rating_pu: np.ndarray

    def nodal_available(self, n_bus: int, pv_factor: float) -> np.ndarray:
        values = np.zeros(n_bus)
        values[self.buses] = self.capacity_pu * np.clip(pv_factor, 0.0, 1.0)
        return values


def create_pv_fleet(
    network: RadialNetwork,
    buses_one_based: list[int],
    penetration: float,
    inverter_oversize: float = 1.10,
) -> PVFleet:
    """Allocate capacity whose sum equals penetration times feeder peak base demand."""
    if penetration < 0 or inverter_oversize < 1:
        raise ValueError("PV penetration must be nonnegative and inverter oversize at least one")
    buses = np.asarray(buses_one_based, dtype=int) - 1
    if buses.size == 0 or np.any(buses <= network.root) or np.any(buses >= network.n_bus):
        raise ValueError("PV buses must be valid non-root one-based bus numbers")
    if len(np.unique(buses)) != len(buses):
        raise ValueError("PV bus numbers must be unique")
    weights = network.base_load_p_pu[buses]
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    total_capacity = penetration * float(network.base_load_p_pu.sum())
    capacity = total_capacity * weights / weights.sum()
    return PVFleet(buses, capacity, inverter_oversize * capacity)


def operating_state(
    network: RadialNetwork,
    fleet: PVFleet,
    load_factor: float,
    pv_factor: float,
    q_injection_pu: np.ndarray | None = None,
    curtailment_pu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build nodal net demand: load consumption minus generation/inverter injection."""
    p_load = network.base_load_p_pu * max(float(load_factor), 0.0)
    q_load = network.base_load_q_pu * max(float(load_factor), 0.0)
    pv = fleet.nodal_available(network.n_bus, pv_factor)
    q_inv = np.zeros(len(fleet.buses)) if q_injection_pu is None else np.asarray(q_injection_pu)
    curt = np.zeros(len(fleet.buses)) if curtailment_pu is None else np.asarray(curtailment_pu)
    if q_inv.shape != fleet.buses.shape or curt.shape != fleet.buses.shape:
        raise ValueError("Control vectors must match the number of PV buses")
    if np.any(curt < -1e-12) or np.any(curt - pv[fleet.buses] > 1e-10):
        raise ValueError("Curtailment must lie between zero and available PV")
    p_net = p_load - pv
    p_net[fleet.buses] += curt
    q_net = q_load.copy()
    q_net[fleet.buses] -= q_inv
    return p_net, q_net
