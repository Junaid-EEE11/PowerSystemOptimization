"""Validated in-memory representation of a balanced radial feeder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RadialNetwork:
    """Per-unit radial feeder; branches are directed parent-to-child."""

    name: str
    base_mva: float
    base_kv: float
    branch_from: np.ndarray
    branch_to: np.ndarray
    resistance_pu: np.ndarray
    reactance_pu: np.ndarray
    base_load_p_pu: np.ndarray
    base_load_q_pu: np.ndarray
    root: int = 0

    def __post_init__(self) -> None:
        arrays = (
            self.branch_from,
            self.branch_to,
            self.resistance_pu,
            self.reactance_pu,
            self.base_load_p_pu,
            self.base_load_q_pu,
        )
        if any(np.asarray(item).ndim != 1 for item in arrays):
            raise ValueError("All network arrays must be one-dimensional")
        if self.base_mva <= 0 or self.base_kv <= 0:
            raise ValueError("Positive base MVA and kV are required")
        if not (
            len(self.branch_from)
            == len(self.branch_to)
            == len(self.resistance_pu)
            == len(self.reactance_pu)
        ):
            raise ValueError("Branch arrays have inconsistent lengths")
        if len(self.branch_from) != self.n_bus - 1:
            raise ValueError("A radial network must contain n_bus - 1 branches")
        if np.any(self.resistance_pu <= 0) or np.any(self.reactance_pu <= 0):
            raise ValueError("Branch impedances must be positive")
        self.validate_topology()

    @property
    def n_bus(self) -> int:
        return int(len(self.base_load_p_pu))

    @property
    def n_branch(self) -> int:
        return int(len(self.branch_from))

    @property
    def z_base_ohm(self) -> float:
        return self.base_kv**2 / self.base_mva

    def validate_topology(self) -> None:
        """Require one parent per non-root bus and root-reachable acyclic orientation."""
        buses = set(range(self.n_bus))
        if self.root not in buses:
            raise ValueError("Root bus lies outside the bus set")
        if set(map(int, self.branch_from)).difference(buses) or set(
            map(int, self.branch_to)
        ).difference(buses):
            raise ValueError("A branch references an unknown bus")
        if self.root in set(map(int, self.branch_to)):
            raise ValueError("The root bus may not have a parent")
        children = list(map(int, self.branch_to))
        if len(children) != len(set(children)):
            raise ValueError("Every non-root bus must have exactly one parent")
        if set(children) != buses.difference({self.root}):
            raise ValueError("Every non-root bus must appear exactly once as a child")
        reached = {self.root}
        remaining = set(range(self.n_branch))
        while remaining:
            ready = {edge for edge in remaining if int(self.branch_from[edge]) in reached}
            if not ready:
                raise ValueError("Branches are cyclic, disconnected, or incorrectly oriented")
            for edge in ready:
                reached.add(int(self.branch_to[edge]))
            remaining.difference_update(ready)
        if reached != buses:
            raise ValueError("Not every bus is reachable from the root")

    def branch_for_child(self) -> np.ndarray:
        """Map each bus to its incoming branch, with -1 at the root."""
        incoming = np.full(self.n_bus, -1, dtype=int)
        incoming[self.branch_to.astype(int)] = np.arange(self.n_branch)
        return incoming

    def topological_branches(self) -> list[int]:
        """Return branch indices ordered from the root toward leaves."""
        ordered: list[int] = []
        reached = {self.root}
        remaining = set(range(self.n_branch))
        while remaining:
            ready = sorted(edge for edge in remaining if int(self.branch_from[edge]) in reached)
            for edge in ready:
                ordered.append(edge)
                reached.add(int(self.branch_to[edge]))
            remaining.difference_update(ready)
        return ordered

    def downstream_matrix(self) -> np.ndarray:
        """Map nodal net demand to lossless downstream branch flows."""
        matrix = np.zeros((self.n_branch, self.n_bus), dtype=float)
        descendants: list[set[int]] = [{int(bus)} for bus in self.branch_to]
        for edge in reversed(self.topological_branches()):
            child = int(self.branch_to[edge])
            descendants[edge].add(child)
            for child_edge in range(self.n_branch):
                if int(self.branch_from[child_edge]) == child:
                    descendants[edge].update(descendants[child_edge])
            matrix[edge, list(descendants[edge])] = 1.0
        return matrix

    def path_matrix(self) -> np.ndarray:
        """Map branch voltage drops to buses along their unique root paths."""
        matrix = np.zeros((self.n_bus, self.n_branch), dtype=float)
        incoming = self.branch_for_child()
        for bus in range(self.n_bus):
            cursor = bus
            while cursor != self.root:
                edge = int(incoming[cursor])
                if edge < 0:
                    raise RuntimeError("Topology validation invariant failed")
                matrix[bus, edge] = 1.0
                cursor = int(self.branch_from[edge])
        return matrix
