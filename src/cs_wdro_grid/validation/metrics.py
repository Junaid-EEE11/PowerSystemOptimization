"""Voltage reliability metrics defined on nonlinear AC magnitudes."""

from __future__ import annotations

import numpy as np


def voltage_metrics(
    voltages: np.ndarray, v_min: float = 0.95, v_max: float = 1.05
) -> dict[str, float]:
    """Calculate event, bus-time, and magnitude metrics without favorable selection."""
    values = np.asarray(voltages, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Voltages must be a non-empty time-by-bus array")
    magnitude = np.maximum(0.0, v_min - values) + np.maximum(0.0, values - v_max)
    violating = magnitude > 0
    return {
        "violation_event_rate": float(np.mean(np.any(violating, axis=1))),
        "bus_time_violation_rate": float(np.mean(violating)),
        "mean_violation_magnitude": float(np.mean(magnitude)),
        "worst_violation_magnitude": float(np.max(magnitude)),
        "minimum_voltage": float(np.min(values)),
        "maximum_voltage": float(np.max(values)),
    }
