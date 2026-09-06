"""Explicit controlled distribution-shift transformations."""

from __future__ import annotations

import numpy as np


def load_bias(actual_load: float, bias: float) -> float:
    """S1: increase realized load while leaving optimizer information unchanged."""
    if bias < 0:
        raise ValueError("Load bias must be nonnegative")
    return float(max(actual_load * (1 + bias), 0.0))


def scale_forecast_error(
    actual_load: float,
    actual_pv: float,
    forecast_load: float,
    forecast_pv: float,
    multiplier: float,
) -> tuple[float, float]:
    """S2/S3: magnify realized forecast errors with physical clipping."""
    if multiplier < 0:
        raise ValueError("Error multiplier must be nonnegative")
    shifted_load = forecast_load + multiplier * (actual_load - forecast_load)
    shifted_pv = forecast_pv + multiplier * (actual_pv - forecast_pv)
    return float(max(shifted_load, 0.0)), float(np.clip(shifted_pv, 0.0, 1.0))


def stress_conditions(
    actual_load: float,
    actual_pv: float,
    forecast_load: float,
    forecast_pv: float,
    load_biases: list[float],
    error_scales: list[float],
) -> dict[str, tuple[float, float]]:
    """Create named non-combined stress cases so methods receive identical realizations."""
    conditions = {"s0_nominal": (float(actual_load), float(actual_pv))}
    for bias in load_biases:
        if bias > 0:
            conditions[f"s1_load_bias_{bias:.2f}"] = (
                load_bias(actual_load, bias),
                float(actual_pv),
            )
    for multiplier in error_scales:
        if multiplier != 1:
            conditions[f"s2_error_scale_{multiplier:.2f}"] = scale_forecast_error(
                actual_load, actual_pv, forecast_load, forecast_pv, multiplier
            )
    conditions["s3_pv_overprediction_0.10"] = (
        float(actual_load),
        float(np.clip(actual_pv - 0.10, 0.0, 1.0)),
    )
    return conditions
