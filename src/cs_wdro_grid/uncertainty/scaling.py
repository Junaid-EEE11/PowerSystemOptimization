"""Forecast-specific conformal scaling and residual standardization."""

from __future__ import annotations

import numpy as np


def predictive_scale(lower: np.ndarray, upper: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Return max(interval half-width, floor)."""
    if floor <= 0:
        raise ValueError("Scale floor must be positive")
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("Invalid interval arrays")
    return np.maximum((hi - lo) / 2.0, floor)


def standardized_errors(actual: np.ndarray, point: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map physical forecast errors into calibrated standardized coordinates."""
    y = np.asarray(actual, dtype=float)
    forecast = np.asarray(point, dtype=float)
    denominator = np.asarray(scale, dtype=float)
    if not (y.shape == forecast.shape == denominator.shape):
        raise ValueError("Actual, point, and scale arrays must have identical shapes")
    if np.any(denominator <= 0):
        raise ValueError("All scales must be positive")
    return (y - forecast) / denominator
