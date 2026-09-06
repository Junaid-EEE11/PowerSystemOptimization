"""Point and probabilistic forecast metrics."""

from __future__ import annotations

import numpy as np


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    """Mean quantile (pinball) loss."""
    y = np.asarray(actual, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if y.shape != estimate.shape or not 0 < quantile < 1:
        raise ValueError("Invalid arrays or quantile")
    error = y - estimate
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def forecast_metrics(
    actual: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    miscoverage: float = 0.10,
) -> dict[str, float]:
    """Compute accuracy, calibration, sharpness, and interval score together."""
    y = np.asarray(actual, dtype=float)
    pred = np.asarray(point, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == pred.shape == lo.shape == hi.shape) or np.any(lo > hi):
        raise ValueError("Forecast arrays must be aligned with valid interval ordering")
    error = y - pred
    width = hi - lo
    interval_score = width.copy()
    interval_score += 2 / miscoverage * np.maximum(lo - y, 0.0)
    interval_score += 2 / miscoverage * np.maximum(y - hi, 0.0)
    coverage = np.mean((y >= lo) & (y <= hi))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "coverage": float(coverage),
        "calibration_error": float(abs(coverage - (1 - miscoverage))),
        "mean_interval_width": float(np.mean(width)),
        "interval_score": float(np.mean(interval_score)),
        "pinball_lower": pinball_loss(y, lo, miscoverage / 2),
        "pinball_upper": pinball_loss(y, hi, 1 - miscoverage / 2),
    }
