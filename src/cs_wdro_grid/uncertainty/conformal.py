"""Finite-sample split conformalized quantile regression."""

from __future__ import annotations

import math

import numpy as np


def nonconformity_scores(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Return CQR scores max(lower-y, y-upper), retaining negative in-interval scores."""
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("Actual and interval arrays must have identical shapes")
    if np.any(lo > hi):
        raise ValueError("Lower bounds may not exceed upper bounds")
    return np.maximum(lo - y, y - hi)


def finite_sample_quantile(scores: np.ndarray, miscoverage: float) -> float:
    """Compute the split-conformal order statistic ceil((n+1)(1-alpha))."""
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be non-empty and finite")
    if not 0 < miscoverage < 1:
        raise ValueError("Miscoverage must lie strictly inside (0, 1)")
    rank = min(values.size, math.ceil((values.size + 1) * (1 - miscoverage)))
    return float(np.partition(values, rank - 1)[rank - 1])


def conformalize(
    lower: np.ndarray, upper: np.ndarray, calibration_quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    """Expand raw quantile bounds by the calibrated CQR order statistic."""
    lo = np.asarray(lower, dtype=float) - calibration_quantile
    hi = np.asarray(upper, dtype=float) + calibration_quantile
    return lo, hi
