"""Wasserstein-1 worst-case CVaR helpers for affine loss."""

from __future__ import annotations

import numpy as np


def affine_worst_case_cvar(
    intercept: float,
    coefficient: np.ndarray,
    residuals: np.ndarray,
    alpha: float,
    epsilon: float,
) -> float:
    """Evaluate the tractable W1/L1 worst-case CVaR formula by scalar minimization.

    The ambiguity support is R^d, the ground metric is L1, and the violation loss is affine.
    The hinge Lipschitz constant is therefore the L-infinity norm of ``coefficient``.
    """
    from scipy.optimize import minimize_scalar

    samples = np.asarray(residuals, dtype=float)
    beta = np.asarray(coefficient, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != beta.size:
        raise ValueError("Residual bank and affine coefficient dimensions do not match")
    if not 0 < alpha < 1 or epsilon < 0:
        raise ValueError("Require alpha in (0,1) and epsilon >= 0")
    losses = intercept + samples @ beta
    penalty = epsilon * float(np.linalg.norm(beta, ord=np.inf))

    def objective(tau: float) -> float:
        return float(tau + (np.mean(np.maximum(losses - tau, 0.0)) + penalty) / (1 - alpha))

    span = max(1.0, float(np.ptp(losses)) + penalty / (1 - alpha))
    result = minimize_scalar(
        objective,
        bounds=(float(np.min(losses) - span), float(np.max(losses) + span)),
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"CVaR scalar minimization failed: {result.message}")
    return float(result.fun)
