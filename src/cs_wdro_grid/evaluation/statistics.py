"""Paired block-bootstrap uncertainty for temporally dependent outcomes."""

from __future__ import annotations

import numpy as np


def paired_block_bootstrap_ci(
    first: np.ndarray,
    second: np.ndarray,
    block_length: int,
    replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate mean(first-second) and a circular moving-block percentile interval."""
    a = np.asarray(first, dtype=float).reshape(-1)
    b = np.asarray(second, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Paired samples must be aligned, non-empty, and finite")
    if not 1 <= block_length <= len(a) or replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid block-bootstrap settings")
    differences = a - b
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(a) / block_length))
    estimates = np.empty(replicates)
    offsets = np.arange(block_length)
    for replicate in range(replicates):
        starts = rng.integers(0, len(a), size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % len(a)).reshape(-1)[: len(a)]
        estimates[replicate] = np.mean(differences[indices])
    tail = (1 - confidence) / 2
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_lower": float(np.quantile(estimates, tail)),
        "ci_upper": float(np.quantile(estimates, 1 - tail)),
        "confidence": confidence,
        "block_length": float(block_length),
    }
