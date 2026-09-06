"""Predefined ablation inputs; no locked-test tuning is performed here."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AblationSpec:
    name: str
    alpha: float
    epsilon: float
    bank_size: int | None
    use_conformal_scale: bool
    forecast_model: str


def required_ablation_grid(selected_epsilon: float) -> list[AblationSpec]:
    """Return A1–A6 settings declared before test evaluation."""
    grid = [
        AblationSpec("a1_no_conformal_scaling", 0.95, selected_epsilon, None, False, "boosting"),
        AblationSpec("a2_wdro_without_scaling", 0.95, selected_epsilon, None, False, "boosting"),
        AblationSpec("a3_conformal_epsilon_zero", 0.95, 0.0, None, True, "boosting"),
    ]
    grid.extend(
        AblationSpec(f"a4_alpha_{alpha:.2f}", alpha, selected_epsilon, None, True, "boosting")
        for alpha in (0.90, 0.95, 0.99)
    )
    grid.extend(
        AblationSpec(f"a5_bank_{size}", 0.95, selected_epsilon, size, True, "boosting")
        for size in (25, 50, 100, 250)
    )
    grid.extend(
        [
            AblationSpec("a5_bank_full", 0.95, selected_epsilon, None, True, "boosting"),
            AblationSpec("a6_seasonal_naive", 0.95, selected_epsilon, None, True, "seasonal_naive"),
            AblationSpec("a6_boosting", 0.95, selected_epsilon, None, True, "boosting"),
        ]
    )
    return grid


def fixed_physical_scale(errors: np.ndarray, floor: float) -> np.ndarray:
    """A1/A2 constant scale estimated only from residual-calibration errors."""
    values = np.asarray(errors, dtype=float)
    if values.ndim != 2:
        raise ValueError("Errors must be a sample-by-variable matrix")
    return np.maximum(np.std(values, axis=0, ddof=1), floor)
