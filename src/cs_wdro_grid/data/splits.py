"""Leakage-resistant chronological split construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplits:
    """Four ordered, non-overlapping half-open index slices."""

    forecast_train: slice
    residual_calibration: slice
    decision_calibration: slice
    test: slice

    def as_dict(self) -> dict[str, slice]:
        return {
            "forecast_train": self.forecast_train,
            "residual_calibration": self.residual_calibration,
            "decision_calibration": self.decision_calibration,
            "test": self.test,
        }


def chronological_slices(length: int, fractions: dict[str, float]) -> ChronologicalSplits:
    """Allocate every row exactly once while preserving temporal order."""
    if length < 4:
        raise ValueError("At least four observations are required")
    names = ("forecast_train", "residual_calibration", "decision_calibration", "test")
    if abs(sum(float(fractions[name]) for name in names) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    first = int(length * fractions["forecast_train"])
    second = first + int(length * fractions["residual_calibration"])
    third = second + int(length * fractions["decision_calibration"])
    if not 0 < first < second < third < length:
        raise ValueError("Split fractions produce an empty partition")
    return ChronologicalSplits(
        slice(0, first), slice(first, second), slice(second, third), slice(third, length)
    )


def split_frame(frame: pd.DataFrame, splits: ChronologicalSplits) -> dict[str, pd.DataFrame]:
    """Return defensive copies of all chronological partitions."""
    return {name: frame.iloc[part].copy() for name, part in splits.as_dict().items()}
