"""Interpretable forecasting baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def persistence(series: pd.Series) -> pd.Series:
    """F0: use the most recent observed value."""
    return series.shift(1).rename("persistence")


def seasonal_naive(series: pd.Series, season: int = 96) -> pd.Series:
    """F1: use the corresponding observation from the previous season."""
    if season <= 0:
        raise ValueError("Season length must be positive")
    return series.shift(season).rename("seasonal_naive")


class LinearForecaster:
    """F2: regularized linear autoregression on supplied causal features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.model = Ridge(alpha=alpha)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LinearForecaster:
        self.model.fit(x, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(x), dtype=float)
