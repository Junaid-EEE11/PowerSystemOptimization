"""Causal one-step-ahead time-series feature construction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def causal_features(
    series: pd.Series,
    lags: Iterable[int] = (1, 4, 96, 672),
    rolling_windows: Iterable[int] = (4, 96),
) -> pd.DataFrame:
    """Build features at time t using observations no later than t-1."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("A DatetimeIndex is required")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("Series timestamps must be unique and sorted")
    features = pd.DataFrame(index=series.index)
    for lag in sorted(set(int(value) for value in lags)):
        if lag <= 0:
            raise ValueError("All lags must be positive")
        features[f"lag_{lag}"] = series.shift(lag)
    history = series.shift(1)
    for window in sorted(set(int(value) for value in rolling_windows)):
        if window <= 0:
            raise ValueError("Rolling windows must be positive")
        features[f"rolling_mean_{window}"] = history.rolling(window).mean()
        features[f"rolling_std_{window}"] = history.rolling(window).std(ddof=0)
    hour = series.index.hour.to_numpy() + series.index.minute.to_numpy() / 60.0
    day_of_year = series.index.dayofyear.to_numpy()
    features["hour"] = hour
    features["day_of_week"] = series.index.dayofweek
    features["month"] = series.index.month
    features["weekend"] = (series.index.dayofweek >= 5).astype(int)
    features["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    features["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    features["sin_year"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["cos_year"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return features


def supervised_frame(
    series: pd.Series,
    lags: Iterable[int] = (1, 4, 96, 672),
    rolling_windows: Iterable[int] = (4, 96),
) -> tuple[pd.DataFrame, pd.Series]:
    """Align causal features with the current target and discard unavailable histories."""
    features = causal_features(series, lags, rolling_windows)
    valid = features.notna().all(axis=1) & series.notna()
    return features.loc[valid], series.loc[valid]
