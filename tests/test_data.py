import numpy as np
import pandas as pd

from cs_wdro_grid.data.features import causal_features
from cs_wdro_grid.data.splits import chronological_slices, split_frame
from cs_wdro_grid.data.synthetic import SYNTHETIC_WATERMARK, generate_synthetic_profiles


def test_chronological_splits_are_ordered_complete_and_disjoint() -> None:
    frame = pd.DataFrame({"value": np.arange(101)})
    splits = chronological_slices(
        len(frame),
        {
            "forecast_train": 0.60,
            "residual_calibration": 0.15,
            "decision_calibration": 0.10,
            "test": 0.15,
        },
    )
    partitions = split_frame(frame, splits)
    combined = np.concatenate([part.index.to_numpy() for part in partitions.values()])
    assert np.array_equal(combined, np.arange(len(frame)))
    assert len(np.unique(combined)) == len(frame)
    starts = [part.index.min() for part in partitions.values()]
    assert starts == sorted(starts)


def test_lag_and_rolling_features_never_use_current_or_future_values() -> None:
    index = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    series = pd.Series(np.arange(20, dtype=float), index=index)
    features = causal_features(series, lags=(1, 4), rolling_windows=(4,))
    timestamp = index[10]
    assert features.loc[timestamp, "lag_1"] == 9
    assert features.loc[timestamp, "lag_4"] == 6
    assert features.loc[timestamp, "rolling_mean_4"] == np.mean([6, 7, 8, 9])
    changed = series.copy()
    changed.loc[timestamp:] = 1_000_000
    changed_features = causal_features(changed, lags=(1, 4), rolling_windows=(4,))
    pd.testing.assert_series_equal(features.loc[timestamp], changed_features.loc[timestamp])


def test_synthetic_profiles_are_bounded_watermarked_and_reproducible() -> None:
    first = generate_synthetic_profiles(800, seed=42)
    second = generate_synthetic_profiles(800, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert first["pv_factor"].between(0, 1).all()
    assert (first["load_factor"] >= 0).all()
    assert set(first["data_classification"]) == {SYNTHETIC_WATERMARK}
