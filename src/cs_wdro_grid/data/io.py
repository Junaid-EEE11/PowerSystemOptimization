"""Measured profile ingestion with strict schema checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_measured_profiles(load_path: str | Path, pv_path: str | Path) -> pd.DataFrame:
    """Load timestamped normalized load and PV factors without interpolation."""
    frames = []
    for path, column in ((load_path, "load_factor"), (pv_path, "pv_factor")):
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Required measured input is missing: {source}. See data/DATA_SOURCES.md."
            )
        frame = pd.read_csv(source)
        required = {"timestamp", column}
        if not required.issubset(frame.columns):
            raise ValueError(f"{source} must contain columns {sorted(required)}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        if frame["timestamp"].duplicated().any():
            raise ValueError(f"Duplicate timestamps in {source}")
        frames.append(frame.set_index("timestamp")[[column]].sort_index())
    profiles = frames[0].join(frames[1], how="inner", validate="one_to_one")
    if profiles.empty or profiles.isna().any().any():
        raise ValueError("Measured load/PV join is empty or contains missing values")
    if (profiles["load_factor"] < 0).any() or not profiles["pv_factor"].between(0, 1).all():
        raise ValueError("Load must be nonnegative and PV availability must lie in [0, 1]")
    return profiles
