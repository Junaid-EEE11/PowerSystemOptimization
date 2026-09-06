"""Clearly watermarked development-only time series."""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_WATERMARK = "SYNTHETIC DEVELOPMENT DATA - NOT FOR SCIENTIFIC CLAIMS"


def generate_synthetic_profiles(
    periods: int,
    frequency_minutes: int = 15,
    seed: int = 0,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Generate causal-looking load/PV factors solely for tests and smoke runs."""
    if periods < 96 * 8:
        raise ValueError("At least eight days are required for development profiles")
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq=f"{frequency_minutes}min", tz="UTC")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    day = np.arange(periods) * frequency_minutes / (24 * 60)

    morning = np.exp(-0.5 * ((hour - 8.0) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hour - 19.0) / 2.8) ** 2)
    weekly = 0.05 * np.cos(2 * np.pi * day / 7)
    load_sigma = 0.012 + 0.025 * evening
    load = 0.52 + 0.16 * morning + 0.30 * evening + weekly
    load += rng.normal(0.0, load_sigma, periods)
    load = np.clip(load, 0.25, 1.10)

    daylight = np.maximum(0.0, np.sin(np.pi * (hour - 6.0) / 12.0)) ** 1.7
    cloud = np.empty(periods)
    cloud[0] = 0.85
    innovations = rng.normal(0.0, 0.10, periods)
    for i in range(1, periods):
        cloud[i] = np.clip(0.92 * cloud[i - 1] + 0.08 * 0.82 + innovations[i], 0.15, 1.05)
    pv = np.clip(daylight * cloud, 0.0, 1.0)

    return pd.DataFrame(
        {
            "load_factor": load,
            "pv_factor": pv,
            "data_classification": SYNTHETIC_WATERMARK,
        },
        index=index,
    )
