"""Predefined paired method comparisons with explicit failure handling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cs_wdro_grid.evaluation.statistics import paired_block_bootstrap_ci


def paired_method_comparisons(
    raw: pd.DataFrame,
    reference: str = "m1",
    candidate: str = "m5",
    block_length: int = 4,
    replicates: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare methods on common nominal timestamps for each PV penetration.

    The primary bus-time estimand is conditional on both solvers and AC validation succeeding;
    a second unconditional endpoint counts any failure or voltage-violation event as one.
    This avoids silently treating failed decisions as missing favorable observations.
    """
    nominal = raw[raw["condition"] == "s0_nominal"].copy()
    rows: list[dict[str, Any]] = []
    for penetration, group in nominal.groupby("pv_penetration"):
        left = group[group["method"] == candidate].set_index("timestamp").sort_index()
        right = group[group["method"] == reference].set_index("timestamp").sort_index()
        common = left.index.intersection(right.index)
        if common.empty:
            continue
        left, right = left.loc[common], right.loc[common]
        failure_left = (~left["successful"].astype(bool)).to_numpy(dtype=float)
        failure_right = (~right["successful"].astype(bool)).to_numpy(dtype=float)
        endpoint_left = np.maximum(failure_left, left["violation_event"].fillna(1).to_numpy())
        endpoint_right = np.maximum(
            failure_right, right["violation_event"].fillna(1).to_numpy()
        )
        interval = paired_block_bootstrap_ci(
            endpoint_left,
            endpoint_right,
            min(block_length, len(common)),
            replicates,
            seed=seed,
        )
        rows.append(
            {
                "pv_penetration": penetration,
                "candidate": candidate,
                "reference": reference,
                "endpoint": "failure_or_voltage_violation_event",
                "paired_observations": len(common),
                **interval,
            }
        )
        valid = left["successful"].astype(bool) & right["successful"].astype(bool)
        valid &= left["bus_time_violation_rate"].notna()
        valid &= right["bus_time_violation_rate"].notna()
        if valid.any():
            conditional = paired_block_bootstrap_ci(
                left.loc[valid, "bus_time_violation_rate"].to_numpy(),
                right.loc[valid, "bus_time_violation_rate"].to_numpy(),
                min(block_length, int(valid.sum())),
                replicates,
                seed=seed,
            )
            rows.append(
                {
                    "pv_penetration": penetration,
                    "candidate": candidate,
                    "reference": reference,
                    "endpoint": "bus_time_violation_rate_conditional_on_success",
                    "paired_observations": int(valid.sum()),
                    **conditional,
                }
            )
    return pd.DataFrame(rows)

