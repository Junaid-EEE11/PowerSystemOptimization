"""Generate tables and figures exclusively from raw run outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _placeholder(title: str, message: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save(fig, path)


def generate_tables(
    raw: pd.DataFrame,
    forecast: pd.DataFrame,
    epsilon: pd.DataFrame,
    ablations: pd.DataFrame,
    network_characteristics: dict[str, float | str],
    output_dir: Path,
) -> None:
    """Write the nine predefined table inputs as CSV, never hand-entering values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([network_characteristics]).to_csv(
        output_dir / "table_1_dataset_feeder.csv", index=False
    )
    forecast.to_csv(output_dir / "table_2_forecasting.csv", index=False)
    interval_columns = [
        column
        for column in (
            "variable",
            "model",
            "coverage",
            "calibration_error",
            "mean_interval_width",
            "interval_score",
        )
        if column in forecast.columns
    ]
    forecast[interval_columns].to_csv(output_dir / "table_3_interval_calibration.csv", index=False)

    nominal = raw[raw["condition"] == "s0_nominal"]
    aggregation = {
        "bus_time_violation_rate": "mean",
        "violation_event": "mean",
        "mean_violation_magnitude": "mean",
        "ac_loss_mw": "mean",
        "curtailment_mw": "mean",
        "reactive_effort_mvar": "mean",
        "objective": "mean",
        "solve_time_seconds": "mean",
        "successful": "mean",
    }
    nominal.groupby("method", as_index=False).agg(aggregation).to_csv(
        output_dir / "table_4_nominal_operation.csv", index=False
    )
    raw.groupby(["condition", "method"], as_index=False).agg(aggregation).to_csv(
        output_dir / "table_5_stress_reliability.csv", index=False
    )
    nominal.groupby(["pv_penetration", "method"], as_index=False).agg(aggregation).to_csv(
        output_dir / "table_6_pv_penetration.csv", index=False
    )
    ablations.to_csv(output_dir / "table_7_ablations.csv", index=False)
    raw.groupby("method", as_index=False).agg(
        mean_solve_seconds=("solve_time_seconds", "mean"),
        p95_solve_seconds=("solve_time_seconds", lambda x: x.quantile(0.95)),
        success_rate=("successful", "mean"),
    ).to_csv(output_dir / "table_8_computation.csv", index=False)
    nominal.groupby(["pv_penetration", "method"], as_index=False).agg(
        mean_voltage_error=("linear_ac_voltage_mae", "mean"),
        maximum_voltage_error=("linear_ac_voltage_max_error", "max"),
        predicted_min_voltage=("linear_actual_min_voltage", "mean"),
        ac_min_voltage=("minimum_voltage", "mean"),
    ).to_csv(output_dir / "table_9_linear_ac_errors.csv", index=False)
    epsilon.to_csv(output_dir / "epsilon_calibration.csv", index=False)


def generate_figures(
    raw: pd.DataFrame,
    forecast_series: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    epsilon: pd.DataFrame,
    ablations: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate all required figure files from machine-readable data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "m0": "#777777",
        "m1": "#1f77b4",
        "m2": "#ff7f0e",
        "m3": "#d62728",
        "m4": "#9467bd",
        "m5": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    stages = [
        "Time series",
        "Forecasts",
        "Conformal\ncalibration",
        "Scaled\nresiduals",
        "M0-M5\ndecisions",
        "Nonlinear AC\nvalidation",
    ]
    xs = np.linspace(0.08, 0.92, len(stages))
    for x, stage in zip(xs, stages, strict=True):
        ax.text(
            x,
            0.5,
            stage,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "#e8f1fa"},
        )
    for left, right in zip(xs[:-1], xs[1:], strict=True):
        ax.annotate("", (right - 0.06, 0.5), (left + 0.06, 0.5), arrowprops={"arrowstyle": "->"})
    _save(fig, output_dir / "figure_1_pipeline.png")

    if forecast_series.empty:
        _placeholder(
            "Forecast intervals",
            "No forecast rows were produced.",
            output_dir / "figure_2_forecasts.png",
        )
    else:
        view = forecast_series.tail(min(192, len(forecast_series)))
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for axis, variable in zip(axes, ("load", "pv"), strict=True):
            axis.plot(
                view.index, view[f"{variable}_actual"], label="actual", color="black", linewidth=1
            )
            axis.plot(
                view.index, view[f"{variable}_point"], label="point", color="#1f77b4", linewidth=1
            )
            axis.fill_between(
                view.index,
                view[f"{variable}_lower"],
                view[f"{variable}_upper"],
                alpha=0.25,
                label="conformal interval",
            )
            axis.set_ylabel(f"{variable} factor")
            axis.legend(loc="upper right", ncol=3)
        _save(fig, output_dir / "figure_2_forecasts.png")

    probabilistic = (
        forecast_metrics.dropna(subset=["coverage"])
        if "coverage" in forecast_metrics
        else pd.DataFrame()
    )
    if probabilistic.empty:
        _placeholder(
            "Forecast calibration",
            "Probabilistic metrics unavailable.",
            output_dir / "figure_3_calibration.png",
        )
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = probabilistic["variable"] + ":" + probabilistic["model"]
        ax.bar(labels, probabilistic["coverage"], color="#1f77b4")
        ax.axhline(0.90, color="black", linestyle="--", label="nominal 90%")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Empirical coverage")
        ax.legend()
        _save(fig, output_dir / "figure_3_calibration.png")

    profile_rows = raw[(raw["condition"] == "s0_nominal") & raw["successful"]].head(6)
    if profile_rows.empty:
        _placeholder(
            "Voltage profiles",
            "No successful nominal power flows.",
            output_dir / "figure_4_voltage_profiles.png",
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        for _, row in profile_rows.iterrows():
            values = json.loads(row["voltage_profile_json"])
            ax.plot(np.arange(1, len(values) + 1), values, label=row["method"], alpha=0.8)
        ax.axhline(0.95, color="red", linestyle="--")
        ax.axhline(1.05, color="red", linestyle="--")
        ax.set(xlabel="Bus", ylabel="AC voltage (pu)")
        ax.legend(ncol=3)
        _save(fig, output_dir / "figure_4_voltage_profiles.png")

    nominal = raw[raw["condition"] == "s0_nominal"]
    reliability = nominal.groupby("method", as_index=False)["bus_time_violation_rate"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        reliability["method"],
        reliability["bus_time_violation_rate"],
        color=[colors.get(x, "gray") for x in reliability["method"]],
    )
    ax.set_ylabel("Bus-time AC violation rate")
    _save(fig, output_dir / "figure_5_reliability.png")

    tradeoff = nominal.groupby("method", as_index=False).agg(
        cost=("objective", "mean"), violation=("bus_time_violation_rate", "mean")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in tradeoff.iterrows():
        ax.scatter(
            row["cost"],
            row["violation"],
            color=colors.get(row["method"], "gray"),
            label=row["method"],
        )
    ax.set(xlabel="Approximate operating objective", ylabel="Bus-time AC violation rate")
    ax.legend(ncol=2)
    _save(fig, output_dir / "figure_6_reliability_cost.png")

    shifted = raw.groupby(["condition", "method"], as_index=False)["bus_time_violation_rate"].mean()
    conditions = list(dict.fromkeys(shifted["condition"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, group in shifted.groupby("method"):
        mapping = dict(zip(group["condition"], group["bus_time_violation_rate"], strict=True))
        ax.plot(
            range(len(conditions)),
            [mapping.get(item, np.nan) for item in conditions],
            marker="o",
            label=method,
            color=colors.get(method),
        )
    ax.set_xticks(range(len(conditions)), conditions, rotation=25, ha="right")
    ax.set_ylabel("Bus-time AC violation rate")
    ax.legend(ncol=3)
    _save(fig, output_dir / "figure_7_distribution_shift.png")

    penetration = nominal.groupby(["pv_penetration", "method"], as_index=False)[
        "bus_time_violation_rate"
    ].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in penetration.groupby("method"):
        ax.plot(
            group["pv_penetration"],
            group["bus_time_violation_rate"],
            marker="o",
            label=method,
            color=colors.get(method),
        )
    ax.set(xlabel="PV penetration", ylabel="Bus-time AC violation rate")
    ax.legend(ncol=3)
    _save(fig, output_dir / "figure_8_pv_penetration.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if not epsilon.empty:
        summary = epsilon.groupby("epsilon", as_index=False).agg(
            violation=("violation_event", "mean"), cost=("objective", "mean")
        )
        ax.plot(summary["epsilon"], summary["violation"], marker="o", label="violation event rate")
        ax.set_xlabel("Wasserstein radius epsilon")
        ax.set_ylabel("Calibration violation-event rate")
    else:
        ax.text(0.5, 0.5, "No epsilon calibration rows", ha="center")
    _save(fig, output_dir / "figure_9_epsilon.png")

    if ablations.empty:
        _placeholder(
            "Ablations",
            "Ablations were not evaluated in this run.",
            output_dir / "figure_10_ablations.png",
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        summary = ablations.groupby("ablation", as_index=False)["bus_time_violation_rate"].mean()
        ax.bar(summary["ablation"], summary["bus_time_violation_rate"], color="#9467bd")
        ax.tick_params(axis="x", rotation=60)
        ax.set_ylabel("Bus-time AC violation rate")
        _save(fig, output_dir / "figure_10_ablations.png")

    comparison = nominal.dropna(subset=["linear_actual_min_voltage", "minimum_voltage"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(comparison["linear_actual_min_voltage"], comparison["minimum_voltage"], alpha=0.5)
    bounds = [
        min(comparison[["linear_actual_min_voltage", "minimum_voltage"]].min(), default=0.9),
        max(comparison[["linear_actual_min_voltage", "minimum_voltage"]].max(), default=1.1),
    ]
    ax.plot(bounds, bounds, "k--")
    ax.set(xlabel="LinDistFlow minimum voltage", ylabel="AC minimum voltage")
    _save(fig, output_dir / "figure_11_linear_vs_ac.png")

    runtime = raw.groupby("method", as_index=False)["solve_time_seconds"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        runtime["method"],
        runtime["solve_time_seconds"],
        color=[colors.get(x, "gray") for x in runtime["method"]],
    )
    ax.set_ylabel("Mean solve time (s)")
    _save(fig, output_dir / "figure_12_runtime.png")
