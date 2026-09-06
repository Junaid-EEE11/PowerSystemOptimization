"""End-to-end forecast-to-decision-to-AC-validation research pipeline."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wasserstein_distance

from cs_wdro_grid.data.features import supervised_frame
from cs_wdro_grid.data.io import load_measured_profiles
from cs_wdro_grid.data.splits import ChronologicalSplits, chronological_slices
from cs_wdro_grid.data.synthetic import SYNTHETIC_WATERMARK, generate_synthetic_profiles
from cs_wdro_grid.evaluation.comparisons import paired_method_comparisons
from cs_wdro_grid.evaluation.forecast import forecast_metrics
from cs_wdro_grid.experiments.ablations import fixed_physical_scale, required_ablation_grid
from cs_wdro_grid.experiments.reporting import generate_figures, generate_tables
from cs_wdro_grid.experiments.stress import stress_conditions
from cs_wdro_grid.experiments.tracking import build_manifest, write_json
from cs_wdro_grid.forecasting.baselines import LinearForecaster
from cs_wdro_grid.forecasting.models import ForecastBundle
from cs_wdro_grid.networks.ieee33 import load_ieee33
from cs_wdro_grid.networks.lindistflow import solve_lindistflow
from cs_wdro_grid.networks.profiles import PVFleet, create_pv_fleet, operating_state
from cs_wdro_grid.optimization.opf import OPFResult, solve_volt_var
from cs_wdro_grid.uncertainty.conformal import (
    conformalize,
    finite_sample_quantile,
    nonconformity_scores,
)
from cs_wdro_grid.uncertainty.scaling import predictive_scale, standardized_errors
from cs_wdro_grid.utils.seed import set_global_seed
from cs_wdro_grid.validation.ac import solve_ac_power_flow
from cs_wdro_grid.validation.metrics import voltage_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastArtifacts:
    frame: pd.DataFrame
    metrics: pd.DataFrame
    splits: ChronologicalSplits
    residuals: np.ndarray
    raw_residuals: np.ndarray
    fixed_residuals: np.ndarray
    fixed_scale: np.ndarray
    seasonal_residuals: np.ndarray
    decision_index: pd.Index
    test_index: pd.Index


def _load_profiles(config: dict[str, Any]) -> pd.DataFrame:
    data_config = config["data"]
    if data_config["source"] == "synthetic":
        return generate_synthetic_profiles(
            periods=int(data_config["periods"]),
            frequency_minutes=int(data_config["frequency_minutes"]),
            seed=int(config["project"]["seed"]),
        )
    if data_config["source"] == "measured":
        return load_measured_profiles(data_config["load_path"], data_config["pv_path"])
    raise ValueError(f"Unsupported data source: {data_config['source']}")


def validate_full_inputs(config: dict[str, Any]) -> None:
    """Fail before computation when full-study evidence inputs are incomplete."""
    if config["project"]["mode"] != "full":
        return
    for key in ("load_path", "pv_path"):
        path = Path(config["data"][key])
        if not path.is_file():
            raise FileNotFoundError(f"Full mode requires {path}; see data/DATA_SOURCES.md")
    master = Path(config["network"].get("ieee123_master", ""))
    if not master.is_file():
        raise FileNotFoundError(
            f"Full mode requires the official IEEE 123 master file at {master}; "
            "see data/external/ieee123/README.md"
        )
    from cs_wdro_grid.validation.opendss import solve_opendss

    preflight = solve_opendss(master)
    if not preflight.converged:
        raise RuntimeError("IEEE 123 OpenDSS preflight did not converge")
    raise NotImplementedError(
        "IEEE 123 OpenDSS validation is available, but the feeder-specific unbalanced "
        "M0-M5 optimization mapping is not yet implemented. Full mode fails closed rather "
        "than silently reporting an IEEE 33-only study."
    )


def _clip_forecasts(variable: str, values: np.ndarray) -> np.ndarray:
    if variable == "pv":
        return np.clip(values, 0.0, 1.0)
    return np.maximum(values, 0.0)


def _prepare_forecasts(profiles: pd.DataFrame, config: dict[str, Any]) -> ForecastArtifacts:
    forecast_config = config["forecast"]
    feature_sets: dict[str, pd.DataFrame] = {}
    targets: dict[str, pd.Series] = {}
    for variable, column in (("load", "load_factor"), ("pv", "pv_factor")):
        x, y = supervised_frame(
            profiles[column], forecast_config["lags"], forecast_config["rolling_windows"]
        )
        feature_sets[variable], targets[variable] = x, y
    common = feature_sets["load"].index.intersection(feature_sets["pv"].index)
    common = common.sort_values()
    splits = chronological_slices(len(common), config["split"])
    split_map = splits.as_dict()
    frame = pd.DataFrame(index=common)
    metric_rows: list[dict[str, Any]] = []
    miscoverage = 1.0 - (
        float(forecast_config["quantiles"][-1]) - float(forecast_config["quantiles"][0])
    )

    for variable in ("load", "pv"):
        x = feature_sets[variable].loc[common]
        y = targets[variable].loc[common]
        train = split_map["forecast_train"]
        model = ForecastBundle(
            tuple(float(item) for item in forecast_config["quantiles"]),
            int(config["project"]["seed"]),
            dict(forecast_config["point_model"]),
            dict(forecast_config["quantile_model"]),
        ).fit(x.iloc[train], y.iloc[train])
        prediction = model.predict(x)
        point = _clip_forecasts(variable, prediction["point"].to_numpy())
        linear = LinearForecaster(float(forecast_config["linear_model"]["alpha"]))
        linear.fit(x.iloc[train], y.iloc[train])
        linear_prediction = _clip_forecasts(variable, linear.predict(x))
        persistence_prediction = y.shift(1).to_numpy()
        raw_lower = _clip_forecasts(variable, prediction.iloc[:, 1].to_numpy())
        raw_upper = _clip_forecasts(variable, prediction.iloc[:, -1].to_numpy())
        raw_lower, raw_upper = np.minimum(raw_lower, raw_upper), np.maximum(raw_lower, raw_upper)
        calibration = split_map["residual_calibration"]
        scores = nonconformity_scores(
            y.iloc[calibration].to_numpy(), raw_lower[calibration], raw_upper[calibration]
        )
        adjustment = finite_sample_quantile(scores, miscoverage)
        lower, upper = conformalize(raw_lower, raw_upper, adjustment)
        lower, upper = _clip_forecasts(variable, lower), _clip_forecasts(variable, upper)
        lower, upper = np.minimum(lower, upper), np.maximum(lower, upper)
        scale = predictive_scale(lower, upper, float(forecast_config["scale_floor"]))
        raw_scale = predictive_scale(raw_lower, raw_upper, float(forecast_config["scale_floor"]))

        seasonal = y.shift(96).to_numpy()
        calibration_positions = np.arange(len(y))[calibration]
        valid_calibration = calibration_positions[np.isfinite(seasonal[calibration_positions])]
        seasonal_radius = finite_sample_quantile(
            np.abs(y.to_numpy()[valid_calibration] - seasonal[valid_calibration]), miscoverage
        )
        seasonal_scale = np.full(
            len(y), max(seasonal_radius, float(forecast_config["scale_floor"]))
        )

        frame[f"{variable}_actual"] = y.to_numpy()
        frame[f"{variable}_point"] = point
        frame[f"{variable}_linear_point"] = linear_prediction
        frame[f"{variable}_persistence_point"] = persistence_prediction
        frame[f"{variable}_raw_lower"] = raw_lower
        frame[f"{variable}_raw_upper"] = raw_upper
        frame[f"{variable}_lower"] = lower
        frame[f"{variable}_upper"] = upper
        frame[f"{variable}_scale"] = scale
        frame[f"{variable}_raw_scale"] = raw_scale
        frame[f"{variable}_seasonal_point"] = seasonal
        frame[f"{variable}_seasonal_scale"] = seasonal_scale

        test = split_map["test"]
        actual_test = y.iloc[test].to_numpy()
        raw_metrics = forecast_metrics(
            actual_test, point[test], raw_lower[test], raw_upper[test], miscoverage
        )
        metric_rows.append({"variable": variable, "model": "boosting_raw", **raw_metrics})
        calibrated_metrics = forecast_metrics(
            actual_test, point[test], lower[test], upper[test], miscoverage
        )
        metric_rows.append(
            {"variable": variable, "model": "boosting_conformal", **calibrated_metrics}
        )
        seasonal_test = seasonal[test]
        baseline_predictions = {
            "persistence": persistence_prediction[test],
            "seasonal_naive": seasonal_test,
            "ridge_linear": linear_prediction[test],
        }
        for baseline_name, baseline_prediction in baseline_predictions.items():
            valid = np.isfinite(baseline_prediction)
            errors = actual_test[valid] - baseline_prediction[valid]
            metric_rows.append(
                {
                    "variable": variable,
                    "model": baseline_name,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "coverage": np.nan,
                    "calibration_error": np.nan,
                    "mean_interval_width": np.nan,
                    "interval_score": np.nan,
                    "pinball_lower": np.nan,
                    "pinball_upper": np.nan,
                }
            )

    calibration = split_map["residual_calibration"]
    physical_errors = np.column_stack(
        [
            frame[f"{variable}_actual"].to_numpy()[calibration]
            - frame[f"{variable}_point"].to_numpy()[calibration]
            for variable in ("load", "pv")
        ]
    )
    residuals = np.column_stack(
        [
            standardized_errors(
                frame[f"{variable}_actual"].to_numpy()[calibration],
                frame[f"{variable}_point"].to_numpy()[calibration],
                frame[f"{variable}_scale"].to_numpy()[calibration],
            )
            for variable in ("load", "pv")
        ]
    )
    raw_residuals = np.column_stack(
        [
            standardized_errors(
                frame[f"{variable}_actual"].to_numpy()[calibration],
                frame[f"{variable}_point"].to_numpy()[calibration],
                frame[f"{variable}_raw_scale"].to_numpy()[calibration],
            )
            for variable in ("load", "pv")
        ]
    )
    floor = float(forecast_config["scale_floor"])
    fixed_scale = fixed_physical_scale(physical_errors, floor)
    fixed_residuals = physical_errors / fixed_scale
    seasonal_errors = np.column_stack(
        [
            frame[f"{variable}_actual"].to_numpy()[calibration]
            - frame[f"{variable}_seasonal_point"].to_numpy()[calibration]
            for variable in ("load", "pv")
        ]
    )
    seasonal_scales = np.column_stack(
        [frame[f"{variable}_seasonal_scale"].to_numpy()[calibration] for variable in ("load", "pv")]
    )
    seasonal_residuals = seasonal_errors / seasonal_scales
    if not np.isfinite(seasonal_residuals).all():
        raise RuntimeError("Seasonal residual bank contains non-finite values")

    return ForecastArtifacts(
        frame=frame,
        metrics=pd.DataFrame(metric_rows),
        splits=splits,
        residuals=residuals,
        raw_residuals=raw_residuals,
        fixed_residuals=fixed_residuals,
        fixed_scale=fixed_scale,
        seasonal_residuals=seasonal_residuals,
        decision_index=common[split_map["decision_calibration"]],
        test_index=common[split_map["test"]],
    )


def _sample_index(index: pd.Index, maximum: int | None) -> pd.Index:
    if maximum is None or maximum >= len(index):
        return index
    if maximum <= 0:
        raise ValueError("Experiment point limit must be positive or null")
    positions = np.linspace(0, len(index) - 1, maximum, dtype=int)
    return index[np.unique(positions)]


def _decision(
    config: dict[str, Any],
    network: Any,
    fleet: PVFleet,
    row: pd.Series,
    method: str,
    residuals: np.ndarray,
    epsilon: float,
    scale_override: np.ndarray | None = None,
    point_override: np.ndarray | None = None,
    alpha_override: float | None = None,
) -> OPFResult:
    optimization = config["optimization"]
    scale = (
        scale_override
        if scale_override is not None
        else np.array([row["load_scale"], row["pv_scale"]])
    )
    point = (
        point_override
        if point_override is not None
        else np.array([row["load_point"], row["pv_point"]])
    )
    return solve_volt_var(
        network=network,
        fleet=fleet,
        point_load=float(point[0]),
        point_pv=float(point[1]),
        method=method,
        weights=optimization["weights"],
        v_min=float(config["network"]["v_min"]),
        v_max=float(config["network"]["v_max"]),
        scale=scale,
        residuals=residuals,
        alpha=float(alpha_override or optimization["cvar_alpha"]),
        epsilon=float(epsilon),
        solver=optimization["solver"],
        scenario_count=int(optimization["scenario_count"]),
        allow_curtailment=bool(optimization["allow_curtailment"]),
    )


def _evaluate_result(
    config: dict[str, Any],
    network: Any,
    fleet: PVFleet,
    decision: OPFResult,
    actual_load: float,
    actual_pv: float,
) -> dict[str, Any]:
    common = {
        "method": decision.method,
        "solver_status": decision.status,
        "solve_time_seconds": decision.solve_time_seconds,
        "objective": decision.objective,
        "successful": False,
        "ac_converged": False,
    }
    if not decision.successful:
        return {
            **common,
            "violation_event": 1.0,
            "bus_time_violation_rate": np.nan,
            "mean_violation_magnitude": np.nan,
            "worst_violation_magnitude": np.nan,
            "minimum_voltage": np.nan,
            "maximum_voltage": np.nan,
            "ac_loss_mw": np.nan,
            "linear_loss_mw": np.nan,
            "curtailment_mw": np.nan,
            "reactive_effort_mvar": np.nan,
            "inverter_capability_margin_max": np.nan,
            "linear_actual_min_voltage": np.nan,
            "linear_actual_max_voltage": np.nan,
            "predicted_min_voltage": np.nan,
            "predicted_max_voltage": np.nan,
            "linear_ac_voltage_mae": np.nan,
            "linear_ac_voltage_max_error": np.nan,
            "voltage_profile_json": "[]",
            "q_injection_json": "[]",
            "curtailment_json": "[]",
        }
    available = fleet.capacity_pu * float(np.clip(actual_pv, 0.0, 1.0))
    applied_curtailment = np.minimum(decision.curtailment_pu, available)
    net_p, net_q = operating_state(
        network, fleet, actual_load, actual_pv, decision.q_injection_pu, applied_curtailment
    )
    linear = solve_lindistflow(network, net_p, net_q)
    ac = solve_ac_power_flow(network, net_p, net_q)
    linear_voltage = np.sqrt(np.maximum(linear.voltage_squared_pu, 0.0))
    ac_voltage = ac.voltage_magnitude_pu
    reliability = voltage_metrics(
        ac_voltage[1:], float(config["network"]["v_min"]), float(config["network"]["v_max"])
    )
    p_used = available - applied_curtailment
    capability_margin = (
        np.sqrt(np.square(decision.q_injection_pu) + np.square(p_used)) - fleet.inverter_rating_pu
    )
    predicted_voltage = np.sqrt(np.maximum(decision.predicted_voltage_squared_pu, 0.0))
    return {
        **common,
        "successful": bool(ac.converged),
        "ac_converged": bool(ac.converged),
        "violation_event": reliability["violation_event_rate"],
        "bus_time_violation_rate": reliability["bus_time_violation_rate"],
        "mean_violation_magnitude": reliability["mean_violation_magnitude"],
        "worst_violation_magnitude": reliability["worst_violation_magnitude"],
        "minimum_voltage": reliability["minimum_voltage"],
        "maximum_voltage": reliability["maximum_voltage"],
        "ac_loss_mw": ac.loss_pu * network.base_mva,
        "linear_loss_mw": linear.loss_pu * network.base_mva,
        "curtailment_mw": float(np.sum(applied_curtailment) * network.base_mva),
        "reactive_effort_mvar": float(np.sum(np.abs(decision.q_injection_pu)) * network.base_mva),
        "inverter_capability_margin_max": float(np.max(capability_margin)),
        "linear_actual_min_voltage": float(np.min(linear_voltage[1:])),
        "linear_actual_max_voltage": float(np.max(linear_voltage[1:])),
        "predicted_min_voltage": float(np.min(predicted_voltage[1:])),
        "predicted_max_voltage": float(np.max(predicted_voltage[1:])),
        "linear_ac_voltage_mae": float(np.mean(np.abs(linear_voltage[1:] - ac_voltage[1:]))),
        "linear_ac_voltage_max_error": float(np.max(np.abs(linear_voltage[1:] - ac_voltage[1:]))),
        "voltage_profile_json": json.dumps(ac_voltage.tolist()),
        "q_injection_json": json.dumps(decision.q_injection_pu.tolist()),
        "curtailment_json": json.dumps(applied_curtailment.tolist()),
    }


def _select_epsilon(
    config: dict[str, Any], artifacts: ForecastArtifacts, network: Any, fleet: PVFleet
) -> tuple[float, pd.DataFrame]:
    points = _sample_index(
        artifacts.decision_index, config["experiment"]["max_decision_calibration_points"]
    )
    records: list[dict[str, Any]] = []
    for epsilon in config["optimization"]["epsilon_grid"]:
        for timestamp in points:
            row = artifacts.frame.loc[timestamp]
            decision = _decision(
                config, network, fleet, row, "m5", artifacts.residuals, float(epsilon)
            )
            evaluated = _evaluate_result(
                config, network, fleet, decision, row["load_actual"], row["pv_actual"]
            )
            records.append({"timestamp": timestamp, "epsilon": float(epsilon), **evaluated})
    results = pd.DataFrame(records)
    summary = results.groupby("epsilon", as_index=False).agg(
        success_rate=("successful", "mean"),
        violation_rate=("violation_event", "mean"),
        cost=("objective", "mean"),
    )
    # Epsilon zero remains the empirical-CVaR calibration reference. M5 is
    # definitionally Wasserstein robust and is selected only from positive radii.
    positive = summary[summary["epsilon"] > 0]
    if positive.empty:
        raise RuntimeError("CS-WDRO radius selection requires at least one positive epsilon")
    fully_solved = positive[positive["success_rate"] == 1.0]
    target = float(config["optimization"]["reliability_target"])
    reliable = fully_solved[fully_solved["violation_rate"] <= target]
    if not reliable.empty:
        selected = float(reliable.sort_values(["cost", "epsilon"]).iloc[0]["epsilon"])
        target_achieved = True
    else:
        # A failed solve is already counted as a violation. Prefer the candidate with
        # greatest solvability, then lowest violation rate and cost; disclose failure.
        selected = float(
            positive.sort_values(
                ["success_rate", "violation_rate", "cost", "epsilon"],
                ascending=[False, True, True, True],
                na_position="last",
            ).iloc[0]["epsilon"]
        )
        target_achieved = False
    results["selected"] = results["epsilon"] == selected
    results["reliability_target_achieved"] = target_achieved
    return selected, results


def _run_test(
    config: dict[str, Any], artifacts: ForecastArtifacts, network: Any, selected_epsilon: float
) -> pd.DataFrame:
    points = _sample_index(artifacts.test_index, config["experiment"]["max_test_points"])
    test_frame = artifacts.frame.loc[artifacts.test_index]
    high_load = float(test_frame["load_actual"].quantile(0.95))
    high_pv = float(test_frame["pv_actual"].quantile(0.95))
    classification = (
        SYNTHETIC_WATERMARK if config["data"]["source"] == "synthetic" else "MEASURED RESEARCH DATA"
    )
    records: list[dict[str, Any]] = []
    for penetration in config["experiment"]["pv_penetrations"]:
        fleet = create_pv_fleet(
            network,
            config["network"]["pv_buses"],
            float(penetration),
            float(config["network"]["inverter_oversize"]),
        )
        for timestamp in points:
            row = artifacts.frame.loc[timestamp]
            conditions = stress_conditions(
                row["load_actual"],
                row["pv_actual"],
                row["load_point"],
                row["pv_point"],
                config["experiment"]["load_biases"],
                config["experiment"]["stress_error_scales"],
            )
            if row["load_actual"] >= high_load:
                conditions["s4_high_load"] = (row["load_actual"], row["pv_actual"])
            if row["pv_actual"] >= high_pv:
                conditions["s5_high_pv"] = (row["load_actual"], row["pv_actual"])
            for method in config["optimization"]["methods"]:
                epsilon = selected_epsilon if method == "m5" else 0.0
                decision = _decision(
                    config, network, fleet, row, method, artifacts.residuals, epsilon
                )
                for condition, (actual_load, actual_pv) in conditions.items():
                    evaluated = _evaluate_result(
                        config, network, fleet, decision, actual_load, actual_pv
                    )
                    records.append(
                        {
                            "timestamp": timestamp,
                            "condition": condition,
                            "pv_penetration": float(penetration),
                            "actual_load": actual_load,
                            "actual_pv": actual_pv,
                            "forecast_load": row["load_point"],
                            "forecast_pv": row["pv_point"],
                            "selected_epsilon": selected_epsilon,
                            "data_classification": classification,
                            **evaluated,
                        }
                    )
    return pd.DataFrame(records)


def _run_ablations(
    config: dict[str, Any], artifacts: ForecastArtifacts, network: Any, selected_epsilon: float
) -> pd.DataFrame:
    points = _sample_index(
        artifacts.test_index,
        min(2, len(artifacts.test_index)) if config["project"]["mode"] == "quick" else None,
    )
    fleet = create_pv_fleet(
        network,
        config["network"]["pv_buses"],
        float(config["network"]["pv_penetration"]),
        float(config["network"]["inverter_oversize"]),
    )
    records: list[dict[str, Any]] = []
    for spec in required_ablation_grid(selected_epsilon):
        for timestamp in points:
            row = artifacts.frame.loc[timestamp]
            scale = np.array([row["load_scale"], row["pv_scale"]])
            point = np.array([row["load_point"], row["pv_point"]])
            bank = artifacts.residuals
            if spec.name == "a1_no_conformal_scaling":
                scale = np.array([row["load_raw_scale"], row["pv_raw_scale"]])
                bank = artifacts.raw_residuals
            elif spec.name == "a2_wdro_without_scaling":
                scale = artifacts.fixed_scale
                bank = artifacts.fixed_residuals
            elif spec.forecast_model == "seasonal_naive":
                point = np.array([row["load_seasonal_point"], row["pv_seasonal_point"]])
                scale = np.array([row["load_seasonal_scale"], row["pv_seasonal_scale"]])
                bank = artifacts.seasonal_residuals
            if spec.bank_size is not None:
                bank = bank[-min(spec.bank_size, len(bank)) :]
            decision = _decision(
                config,
                network,
                fleet,
                row,
                "m5",
                bank,
                spec.epsilon,
                scale_override=scale,
                point_override=point,
                alpha_override=spec.alpha,
            )
            evaluated = _evaluate_result(
                config, network, fleet, decision, row["load_actual"], row["pv_actual"]
            )
            records.append(
                {
                    "timestamp": timestamp,
                    "ablation": spec.name,
                    "alpha": spec.alpha,
                    "epsilon": spec.epsilon,
                    "bank_size": len(bank),
                    "forecast_model": spec.forecast_model,
                    **evaluated,
                }
            )
    return pd.DataFrame(records)


def _natural_shift(artifacts: ForecastArtifacts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibration = artifacts.frame.loc[artifacts.decision_index]
    test = artifacts.frame.loc[artifacts.test_index]
    for variable in ("load", "pv"):
        first = calibration[f"{variable}_actual"] - calibration[f"{variable}_point"]
        second = test[f"{variable}_actual"] - test[f"{variable}_point"]
        rows.append(
            {
                "variable": variable,
                "decision_calibration_mean_error": first.mean(),
                "test_mean_error": second.mean(),
                "decision_calibration_std": first.std(ddof=1),
                "test_std": second.std(ddof=1),
                "wasserstein_1_univariate": wasserstein_distance(first, second),
            }
        )
    return pd.DataFrame(rows)


def run_pipeline(config: dict[str, Any], output_root: str | Path | None = None) -> Path:
    """Run a complete configured experiment and return its run directory."""
    validate_full_inputs(config)
    set_global_seed(int(config["project"]["seed"]))
    profiles = _load_profiles(config)
    artifacts = _prepare_forecasts(profiles, config)
    network = load_ieee33(float(config["network"]["base_mva"]), float(config["network"]["base_kv"]))
    calibration_fleet = create_pv_fleet(
        network,
        config["network"]["pv_buses"],
        float(config["network"]["pv_penetration"]),
        float(config["network"]["inverter_oversize"]),
    )
    selected_epsilon, epsilon_results = _select_epsilon(
        config, artifacts, network, calibration_fleet
    )
    raw = _run_test(config, artifacts, network, selected_epsilon)
    ablations = _run_ablations(config, artifacts, network, selected_epsilon)
    comparisons = paired_method_comparisons(
        raw,
        reference="m1",
        candidate="m5",
        block_length=int(config["experiment"]["bootstrap_block_length"]),
        replicates=int(config["experiment"]["bootstrap_replicates"]),
        seed=int(config["project"]["seed"]),
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['project']['mode']}-{stamp}"
    root = Path(output_root or config["project"]["output_root"])
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    (run_dir / "raw").mkdir(parents=True)
    raw.to_csv(run_dir / "raw" / "operational_results.csv", index=False)
    artifacts.frame.to_csv(run_dir / "raw" / "forecast_series.csv")
    artifacts.metrics.to_csv(run_dir / "raw" / "forecast_metrics.csv", index=False)
    epsilon_results.to_csv(run_dir / "raw" / "epsilon_calibration.csv", index=False)
    ablations.to_csv(run_dir / "raw" / "ablation_results.csv", index=False)
    comparisons.to_csv(run_dir / "raw" / "paired_comparisons.csv", index=False)
    _natural_shift(artifacts).to_csv(run_dir / "raw" / "natural_shift.csv", index=False)
    with (run_dir / "configuration.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {key: value for key, value in config.items() if key != "_config_path"},
            handle,
            sort_keys=False,
        )

    characteristics = {
        "run_classification": SYNTHETIC_WATERMARK
        if config["data"]["source"] == "synthetic"
        else "MEASURED",
        "dataset_start": str(profiles.index.min()),
        "dataset_end": str(profiles.index.max()),
        "observations": len(profiles),
        "resolution_minutes": config["data"].get("frequency_minutes", "native"),
        "network": network.name,
        "buses": network.n_bus,
        "branches": network.n_branch,
        "base_mva": network.base_mva,
        "base_kv": network.base_kv,
    }
    generate_tables(
        raw, artifacts.metrics, epsilon_results, ablations, characteristics, run_dir / "tables"
    )
    comparisons.to_csv(run_dir / "tables" / "paired_comparisons.csv", index=False)
    generate_figures(
        raw,
        artifacts.frame.loc[artifacts.test_index],
        artifacts.metrics,
        epsilon_results,
        ablations,
        run_dir / "figures",
    )
    statuses = Counter(raw["solver_status"].astype(str))
    manifest = build_manifest(
        config,
        run_id,
        (str(profiles.index.min()), str(profiles.index.max())),
        selected_epsilon,
        dict(statuses),
    )
    manifest["data_classification"] = characteristics["run_classification"]
    manifest["reliability_target_achieved"] = bool(
        epsilon_results["reliability_target_achieved"].iloc[0]
    )
    manifest["executed"] = True
    write_json(run_dir / "experiment_manifest.json", manifest)
    LOGGER.info("Completed %s at %s", run_id, run_dir)
    return run_dir
