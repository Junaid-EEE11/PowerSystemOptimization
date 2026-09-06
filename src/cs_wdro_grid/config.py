"""Configuration loading and validation with explicit inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or unsafe."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML, recursively merge its optional ``extends`` parent, and validate it."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (config_path.parent / str(parent)).resolve()
        config = _merge(load_config(parent_path), raw)
    else:
        config = raw
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject configuration errors that could compromise an experiment."""
    required = {"project", "data", "split", "forecast", "network", "optimization", "experiment"}
    missing = required.difference(config)
    if missing:
        raise ConfigurationError(f"Missing configuration sections: {sorted(missing)}")
    fractions = config["split"]
    names = ("forecast_train", "residual_calibration", "decision_calibration", "test")
    if any(float(fractions.get(name, 0.0)) <= 0 for name in names):
        raise ConfigurationError("All four chronological split fractions must be positive")
    if abs(sum(float(fractions[name]) for name in names) - 1.0) > 1e-9:
        raise ConfigurationError("Chronological split fractions must sum to one")
    quantiles = list(config["forecast"]["quantiles"])
    if quantiles != sorted(quantiles) or not all(0 < float(q) < 1 for q in quantiles):
        raise ConfigurationError("Forecast quantiles must be sorted and strictly inside (0, 1)")
    alpha = float(config["optimization"]["cvar_alpha"])
    if not 0 < alpha < 1:
        raise ConfigurationError("CVaR alpha must lie strictly inside (0, 1)")
    epsilon_grid = [float(item) for item in config["optimization"]["epsilon_grid"]]
    if any(item < 0 for item in epsilon_grid) or not any(item > 0 for item in epsilon_grid):
        raise ConfigurationError(
            "The epsilon grid must contain a positive M5 radius and no negatives"
        )
    if config["project"]["mode"] == "full" and config["data"]["source"] == "synthetic":
        raise ConfigurationError("Full mode may not use synthetic development data")
