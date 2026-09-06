"""Experiment provenance and immutable manifest generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    """Return the commit when Git exists; absence is recorded rather than hidden."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "cvxpy", "matplotlib", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_manifest(
    config: dict[str, Any],
    run_id: str,
    dataset_range: tuple[str, str],
    selected_epsilon: float,
    solver_status_counts: dict[str, int],
) -> dict[str, Any]:
    research_spec = Path("RESEARCH_SPEC.md")
    return {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "research_spec_sha256": sha256_file(research_spec) if research_spec.is_file() else None,
        "configuration_path": config.get("_config_path"),
        "seed": config["project"]["seed"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "dataset_source": config["data"]["source"],
        "dataset_identifier": config["data"].get("dataset_identifier", config["data"]["source"]),
        "dataset_time_range": list(dataset_range),
        "selected_epsilon": selected_epsilon,
        "forecast_parameters": config["forecast"],
        "objective_weights": config["optimization"]["weights"],
        "cvar_alpha": config["optimization"]["cvar_alpha"],
        "epsilon_grid": config["optimization"]["epsilon_grid"],
        "network": config["network"].get("name", config["network"].get("names")),
        "pv_penetrations": config["experiment"]["pv_penetrations"],
        "methods": config["optimization"]["methods"],
        "solver": config["optimization"]["solver"],
        "solver_status_counts": solver_status_counts,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
