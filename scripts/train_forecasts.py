"""Fit forecasts and save predictions/metrics without running grid decisions."""

from __future__ import annotations

import argparse
from pathlib import Path

from cs_wdro_grid.config import load_config
from cs_wdro_grid.experiments.pipeline import _load_profiles, _prepare_forecasts
from cs_wdro_grid.utils.seed import set_global_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--output", default="results/forecast-development")
    args = parser.parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["project"]["seed"]))
    artifacts = _prepare_forecasts(_load_profiles(config), config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts.frame.to_csv(output / "forecast_series.csv")
    artifacts.metrics.to_csv(output / "forecast_metrics.csv", index=False)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
