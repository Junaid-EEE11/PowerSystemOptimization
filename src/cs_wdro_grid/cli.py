"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from cs_wdro_grid.config import load_config
from cs_wdro_grid.experiments.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CS-WDRO research experiments")
    parser.add_argument(
        "--config", default="configs/quick.yaml", help="YAML experiment configuration"
    )
    parser.add_argument("--output-root", default=None, help="Optional run-output root override")
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)
    run_dir = run_pipeline(config, args.output_root)
    print(run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
