"""Validate and align measured load/PV inputs without interpolation."""

from __future__ import annotations

import argparse
from pathlib import Path

from cs_wdro_grid.data.io import load_measured_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="data/raw/load.csv")
    parser.add_argument("--pv", default="data/raw/pv.csv")
    parser.add_argument("--output", default="data/processed/profiles.csv")
    args = parser.parse_args()
    frame = load_measured_profiles(args.load, args.pv)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index_label="timestamp")
    print(f"Wrote {len(frame)} aligned measured rows to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
