"""Download an explicitly supplied public file with optional SHA256 verification."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download one documented public input; no undocumented default source is assumed"
        )
    )
    parser.add_argument("url")
    parser.add_argument("destination")
    parser.add_argument("--sha256", default=None)
    args = parser.parse_args()
    destination = Path(args.destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url) as response, destination.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if args.sha256 and digest.lower() != args.sha256.lower():
        destination.unlink()
        raise RuntimeError("Downloaded file SHA256 mismatch; the untrusted file was removed")
    print(f"{destination} sha256={digest}")
    print("Record source, access date, license, and preprocessing in data/DATA_SOURCES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
