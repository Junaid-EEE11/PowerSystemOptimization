"""CS-WDRO research software."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cs-wdro-grid")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.1.0"
