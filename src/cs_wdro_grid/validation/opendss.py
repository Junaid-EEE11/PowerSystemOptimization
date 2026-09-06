"""Optional OpenDSS adapter for external IEEE 123 nonlinear validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OpenDSSResult:
    converged: bool
    node_names: tuple[str, ...]
    voltage_magnitude_pu: np.ndarray
    active_loss_mw: float
    reactive_loss_mvar: float


def solve_opendss(
    master_dss: str | Path,
    load_multiplier: float = 1.0,
    pv_irradiance: float = 1.0,
    pv_kvar: dict[str, float] | None = None,
) -> OpenDSSResult:
    """Compile and solve an OpenDSS feeder with explicit PV inverter setpoints.

    ``pv_kvar`` maps existing PVSystem names to three-phase kvar injection. The adapter never
    creates or relocates DERs because doing so would silently alter the benchmark definition.
    """
    try:
        import opendssdirect as dss
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install the 'ac' extra to validate IEEE 123 with OpenDSSDirect"
        ) from exc
    master = Path(master_dss).resolve()
    if not master.is_file():
        raise FileNotFoundError(f"OpenDSS master file not found: {master}")
    if load_multiplier < 0 or not 0 <= pv_irradiance <= 1:
        raise ValueError("Load multiplier must be nonnegative and PV irradiance in [0, 1]")
    dss.Basic.ClearAll()
    dss.Text.Command(f"Redirect [{master}]")
    dss.Solution.LoadMult(float(load_multiplier))
    if dss.PVsystems.First():
        while True:
            name = dss.PVsystems.Name()
            dss.Text.Command(f"Edit PVSystem.{name} irradiance={float(pv_irradiance)}")
            if pv_kvar and name in pv_kvar:
                dss.Text.Command(f"Edit PVSystem.{name} kvar={float(pv_kvar[name])}")
            if not dss.PVsystems.Next():
                break
    dss.Solution.Solve()
    losses_watt_var = dss.Circuit.Losses()
    return OpenDSSResult(
        converged=bool(dss.Solution.Converged()),
        node_names=tuple(dss.Circuit.AllNodeNames()),
        voltage_magnitude_pu=np.asarray(dss.Circuit.AllBusMagPu(), dtype=float),
        active_loss_mw=float(losses_watt_var[0]) / 1e6,
        reactive_loss_mvar=float(losses_watt_var[1]) / 1e6,
    )
