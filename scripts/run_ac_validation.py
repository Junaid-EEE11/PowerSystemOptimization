"""Validate the unscaled IEEE 33 base state with nonlinear AC power flow."""

from __future__ import annotations

from cs_wdro_grid.networks.ieee33 import load_ieee33
from cs_wdro_grid.validation.ac import solve_ac_power_flow


def main() -> int:
    network = load_ieee33()
    result = solve_ac_power_flow(network, network.base_load_p_pu, network.base_load_q_pu)
    print(
        f"converged={result.converged} iterations={result.iterations} "
        f"v_min={result.voltage_magnitude_pu.min():.6f} "
        f"loss_mw={result.loss_pu * network.base_mva:.6f}"
    )
    return 0 if result.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
