"""Standard Baran-Wu IEEE 33-bus radial distribution test system."""

from __future__ import annotations

import numpy as np

from cs_wdro_grid.networks.model import RadialNetwork

_BRANCHES = np.array(
    [
        (1, 2, 0.0922, 0.0470),
        (2, 3, 0.4930, 0.2511),
        (3, 4, 0.3660, 0.1864),
        (4, 5, 0.3811, 0.1941),
        (5, 6, 0.8190, 0.7070),
        (6, 7, 0.1872, 0.6188),
        (7, 8, 0.7114, 0.2351),
        (8, 9, 1.0300, 0.7400),
        (9, 10, 1.0440, 0.7400),
        (10, 11, 0.1966, 0.0650),
        (11, 12, 0.3744, 0.1238),
        (12, 13, 1.4680, 1.1550),
        (13, 14, 0.5416, 0.7129),
        (14, 15, 0.5910, 0.5260),
        (15, 16, 0.7463, 0.5450),
        (16, 17, 1.2890, 1.7210),
        (17, 18, 0.7320, 0.5740),
        (2, 19, 0.1640, 0.1565),
        (19, 20, 1.5042, 1.3554),
        (20, 21, 0.4095, 0.4784),
        (21, 22, 0.7089, 0.9373),
        (3, 23, 0.4512, 0.3083),
        (23, 24, 0.8980, 0.7091),
        (24, 25, 0.8960, 0.7011),
        (6, 26, 0.2030, 0.1034),
        (26, 27, 0.2842, 0.1447),
        (27, 28, 1.0590, 0.9337),
        (28, 29, 0.8042, 0.7006),
        (29, 30, 0.5075, 0.2585),
        (30, 31, 0.9744, 0.9630),
        (31, 32, 0.3105, 0.3619),
        (32, 33, 0.3410, 0.5302),
    ],
    dtype=float,
)

_LOADS_KW_KVAR = np.array(
    [
        (0, 0),
        (100, 60),
        (90, 40),
        (120, 80),
        (60, 30),
        (60, 20),
        (200, 100),
        (200, 100),
        (60, 20),
        (60, 20),
        (45, 30),
        (60, 35),
        (60, 35),
        (120, 80),
        (60, 10),
        (60, 20),
        (60, 20),
        (90, 40),
        (90, 40),
        (90, 40),
        (90, 40),
        (90, 40),
        (90, 50),
        (420, 200),
        (420, 200),
        (60, 25),
        (60, 25),
        (60, 20),
        (120, 70),
        (200, 600),
        (150, 70),
        (210, 100),
        (60, 40),
    ],
    dtype=float,
)


def load_ieee33(base_mva: float = 10.0, base_kv: float = 12.66) -> RadialNetwork:
    """Load IEEE 33 in per unit using three-phase MW/MVA and line-line kV bases."""
    z_base = base_kv**2 / base_mva
    return RadialNetwork(
        name="ieee33",
        base_mva=base_mva,
        base_kv=base_kv,
        branch_from=_BRANCHES[:, 0].astype(int) - 1,
        branch_to=_BRANCHES[:, 1].astype(int) - 1,
        resistance_pu=_BRANCHES[:, 2] / z_base,
        reactance_pu=_BRANCHES[:, 3] / z_base,
        base_load_p_pu=_LOADS_KW_KVAR[:, 0] / (1000.0 * base_mva),
        base_load_q_pu=_LOADS_KW_KVAR[:, 1] / (1000.0 * base_mva),
    )
