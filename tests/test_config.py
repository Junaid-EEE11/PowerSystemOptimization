from pathlib import Path

import pytest

from cs_wdro_grid.config import ConfigurationError, load_config, validate_config


def test_quick_config_inherits_base() -> None:
    config = load_config(Path("configs/quick.yaml"))
    assert config["project"]["mode"] == "quick"
    assert config["network"]["name"] == "ieee33"
    assert config["split"]["test"] == pytest.approx(0.15)


def test_full_mode_rejects_synthetic_source() -> None:
    config = load_config(Path("configs/quick.yaml"))
    config["project"]["mode"] = "full"
    with pytest.raises(ConfigurationError, match="may not use synthetic"):
        validate_config(config)


def test_m5_requires_at_least_one_positive_radius() -> None:
    config = load_config(Path("configs/quick.yaml"))
    config["optimization"]["epsilon_grid"] = [0.0]
    with pytest.raises(ConfigurationError, match="positive M5 radius"):
        validate_config(config)
