from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.analyze_validation_thresholds import build_threshold_grid, select_threshold
from scripts.run_development_baseline import validate_baseline_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    ROOT / "configs/baselines/unet_resnet34_random.yaml",
    ROOT / "configs/baselines/deep_res_unet_historical.yaml",
)


@pytest.mark.parametrize("path", CONFIG_PATHS)
def test_baseline_configs_are_development_only(path: Path) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_baseline_config(config)
    assert config["data"]["train_manifest"].endswith("dfuc_train.csv")
    assert config["data"]["validation_manifest"].endswith("dfuc_validation.csv")
    assert all(value is False for value in config["protection"].values())


@pytest.mark.parametrize("protected", ("dfuc_test.csv", "fuseg_external.csv"))
def test_baseline_config_rejects_protected_manifest(protected: str) -> None:
    config = yaml.safe_load(CONFIG_PATHS[0].read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["data"]["validation_manifest"] = f"manifests/splits/{protected}"
    with pytest.raises(ValueError, match="validation_manifest"):
        validate_baseline_config(config)


def test_threshold_grid_and_tie_break_are_preregistered() -> None:
    grid = build_threshold_grid(0.05, 0.95, 0.01)
    assert len(grid) == 91
    assert 0.5 in grid
    curve = [
        {"threshold": 0.40, "dice_mean": 0.8000},
        {"threshold": 0.49, "dice_mean": 0.79995},
        {"threshold": 0.51, "dice_mean": 0.79995},
    ]
    result = select_threshold(curve, plateau_tolerance=0.0001)
    assert result["selected_threshold"] == pytest.approx(0.49)
