from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.run_overfit_smoke import gate_passed, streaks_from_history, validate_config


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load((ROOT / "configs" / "overfit_smoke.yaml").read_text(encoding="utf-8"))


def test_overfit_config_is_strictly_train_only() -> None:
    config = load_config()
    validate_config(config)
    assert config["data"]["manifest"].endswith("dfuc_train.csv")
    assert config["scientific_result"] is False
    assert all(value is False for value in config["protection"].values())


@pytest.mark.parametrize(
    ("field", "value"),
    [("split", "validation"), ("manifest", "manifests/splits/dfuc_test.csv")],
)
def test_overfit_config_rejects_nontraining_data(field: str, value: str) -> None:
    config = deepcopy(load_config())
    config["data"][field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_overfit_config_rejects_augmentation() -> None:
    config = deepcopy(load_config())
    config["data"]["augmentation"] = True
    with pytest.raises(ValueError, match="Augmentation"):
        validate_config(config)


def test_gate_requires_full_consecutive_streak() -> None:
    assert gate_passed(max_target_streak=5, target_patience=5)
    assert not gate_passed(max_target_streak=1, target_patience=5)


def test_streaks_are_recovered_from_history(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text(
        "memorization_dice_mean\n0.96\n0.971\n0.972\n0.95\n0.98\n",
        encoding="utf-8",
    )
    assert streaks_from_history(history, target_dice=0.97) == (1, 2)
