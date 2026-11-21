from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/baseline_d0.yaml").read_text(encoding="utf-8"))


def test_d0_config_is_central_and_has_no_protected_manifest_path() -> None:
    config = load_config()
    text = (ROOT / "configs/baseline_d0.yaml").read_text(encoding="utf-8").casefold()
    assert config["experiment_id"] == "D0"
    assert config["model"]["architecture"] == "deep_res_unet"
    assert config["model"]["encoder_weights"] is None
    assert config["data"]["augmentation"] == "none"
    assert config["data"]["color_preprocessing"] == "none"
    assert "dfuc_test.csv" not in text
    assert "fuseg_external" not in text


def test_resolution_training_and_threshold_rules_are_frozen() -> None:
    config = load_config()
    assert (config["data"]["height"], config["data"]["width"]) == (384, 512)
    assert config["loss"]["bce_weight"] == config["loss"]["dice_weight"] == 0.5
    assert config["training"]["batch_size"] == 4
    assert config["training"]["oom_fallback_batch_size"] == 2
    assert config["training"]["max_epochs"] == 100
    assert config["training"]["checkpoint_threshold"] == 0.5
    assert config["threshold_selection"] == {
        "start": 0.05,
        "stop": 0.95,
        "step": 0.01,
        "objective": "macro_dice",
        "tie_break": "closest_to_0.5",
    }


def test_fixed_visual_ids_belong_only_to_validation() -> None:
    config = load_config()
    with (ROOT / "manifests/splits/dfuc_validation.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        validation_ids = {row["canonical_id"] for row in csv.DictReader(stream)}
    fixed = config["visualization"]["fixed_validation_ids"]
    assert len(fixed) == len(set(fixed)) == 8
    assert set(fixed).issubset(validation_ids)


def test_day5_pure_metric_helpers() -> None:
    torch = pytest.importorskip("torch")
    from src.tcc_segmentation.baseline_training import (  # noqa: PLC0415
        _metrics_from_counts,
        thresholds_from_config,
    )
    import numpy as np

    thresholds = thresholds_from_config(load_config())
    assert len(thresholds) == 91
    assert thresholds[0] == pytest.approx(0.05)
    assert thresholds[-1] == pytest.approx(0.95)
    metrics = _metrics_from_counts(
        np.asarray([8]), np.asarray([2]), np.asarray([2]), np.asarray([88])
    )
    assert metrics["dice"][0] == pytest.approx(0.8)
    assert metrics["iou"][0] == pytest.approx(2 / 3)
