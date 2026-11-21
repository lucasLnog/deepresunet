from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.preprocessing_experiment import (
    make_preprocessing_pair_transform,
    validate_preprocessing_training_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "experiment_id", "method"),
    [
        ("configs/preprocessing_dp1_shades_of_gray.yaml", "D-P1", "shades_of_gray"),
        ("configs/preprocessing_dp2_clahe_lstar.yaml", "D-P2", "clahe_l_star"),
    ],
)
def test_preprocessing_arms_are_authorized(path: str, experiment_id: str, method: str) -> None:
    config = load(path)
    validate_preprocessing_training_config(config)
    assert config["experiment_id"] == experiment_id
    assert config["data"]["color_preprocessing"] == method
    assert config["data"]["augmentation"] == "none"
    assert config["seed"] == 42


@pytest.mark.parametrize(
    "path",
    ["configs/preprocessing_dp1_shades_of_gray.yaml", "configs/preprocessing_dp2_clahe_lstar.yaml"],
)
def test_everything_except_preprocessing_matches_d0(path: str) -> None:
    baseline = load("configs/baseline_d0.yaml")
    candidate = load(path)
    for key in ("model", "loss", "training", "threshold_selection"):
        assert candidate[key] == baseline[key]
    assert candidate["seed"] == baseline["seed"]
    assert (
        candidate["visualization"]["fixed_validation_ids"]
        == baseline["visualization"]["fixed_validation_ids"]
    )
    for key in (
        "train_manifest",
        "validation_manifest",
        "height",
        "width",
        "mask_threshold",
        "image_scale",
        "augmentation",
        "protected_partitions",
    ):
        assert candidate["data"][key] == baseline["data"][key]


def test_p1_transform_preserves_binary_mask_contract() -> None:
    config = load("configs/preprocessing_dp1_shades_of_gray.yaml")
    transform = make_preprocessing_pair_transform(config)
    image = np.full((48, 64, 3), [170, 120, 80], dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:30, 18:42] = 1
    transformed_image, transformed_mask = transform(image, mask)
    assert transformed_image.shape == (384, 512, 3)
    assert transformed_mask.shape == (384, 512)
    assert transformed_image.dtype == np.uint8
    assert set(np.unique(transformed_mask)).issubset({0, 1})


def test_p2_transform_contract_when_opencv_is_available() -> None:
    pytest.importorskip("cv2")
    config = load("configs/preprocessing_dp2_clahe_lstar.yaml")
    transform = make_preprocessing_pair_transform(config)
    image = np.full((48, 64, 3), [170, 120, 80], dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    transformed_image, transformed_mask = transform(image, mask)
    assert transformed_image.shape == (384, 512, 3)
    assert transformed_mask.shape == (384, 512)
    assert set(np.unique(transformed_mask)).issubset({0, 1})


def test_unregistered_combination_is_rejected() -> None:
    config = load("configs/preprocessing_dp1_shades_of_gray.yaml")
    config["data"]["color_preprocessing"] = "shades_of_gray_plus_clahe"
    with pytest.raises(ValueError, match="exige"):
        validate_preprocessing_training_config(config)


def test_output_directories_are_distinct() -> None:
    p1 = load("configs/preprocessing_dp1_shades_of_gray.yaml")
    p2 = load("configs/preprocessing_dp2_clahe_lstar.yaml")
    assert p1["runtime"]["output_root"] != p2["runtime"]["output_root"]
    assert p1["runtime"]["run_prefix"] != p2["runtime"]["run_prefix"]
