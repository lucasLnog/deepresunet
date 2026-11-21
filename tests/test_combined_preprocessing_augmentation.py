from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.preprocessing_experiment import make_preprocessing_pair_transform
from tcc_segmentation.combined_preprocessing_augmentation import (
    P1A1TrainTransform,
    make_p1_a1_train_transform,
    make_p1_validation_transform,
    validate_p1_a1_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_combination_matches_frozen_d0_p1_and_a1() -> None:
    config = load("configs/combined_preprocessing_augmentation_p1_a1.yaml")
    d0 = load("configs/baseline_d0.yaml")
    p1 = load("configs/preprocessing_dp1_shades_of_gray.yaml")
    a1 = load("configs/augmentation_d0_a1.yaml")
    audit = load("configs/augmentation_audit.yaml")
    validate_p1_a1_config(config, d0_config=d0, p1_config=p1, a1_config=a1, audit_config=audit)
    assert config["data"]["color_preprocessing"] == "shades_of_gray"
    assert config["data"]["color_preprocessing_parameters"] == {"minkowski_p": 6.0}
    assert config["data"]["augmentation"] == "A1"
    assert config["model"]["architecture"] == "deep_res_unet"
    assert config["comparison_plan"]["primary_contrast"] == "P1A1_minus_A1"
    assert config["comparison_plan"]["test_access_allowed"] is False


def test_combination_rejects_a2_and_pretrained_weights() -> None:
    config = load("configs/combined_preprocessing_augmentation_p1_a1.yaml")
    config["data"]["augmentation"] = "A2"
    with pytest.raises(ValueError, match="A1"):
        validate_p1_a1_config(config)
    config = load("configs/combined_preprocessing_augmentation_p1_a1.yaml")
    config["model"]["pretrained"] = True
    with pytest.raises(ValueError, match="zero"):
        validate_p1_a1_config(config)


def test_validation_is_exactly_p1_without_augmentation() -> None:
    combined = load("configs/combined_preprocessing_augmentation_p1_a1.yaml")
    p1 = load("configs/preprocessing_dp1_shades_of_gray.yaml")
    image = np.full((48, 64, 3), [175, 120, 75], dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:24, 20:38] = 1
    observed = make_p1_validation_transform(combined)(image, mask)
    expected = make_preprocessing_pair_transform(p1)(image, mask)
    assert np.array_equal(observed[0], expected[0])
    assert np.array_equal(observed[1], expected[1])


class _ColorRecorder:
    def __call__(self, image, mask):
        return np.clip(image.astype(np.int16) + 7, 0, 255).astype(np.uint8), mask.copy()


class _AugmentationRecorder:
    def __init__(self):
        self.received = None
        self.epoch = None

    def __call__(self, image, mask):
        self.received = image.copy()
        return image.copy(), mask.copy()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def get_epoch_stats(self):
        return {"epoch_seed": 42000 + int(self.epoch or 0)}


def test_training_applies_color_before_augmentation_and_forwards_epoch() -> None:
    color = _ColorRecorder()
    augmentation = _AugmentationRecorder()
    transform = P1A1TrainTransform(color, augmentation)
    image = np.full((8, 10, 3), 20, dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    result_image, result_mask = transform(image, mask)
    assert np.all(augmentation.received == 27)
    assert np.array_equal(result_image, augmentation.received)
    assert np.array_equal(result_mask, mask)
    transform.set_epoch(8)
    assert transform.get_epoch_stats()["epoch_seed"] == 42008


def test_real_training_contract_when_albumentations_is_available() -> None:
    pytest.importorskip("albumentations")
    transform = make_p1_a1_train_transform(load("configs/combined_preprocessing_augmentation_p1_a1.yaml"))
    transform.set_epoch(3)
    image = np.full((48, 64, 3), [175, 120, 75], dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:24, 20:38] = 1
    result_image, result_mask = transform(image, mask)
    assert result_image.shape == (384, 512, 3)
    assert result_mask.shape == (384, 512)
    assert set(np.unique(result_mask)).issubset({0, 1})
    assert transform.get_epoch_stats()["epoch_seed"] == 42003
