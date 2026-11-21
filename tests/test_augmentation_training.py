from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.augmentation_experiment import (
    EpochSeededSafeAugmentation,
    make_augmentation_pair_transform,
    validate_augmentation_training_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("path", "experiment_id", "policy"),
    [
        ("configs/augmentation_d0_a1.yaml", "D0-A1", "A1"),
        ("configs/augmentation_d0_a2.yaml", "D0-A2", "A2"),
    ],
)
def test_augmentation_arms_match_frozen_controls(path, experiment_id, policy) -> None:
    config = load(path)
    d0 = load("configs/baseline_d0.yaml")
    audit = load("configs/augmentation_audit.yaml")
    validate_augmentation_training_config(config, d0_config=d0, audit_config=audit)
    assert config["experiment_id"] == experiment_id
    assert config["data"]["augmentation"] == policy
    assert config["augmentation_pipelines"] == audit["pipelines"]
    assert config["augmentation_safety"]["retention_threshold_filters_training"] is False


def test_validation_rejects_using_fifty_percent_as_training_filter() -> None:
    config = load("configs/augmentation_d0_a1.yaml")
    config["augmentation_safety"]["retention_threshold_filters_training"] = True
    with pytest.raises(ValueError, match="50%"):
        validate_augmentation_training_config(config)


class _AlwaysEmpty:
    def __call__(self, image, mask):
        return image.copy(), np.zeros_like(mask)


class _EmptyThenValid:
    def __init__(self):
        self.calls = 0

    def __call__(self, image, mask):
        self.calls += 1
        return (
            (image.copy(), np.zeros_like(mask)) if self.calls == 1 else (image.copy(), mask.copy())
        )


def _safe_transform_with_fake(fake) -> EpochSeededSafeAugmentation:
    transform = object.__new__(EpochSeededSafeAugmentation)
    transform.name = "A1"
    transform.height = 48
    transform.width = 64
    transform.pipelines = {}
    transform.augmentation_seed_base = 42000
    transform.max_resamples = 3
    from tcc_segmentation.data import ResizePad

    transform._resize = ResizePad(48, 64)
    transform._pipeline = fake
    transform._epoch = 1
    transform._stats = {
        "epoch_seed": 42001,
        "samples_seen": 0,
        "empty_mask_rejections": 0,
        "resample_attempts": 0,
        "identity_fallbacks": 0,
    }
    return transform


def test_empty_mask_is_resampled_and_valid_attempt_is_kept() -> None:
    transform = _safe_transform_with_fake(_EmptyThenValid())
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:20, 20:30] = 1
    _, result = transform(image, mask)
    assert np.any(result)
    assert transform.get_epoch_stats()["empty_mask_rejections"] == 1
    assert transform.get_epoch_stats()["resample_attempts"] == 1
    assert transform.get_epoch_stats()["identity_fallbacks"] == 0


def test_identity_fallback_after_three_resamples() -> None:
    transform = _safe_transform_with_fake(_AlwaysEmpty())
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:20, 20:30] = 1
    result_image, result_mask = transform(image, mask)
    assert np.array_equal(result_image, image)
    assert np.array_equal(result_mask, mask)
    stats = transform.get_epoch_stats()
    assert stats["empty_mask_rejections"] == 4
    assert stats["resample_attempts"] == 3
    assert stats["identity_fallbacks"] == 1


def test_real_transform_contract_when_dependencies_are_available() -> None:
    pytest.importorskip("albumentations")
    transform = make_augmentation_pair_transform(load("configs/augmentation_d0_a2.yaml"))
    transform.set_epoch(7)
    image = np.full((48, 64, 3), [170, 120, 80], dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:30, 18:42] = 1
    result_image, result_mask = transform(image, mask)
    assert result_image.shape == (384, 512, 3)
    assert result_mask.shape == (384, 512)
    assert set(np.unique(result_mask)).issubset({0, 1})
    assert transform.get_epoch_stats()["epoch_seed"] == 42007
