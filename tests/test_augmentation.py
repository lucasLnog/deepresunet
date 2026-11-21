from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.data.augmentation import build_augmentation_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/augmentation_audit.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    mask = np.zeros((96, 128), dtype=np.uint8)
    mask[24:72, 36:92] = 1
    image[mask == 1] = (220, 80, 40)
    return image, mask


def test_protocol_is_frozen_and_train_only() -> None:
    config = load_config()
    a1 = config["pipelines"]["A1"]
    assert config["runtime"]["allowed_split"] == "train"
    assert config["runtime"]["model_training_allowed"] is False
    assert config["visualization"]["selected_before_augmentation_results"] is True
    assert len(config["visualization"]["fixed_train_ids"]) == 8
    assert a1["affine"]["rotation_degrees"] == [-20.0, 20.0]
    assert a1["affine"]["scale"] == [0.90, 1.10]
    assert a1["affine"]["shear_degrees"] == [0.0, 0.0]


def test_a0_returns_independent_unchanged_arrays() -> None:
    image, mask = synthetic_pair()
    transform = build_augmentation_pipeline("A0", load_config()["pipelines"], seed=42)
    result_image, result_mask, trace = transform.apply_with_trace(image, mask)
    assert np.array_equal(result_image, image)
    assert np.array_equal(result_mask, mask)
    assert result_image is not image and result_mask is not mask
    assert trace == []


def test_unknown_pipeline_and_nonzero_shear_are_rejected() -> None:
    config = load_config()["pipelines"]
    with pytest.raises(ValueError, match="desconhecido"):
        build_augmentation_pipeline("A3", config, seed=42)

    pytest.importorskip("albumentations")
    altered = deepcopy(config)
    altered["A1"]["affine"]["shear_degrees"] = [-5.0, 5.0]
    with pytest.raises(ValueError, match="Shear"):
        build_augmentation_pipeline("A1", altered, seed=42)


@pytest.mark.parametrize("name", ["A1", "A2"])
def test_pair_transform_is_reproducible_and_preserves_mask_contract(name: str) -> None:
    pytest.importorskip("albumentations")
    image, mask = synthetic_pair()
    config = load_config()["pipelines"]
    first = build_augmentation_pipeline(name, config, seed=2026)
    second = build_augmentation_pipeline(name, config, seed=2026)
    image_a, mask_a, trace_a = first.apply_with_trace(image, mask)
    image_b, mask_b, trace_b = second.apply_with_trace(image, mask)
    assert np.array_equal(image_a, image_b)
    assert np.array_equal(mask_a, mask_b)
    assert trace_a == trace_b
    assert image_a.shape[:2] == mask_a.shape == mask.shape
    assert image_a.dtype == np.uint8
    assert set(np.unique(mask_a)).issubset({0, 1})


def test_a2_uses_at_most_one_photometric_transform() -> None:
    pytest.importorskip("albumentations")
    image, mask = synthetic_pair()
    transform = build_augmentation_pipeline("A2", load_config()["pipelines"], seed=7)
    photometric = {"RandomBrightnessContrast", "RandomGamma", "HueSaturationValue"}
    for seed in range(20):
        transform.set_seed(seed)
        _, _, trace = transform.apply_with_trace(image, mask)
        names = [record["transform"] for record in trace if record["transform"] in photometric]
        assert len(names) <= 1
