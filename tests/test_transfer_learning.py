from pathlib import Path
import copy
import numpy as np
import pytest
import yaml

from tcc_segmentation.transfer_learning import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    validate_transfer_config,
    validate_transfer_pair,
)

ROOT = Path(__file__).resolve().parents[1]


def load_arm(name):
    base = yaml.safe_load((ROOT / "configs/augmentation_d0_a1.yaml").read_text(encoding="utf-8"))
    arm = yaml.safe_load((ROOT / f"configs/{name}").read_text(encoding="utf-8"))
    arm.pop("base_config")

    def merge(a, b):
        out = copy.deepcopy(a)
        for k, v in b.items():
            out[k] = (
                merge(out[k], v)
                if k in out and isinstance(out[k], dict) and isinstance(v, dict)
                else v
            )
        return out

    return merge(base, arm)


def test_transfer_pair_changes_only_encoder_initialization_and_metadata():
    rt, it = load_arm("transfer_learning_random_encoder.yaml"), load_arm("transfer_learning_imagenet_encoder.yaml")
    validate_transfer_pair(rt, it)
    assert rt["model"]["encoder_weights"] is None
    assert it["model"]["encoder_weights"] == "imagenet"
    assert tuple(rt["data"]["normalization"]["mean"]) == IMAGENET_MEAN
    assert tuple(it["data"]["normalization"]["std"]) == IMAGENET_STD


def test_transfer_config_rejects_frozen_encoder():
    cfg = load_arm("transfer_learning_imagenet_encoder.yaml")
    cfg["model"]["freeze_encoder"] = True
    with pytest.raises(ValueError, match="treinável"):
        validate_transfer_config(cfg)


def test_dataset_imagenet_normalization(tmp_path, monkeypatch):
    from PIL import Image
    from tcc_segmentation.data import DevelopmentDataset

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(tmp_path / "images/a.png")
    Image.fromarray(np.ones((2, 2), dtype=np.uint8) * 255).save(tmp_path / "masks/a.png")
    (tmp_path / "manifest.csv").write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "tcc_segmentation.data.dataset.load_development_manifest",
        lambda *_: [
            {
                "canonical_id": "a",
                "split": "train",
                "image_path": "images/a.png",
                "mask_paths": "masks/a.png",
                "mask_policy": "logical_or",
                "height": "2",
                "width": "2",
            }
        ],
    )
    ds = DevelopmentDataset(
        tmp_path / "manifest.csv",
        tmp_path,
        "train",
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
    )
    expected = (np.ones(3, dtype=np.float32) - np.asarray(IMAGENET_MEAN)) / np.asarray(IMAGENET_STD)
    assert np.allclose(ds[0]["image"][:, 0, 0], expected)


def test_panel_display_reverses_dataset_normalization():
    pytest.importorskip("torch")
    from tcc_segmentation.baseline_training import _image_for_display

    class Dataset:
        pass

    ds = Dataset()
    ds.normalization_mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    ds.normalization_std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    original = np.asarray([[[1.0]], [[0.5]], [[0.0]]], dtype=np.float32)
    normalized = (original - ds.normalization_mean) / ds.normalization_std
    shown = _image_for_display(ds, normalized)
    assert np.allclose(shown[0, 0], [255, 127, 0], atol=1)
