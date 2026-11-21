from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.tcc_segmentation.data.dataset import DevelopmentDataset
from src.tcc_segmentation.data.manifest import (
    ManifestIntegrityError,
    ProtectedPartitionError,
    load_development_manifest,
    resolve_dataset_path,
)
from src.tcc_segmentation.data.masks import load_binary_mask, load_canonical_mask
from src.tcc_segmentation.data.transforms import ResizePad


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "manifests" / "splits"
requires_local_dataset = pytest.mark.skipif(
    not (ROOT / "Dataset").is_dir(),
    reason="Este teste requer imagens e máscaras do conjunto local.",
)


def test_development_manifests_have_expected_counts() -> None:
    train = load_development_manifest(SPLITS / "dfuc_train.csv", "train")
    validation = load_development_manifest(SPLITS / "dfuc_validation.csv", "validation")
    assert len(train) == 1373
    assert len(validation) == 294


def test_test_manifest_is_rejected_before_opening() -> None:
    with pytest.raises(ProtectedPartitionError, match="protegido"):
        load_development_manifest(SPLITS / "dfuc_test.csv", "train")


@pytest.mark.parametrize("split", ["test", "external", "fuseg"])
def test_protected_expected_split_is_rejected(split: str) -> None:
    with pytest.raises(ProtectedPartitionError):
        load_development_manifest(SPLITS / "dfuc_train.csv", split)


def test_path_escape_is_rejected() -> None:
    with pytest.raises(ValueError, match="fora da raiz"):
        resolve_dataset_path(ROOT, "../arquivo-fora-do-projeto.png")


def test_renamed_manifest_without_approved_hash_record_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "dfuc_train.csv"
    copied.write_bytes((SPLITS / "dfuc_train.csv").read_bytes())
    with pytest.raises(ManifestIntegrityError, match="Resumo de hashes"):
        load_development_manifest(copied, "train")


@requires_local_dataset
def test_real_training_sample_is_binary_and_aligned() -> None:
    dataset = DevelopmentDataset(
        SPLITS / "dfuc_train.csv",
        ROOT,
        expected_split="train",
        transform=ResizePad(height=192, width=256),
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 192, 256)
    assert sample["mask"].shape == (1, 192, 256)
    assert sample["image"].dtype == np.float32
    assert sample["mask"].dtype == np.float32
    assert set(np.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})
    assert sample["split"] == "train"


@requires_local_dataset
def test_logical_or_matches_duplicate_sources() -> None:
    rows = load_development_manifest(SPLITS / "dfuc_train.csv", "train")
    row = next(item for item in rows if int(item["source_count"]) > 1)
    paths = [resolve_dataset_path(ROOT, part) for part in row["mask_paths"].split("|")]
    canonical = load_canonical_mask(paths, row["mask_policy"], threshold=128)
    expected = np.logical_or.reduce([load_binary_mask(path, 128) for path in paths])
    assert np.array_equal(canonical, expected)


def test_resize_pad_keeps_image_and_mask_geometry() -> None:
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 5:15] = 1
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[mask == 1] = (255, 255, 255)
    resized_image, resized_mask = ResizePad(height=20, width=20)(image, mask)
    assert resized_image.shape == (20, 20, 3)
    assert resized_mask.shape == (20, 20)
    assert set(np.unique(resized_mask).tolist()) == {0, 1}
    # A regiao branca e a referencia continuam ocupando os mesmos pixels centrais.
    assert np.all(resized_image[resized_mask == 1] > 0)
