from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.tcc_segmentation.pipeline_sanity import (  # noqa: E402
    batch_diagnostics,
    deterministic_prefix_indices,
    prediction_max_abs_difference,
)
from src.tcc_segmentation.models import DeepResUNet  # noqa: E402
from src.tcc_segmentation.training import BCEDiceLoss  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_day4_config_never_names_protected_manifest() -> None:
    text = (ROOT / "configs/pipeline_sanity.yaml").read_text(encoding="utf-8").casefold()
    assert "dfuc_test.csv" not in text
    assert "fuseg_external" not in text


def test_deepresunet_logits_and_approved_loss_are_compatible() -> None:
    model = DeepResUNet(base_channels=2)
    images = torch.rand(2, 3, 32, 48)
    masks = (torch.rand(2, 1, 32, 48) > 0.9).float()
    logits = model(images)
    components = BCEDiceLoss(0.5, 0.5).components(logits, masks)
    assert logits.shape == masks.shape
    assert all(torch.isfinite(value) for value in components.values())
    components["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_batch_diagnostics_and_prefix_are_deterministic() -> None:
    batch = {
        "canonical_id": ["a", "b"],
        "image": torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1),
        "mask": torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1),
    }
    report = batch_diagnostics(batch)
    assert report["mask_unique_values"] == [0.0, 1.0]
    assert report["all_finite"] is True
    assert deterministic_prefix_indices(10, 3) == [0, 1, 2]


def test_prediction_roundtrip_comparison() -> None:
    first = [{"canonical_id": "x", "logits": np.asarray([[1.0, 2.0]])}]
    second = [{"canonical_id": "x", "logits": np.asarray([[1.0, 2.0]])}]
    assert prediction_max_abs_difference(first, second) == 0.0
