from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tcc_segmentation.external_evaluation import (
    CONFIRMATION_TOKEN,
    average_reference_metric_rows,
    bootstrap_mean_ci,
    load_external_manifest,
)
from tcc_segmentation.external_protocol import sha256_file


def write_manifest(path: Path) -> None:
    fields = [
        "dataset",
        "canonical_id",
        "image_path",
        "mask_paths",
        "mask_policy",
        "source_count",
        "reference_count",
        "height",
        "width",
        "foreground_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "FUSeg",
                "canonical_id": "FUSeg_0001",
                "image_path": "Dataset/FUSeg_Images/0001.png",
                "mask_paths": "Dataset/FUSeg_Masks/0001.png",
                "mask_policy": "reference_mean",
                "source_count": 1,
                "reference_count": 1,
                "height": 512,
                "width": 512,
                "foreground_fraction": 0.01,
            }
        )


def test_confirmation_token_is_specific_to_external_evaluation() -> None:
    assert CONFIRMATION_TOKEN == "YES_I_UNDERSTAND_THIS_OPENS_FUSEG_EXTERNAL_EVALUATION"
    assert "INTERNAL_TEST" not in CONFIRMATION_TOKEN


def test_external_manifest_requires_exact_file_hash_and_policy(tmp_path: Path) -> None:
    path = tmp_path / "fuseg_external_canonical.csv"
    write_manifest(path)
    rows = load_external_manifest(path, expected_sha256=sha256_file(path), expected_units=1)
    assert rows[0]["canonical_id"] == "FUSeg_0001"
    with pytest.raises(RuntimeError):
        load_external_manifest(path, expected_sha256="0" * 64, expected_units=1)


def test_reference_metrics_are_averaged_before_unit_weighting() -> None:
    base = {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "tn": 6,
        "dice": 2 / 3,
        "iou": 0.5,
        "precision": 2 / 3,
        "recall": 2 / 3,
        "specificity": 6 / 7,
        "accuracy": 0.8,
        "balanced_accuracy": 16 / 21,
        "boundary_iou": 0.4,
    }
    other = {key: (value / 2 if key not in {"tn"} else value) for key, value in base.items()}
    result = average_reference_metric_rows([base, other], reference_count=2)
    assert result["reference_count"] == 2
    assert result["dice"] == pytest.approx(0.5)
    assert result["tn"] == pytest.approx(6.0)


def test_bootstrap_ci_is_deterministic_and_contains_constant_mean() -> None:
    first = bootstrap_mean_ci([0.7] * 20, 100, 20260929)
    second = bootstrap_mean_ci([0.7] * 20, 100, 20260929)
    assert first == second
    assert first[0] == pytest.approx(0.7)
    assert first[1] == pytest.approx(0.7)


def test_runner_does_not_contain_selection_or_training_loops() -> None:
    runner = Path(__file__).resolve().parents[1] / "scripts/run_final_external_fuseg.py"
    text = runner.read_text(encoding="utf-8")
    assert "optimizer" not in text.casefold()
    assert "threshold_curve" not in text
    assert "load_external_manifest" in text
    assert "all_predictions_completed_before_summary" in text
