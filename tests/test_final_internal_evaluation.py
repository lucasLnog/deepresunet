from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.final_evaluation import (
    CONFIRMATION_TOKEN,
    aggregate_rows,
    apply_primary_holm,
    binary_metric_row,
    deterministic_panel_ids,
    paired_contrast,
    verify_preflight_lock,
)
from tcc_segmentation.final_protocol import EXPECTED_ARM_IDS, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/final_internal_test_protocol_v1.yaml"


def test_confirmation_token_is_explicit_and_runner_requires_it():
    assert CONFIRMATION_TOKEN == "YES_I_UNDERSTAND_THIS_OPENS_THE_INTERNAL_TEST"
    source = (ROOT / "scripts/run_final_internal_test.py").read_text(encoding="utf-8")
    assert "if args.confirm != CONFIRMATION_TOKEN" in source
    assert "thresholds_from_config" not in source
    assert "sweep_validation_thresholds" not in source


def test_panel_selection_is_deterministic_and_performance_blind():
    ids = [f"case_{index:03d}" for index in range(30, 0, -1)]
    first = deterministic_panel_ids(ids, 12)
    second = deterministic_panel_ids(list(reversed(ids)), 12)
    assert first == second
    assert len(first) == 12
    assert len(set(first)) == 12


def test_binary_metrics_include_overlap_and_boundary():
    reference = np.zeros((32, 32), dtype=bool)
    reference[8:24, 8:24] = True
    same = binary_metric_row(reference, reference)
    assert same["dice"] == pytest.approx(1.0)
    assert same["iou"] == pytest.approx(1.0)
    assert same["boundary_iou"] == pytest.approx(1.0)
    empty = binary_metric_row(np.zeros_like(reference), reference)
    assert empty["dice"] == 0.0
    assert empty["precision"] == 0.0
    assert empty["recall"] == 0.0


def test_aggregation_and_paired_contrast_are_per_image():
    rows = []
    for index in range(8):
        for arm, dice in (("control", 0.4 + index / 100), ("candidate", 0.5 + index / 100)):
            rows.append(
                {
                    "canonical_id": f"id_{index}",
                    "arm_id": arm,
                    "threshold_policy": "selected",
                    "dice": dice,
                    "iou": dice - 0.1,
                    "precision": dice,
                    "recall": dice,
                    "specificity": 0.9,
                    "accuracy": 0.8,
                    "balanced_accuracy": (dice + 0.9) / 2,
                    "boundary_iou": dice - 0.2,
                }
            )
    summary = aggregate_rows(rows)
    assert len(summary) == 2
    contrast = paired_contrast(
        rows,
        contrast_id="example",
        candidate="candidate",
        control="control",
        threshold_policy="selected",
        family="secondary",
        bootstrap_samples=100,
        permutation_samples=100,
        seed=7,
    )
    dice = next(row for row in contrast if row["metric"] == "dice")
    iou = next(row for row in contrast if row["metric"] == "iou")
    assert dice["mean_difference"] == pytest.approx(0.1)
    assert dice["images"] == 8
    assert dice["permutation_p_raw"] is not None
    assert iou["permutation_p_raw"] is None


def test_holm_applies_only_to_five_primary_dice_rows():
    rows = []
    for index, p_value in enumerate((0.01, 0.02, 0.03, 0.04, 0.05)):
        rows.append(
            {
                "family": "primary",
                "metric": "dice",
                "contrast_id": f"c{index}",
                "permutation_p_raw": p_value,
                "permutation_p_holm": None,
            }
        )
    rows.append(
        {
            "family": "exploratory",
            "metric": "dice",
            "contrast_id": "boundary",
            "permutation_p_raw": 0.001,
            "permutation_p_holm": None,
        }
    )
    apply_primary_holm(rows)
    assert all(row["permutation_p_holm"] is not None for row in rows[:5])
    assert rows[-1]["permutation_p_holm"] is None


def test_preflight_verifier_rejects_protocol_or_artifact_changes(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    sam = project / "sam.pth"
    sam.write_bytes(b"sam")
    manifest_hash = "a" * 64
    arms, locked_arms = [], []
    for arm_id in sorted(EXPECTED_ARM_IDS):
        if arm_id == "DRIFT_SAM_BPM":
            arms.append(
                {
                    "id": arm_id,
                    "role": "x",
                    "derived_from": "DRIFT",
                    "config": "config.yaml",
                    "threshold": 0.18,
                    "threshold_source": "validation_only",
                    "sam_checkpoint": "sam.pth",
                }
            )
            locked_arms.append(
                {
                    "id": arm_id,
                    "config_sha256": sha256_file(config),
                    "sam_checkpoint_sha256": sha256_file(sam),
                }
            )
        else:
            run_id = f"{arm_id}_seed42"
            checkpoint = project / "runs" / run_id / "best_checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(arm_id.encode())
            arms.append(
                {
                    "id": arm_id,
                    "role": "x",
                    "config": "config.yaml",
                    "run_id": run_id,
                    "run_directory": f"runs/{run_id}",
                    "checkpoint": "best_checkpoint.pt",
                    "threshold": 0.2,
                    "threshold_source": "validation_only",
                }
            )
            locked_arms.append(
                {
                    "id": arm_id,
                    "config_sha256": sha256_file(config),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    protocol["data"]["manifest_sha256"] = manifest_hash
    protocol["arms"] = arms
    protocol_path = project / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    lock = {
        "status": "passed",
        "scientific_data_accessed": False,
        "inference_executed": False,
        "protocol_sha256": sha256_file(protocol_path),
        "test_manifest": {"sha256": manifest_hash},
        "arms": locked_arms,
    }
    lock_path = project / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    verify_preflight_lock(project, protocol_path, lock_path)
    config.write_text("x: 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Configuração alterada"):
        verify_preflight_lock(project, protocol_path, lock_path)
