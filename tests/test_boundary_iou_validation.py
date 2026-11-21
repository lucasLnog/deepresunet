from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.sam_validation import boundary_iou


ROOT = Path(__file__).resolve().parents[1]


def test_day14b_protocol_is_validation_only_and_frozen():
    config = yaml.safe_load(
        (ROOT / "configs/boundary_iou_validation_protocol.yaml").read_text(encoding="utf-8")
    )
    assert config["data"]["split"] == "validation"
    assert config["data"]["expected_records"] == 294
    assert config["comparison"]["seeds"] == [42, 1337, 2026]
    assert config["comparison"]["new_training"] is False
    assert config["comparison"]["threshold_optimization_allowed"] is False
    assert config["evaluation"]["boundary_dilation_ratio"] == pytest.approx(0.02)
    assert config["evaluation"]["role"] == "secondary_descriptive_only"


def test_boundary_iou_penalizes_shifted_contour():
    reference = np.zeros((384, 512), dtype=bool)
    reference[120:260, 180:340] = True
    equal = boundary_iou(reference, reference, 0.02)
    shifted = np.zeros_like(reference)
    shifted[130:270, 190:350] = True
    shifted_score = boundary_iou(shifted, reference, 0.02)
    assert equal == pytest.approx(1.0)
    assert 0.0 < shifted_score < equal


def test_evaluation_scripts_block_protected_partitions_and_reuse_threshold():
    source = (ROOT / "scripts/evaluate_boundary_iou_validation.py").read_text(encoding="utf-8")
    assert 'for marker in ("test", "external", "fuseg")' in source
    assert 'expected = read_csv(run_dir / "validation_per_image_selected.csv")' in source
    assert "probabilities >= threshold" in source
    assert "len(dataset) != 294" in source


def test_comparison_is_paired_and_does_not_select_model():
    source = (ROOT / "scripts/compare_boundary_iou_validation.py").read_text(encoding="utf-8")
    assert "if base_ids != cand_ids" in source
    assert "paired_bootstrap_ci" in source
    assert "paired_sign_flip_pvalue" in source
    assert "secondary_descriptive; no model or threshold selection" in source
