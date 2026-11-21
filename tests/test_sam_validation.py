from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.sam_validation import (
    BASELINE_VARIANT,
    METRICS,
    SAM_VARIANTS,
    VARIANTS,
    boundary_iou,
    holm_adjust,
    paired_comparisons,
    select_sam_variant,
    validate_day13_config,
)


ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load((ROOT / "configs/sam_validation.yaml").read_text(encoding="utf-8"))


def test_day13_contract_is_frozen_before_full_validation():
    config = load_config()
    validate_day13_config(config)
    assert config["data"]["expected_records"] == 294
    assert config["network"]["threshold"] == 0.18
    assert tuple(config["evaluation"]["variants"]) == SAM_VARIANTS
    assert config["evaluation"]["threshold_optimization_allowed"] is False
    assert config["statistics"]["bootstrap_samples"] == 10_000
    assert config["statistics"]["permutation_samples"] == 50_000


def test_day13_rejects_test_manifest():
    config = load_config()
    config["data"]["manifest"] = "manifests/splits/dfuc_test.csv"
    with pytest.raises(ValueError, match="protegida"):
        validate_day13_config(config)


def test_day13_rejects_changed_threshold_and_candidate_order():
    config = load_config()
    config["network"]["threshold"] = 0.5
    with pytest.raises(ValueError, match="0,18"):
        validate_day13_config(config)
    config = load_config()
    config["evaluation"]["variants"] = list(reversed(SAM_VARIANTS))
    with pytest.raises(ValueError, match="candidatos"):
        validate_day13_config(config)


def test_boundary_iou_is_one_for_equal_masks_and_zero_for_disjoint_masks():
    first = np.zeros((64, 64), dtype=bool)
    first[10:30, 10:30] = True
    second = np.zeros_like(first)
    second[35:55, 35:55] = True
    assert boundary_iou(first, first) == pytest.approx(1.0)
    assert boundary_iou(first, second) == pytest.approx(0.0)
    assert boundary_iou(np.zeros_like(first), np.zeros_like(first)) == pytest.approx(1.0)


def test_selection_uses_highest_dice_and_conservative_frozen_tie_break():
    summary = [
        {"variant": "SAM_box", "dice_mean": 0.70000},
        {"variant": "SAM_mask_input", "dice_mean": 0.70005},
        {"variant": "SAM_box_plus_mask", "dice_mean": 0.69},
    ]
    result = select_sam_variant(summary, tolerance=0.0001, tie_break_order=SAM_VARIANTS)
    assert result["selected_variant"] == "SAM_box"
    assert result["plateau_candidates"] == ["SAM_box", "SAM_mask_input"]


def _row(canonical_id, variant, dice):
    row = {
        "canonical_id": canonical_id,
        "variant": variant,
        "network_seconds": 0.1,
        "sam_seconds": 0.2,
    }
    for metric in METRICS:
        row[metric] = dice
    return row


def test_paired_comparisons_keep_image_as_statistical_unit():
    rows = []
    values = {
        BASELINE_VARIANT: [0.2, 0.4, 0.6],
        "SAM_box": [0.3, 0.5, 0.7],
        "SAM_mask_input": [0.1, 0.3, 0.5],
        "SAM_box_plus_mask": [0.2, 0.4, 0.6],
    }
    for variant in VARIANTS:
        for index, value in enumerate(values[variant]):
            rows.append(_row(f"case_{index}", variant, value))
    comparisons, per_image = paired_comparisons(
        rows, bootstrap_samples=200, permutation_samples=500, seed=13
    )
    dice = {row["comparison"]: row for row in comparisons if row["metric"] == "dice"}
    assert dice[f"SAM_box_minus_{BASELINE_VARIANT}"]["mean_difference"] == pytest.approx(0.1)
    assert dice[f"SAM_mask_input_minus_{BASELINE_VARIANT}"]["mean_difference"] == pytest.approx(
        -0.1
    )
    assert dice[f"SAM_box_plus_mask_minus_{BASELINE_VARIANT}"]["mean_difference"] == pytest.approx(
        0.0
    )
    assert len(per_image) == 3


def test_holm_adjustment_is_monotone_in_sorted_pvalues():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.2)
