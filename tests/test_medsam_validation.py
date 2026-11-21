from pathlib import Path
import inspect

import numpy as np
import pytest
import yaml

from tcc_segmentation.sam_validation import METRICS
from tcc_segmentation.medsam_refinement import refine_prediction_with_medsam_box
from tcc_segmentation.medsam_validation import (
    BASELINE_VARIANT,
    MEDSAM_VARIANT,
    VARIANTS,
    aggregate_variant_rows,
    apply_advance_rule,
    paired_comparisons,
    validate_day13b_config,
)

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    return yaml.safe_load(
        (ROOT / "configs/medsam_validation.yaml").read_text(encoding="utf-8")
    )


def test_day13b_contract_is_frozen_before_results():
    config = load_config()
    validate_day13b_config(config)
    assert config["data"]["expected_records"] == 294
    assert config["network"]["threshold"] == 0.18
    assert config["medsam"]["preprocessing"] == "official_square_1024_minmax"
    assert config["medsam"]["prompt_mode"] == "box"
    assert tuple(config["evaluation"]["variants"]) == VARIANTS


def test_day13b_rejects_protected_manifest_and_ground_truth_prompt():
    config = load_config()
    config["data"]["manifest"] = "manifests/splits/dfuc_test.csv"
    with pytest.raises(ValueError, match="protegida"):
        validate_day13b_config(config)
    config = load_config()
    config["prompts"]["ground_truth_allowed"] = True
    with pytest.raises(ValueError, match="somente da DR-IFT"):
        validate_day13b_config(config)


def test_day13b_rejects_changed_medsam_hash_or_preprocessing():
    config = load_config()
    config["medsam"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        validate_day13b_config(config)
    config = load_config()
    config["medsam"]["preprocessing"] = "sam_predictor"
    with pytest.raises(ValueError, match="oficial"):
        validate_day13b_config(config)


def _row(canonical_id, variant, value):
    row = {
        "canonical_id": canonical_id,
        "variant": variant,
        "network_seconds": 0.01,
        "postprocess_seconds": 0.2,
    }
    for metric in METRICS:
        row[metric] = value
    return row


def _rows(medsam_gain):
    values = {
        BASELINE_VARIANT: [0.4, 0.5, 0.6],
        "SAM_box": [0.3, 0.4, 0.5],
        "SAM_mask_input": [0.1, 0.1, 0.1],
        "SAM_box_plus_mask": [0.39, 0.49, 0.59],
        MEDSAM_VARIANT: [0.4 + medsam_gain, 0.5 + medsam_gain, 0.6 + medsam_gain],
    }
    return [
        _row(f"case_{index}", variant, value)
        for variant in VARIANTS
        for index, value in enumerate(values[variant])
    ]


def test_summary_and_comparisons_preserve_all_variants_and_pairing():
    rows = _rows(0.02)
    summary = aggregate_variant_rows(rows)
    comparisons, differences = paired_comparisons(
        rows, bootstrap_samples=200, permutation_samples=500, seed=13
    )
    assert [row["variant"] for row in summary] == list(VARIANTS)
    assert len(differences) == 3
    primary_family = [
        row for row in comparisons if row["metric"] == "dice" and row["primary_family"]
    ]
    assert len(primary_family) == 4
    direct = next(
        row
        for row in comparisons
        if row["comparison"] == "MedSAM_box_minus_SAM_box" and row["metric"] == "dice"
    )
    assert direct["mean_difference"] == pytest.approx(0.12)
    assert direct["permutation_p_holm"] is None


def test_advance_rule_requires_both_practical_gain_and_positive_interval():
    config = load_config()
    effect = {
        "comparison": f"{MEDSAM_VARIANT}_minus_{BASELINE_VARIANT}",
        "metric": "dice",
        "mean_difference": 0.015,
        "ci95_low": 0.003,
        "ci95_high": 0.025,
    }
    decision = apply_advance_rule([effect], config)
    assert decision["advance_to_locked_internal_test"] is True
    effect["ci95_low"] = -0.001
    decision = apply_advance_rule([effect], config)
    assert decision["advance_to_locked_internal_test"] is False
    effect["ci95_low"] = 0.001
    effect["mean_difference"] = 0.009
    decision = apply_advance_rule([effect], config)
    assert decision["advance_to_locked_internal_test"] is False


def test_advance_rule_threshold_cannot_be_changed_after_results():
    config = load_config()
    config["advance_rule"]["minimum_mean_dice_gain"] = 0.0
    with pytest.raises(ValueError, match="0,01"):
        validate_day13b_config(config)


def test_medsam_prompt_function_cannot_receive_reference_mask():
    parameters = inspect.signature(refine_prediction_with_medsam_box).parameters
    assert "reference" not in parameters
    assert "ground_truth" not in parameters


def test_empty_drift_prediction_returns_empty_without_calling_medsam():
    rgb = np.zeros((12, 20, 3), dtype=np.uint8)
    probabilities = np.zeros((12, 20), dtype=np.float32)
    prediction, audit = refine_prediction_with_medsam_box(
        model=None,
        rgb_image=rgb,
        network_probabilities=probabilities,
        network_threshold=0.18,
        max_components_safety=64,
    )
    assert prediction.shape == probabilities.shape
    assert not prediction.any()
    assert audit["component_count"] == 0
    assert audit["empty_prediction_policy_applied"] is True
    assert audit["ground_truth_used_for_prompt"] is False
