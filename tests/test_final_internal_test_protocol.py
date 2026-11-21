from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tcc_segmentation.final_protocol import (
    EXPECTED_ARM_IDS,
    audit_runtime_artifacts,
    load_and_validate_protocol,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/final_internal_test_protocol_v1.yaml"


def test_final_protocol_is_frozen_before_test_access():
    protocol = load_and_validate_protocol(PROTOCOL)
    assert protocol["data"]["expected_records"] == 294
    assert protocol["data"]["access_policy"] == "one_batched_run_after_successful_preflight"
    assert {arm["id"] for arm in protocol["arms"]} == EXPECTED_ARM_IDS
    assert protocol["representative_run_policy"]["seed"] == 42
    assert protocol["representative_run_policy"]["ensemble_allowed"] is False


def test_primary_secondary_and_exploratory_roles_are_separate():
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert len(protocol["primary_contrasts"]) == 5
    assert protocol["statistics"]["multiplicity"] == "Holm_across_five_primary_Dice_contrasts"
    assert protocol["exploratory_contrasts"] == [
        {
            "id": "boundary_loss",
            "candidate": "DRIFTB",
            "control": "DRIFT",
            "interpretation": "exploratory_post_registration_not_a_primary_claim",
        }
    ]
    assert "MedSAM" in protocol["excluded_from_internal_test"]


def test_all_thresholds_are_validation_frozen_and_fixed_point_five_is_sensitivity():
    protocol = load_and_validate_protocol(PROTOCOL)
    assert all("validation" in arm["threshold_source"] for arm in protocol["arms"])
    assert protocol["metrics"]["primary"].endswith("validation_selected_threshold")
    assert protocol["metrics"]["sensitivity"].endswith("fixed_threshold_0_5")
    assert protocol["representative_run_policy"]["threshold_reselection_allowed"] is False


def test_audit_hashes_files_without_scientific_access(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    manifest = project / "manifests/splits/dfuc_test.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("opaque manifest bytes\n", encoding="utf-8")
    sam = project / "models/sam/sam.pth"
    sam.parent.mkdir(parents=True)
    sam.write_bytes(b"sam")
    config = project / "config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")

    arms = []
    for arm_id in sorted(EXPECTED_ARM_IDS):
        if arm_id == "DRIFT_SAM_BPM":
            arms.append(
                {
                    "id": arm_id,
                    "role": "planned_SAM_effect",
                    "derived_from": "DRIFT",
                    "config": "config.yaml",
                    "threshold": 0.18,
                    "threshold_source": "validation_only",
                    "sam_checkpoint": "models/sam/sam.pth",
                }
            )
        else:
            run_id = f"{arm_id}_seed42"
            checkpoint = project / f"runs/{run_id}/best_checkpoint.pt"
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

    source = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    source["data"]["manifest"] = "manifests/splits/dfuc_test.csv"
    source["data"]["manifest_sha256"] = sha256_file(manifest)
    source["arms"] = arms
    temp_protocol = project / "protocol.yaml"
    temp_protocol.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    output = project / "lock.json"
    report = audit_runtime_artifacts(project, temp_protocol, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert saved["scientific_data_accessed"] is False
    assert saved["inference_executed"] is False
    assert saved["test_manifest"]["contents_read"] is False
    assert len(saved["arms"]) == 10
