from __future__ import annotations

import json
from pathlib import Path

import yaml

from tcc_segmentation.external_protocol import (
    PROHIBITED_EXTERNAL_IDS,
    audit_external_runtime_artifacts,
    audit_fuseg_filename_structure,
    load_and_validate_external_protocol,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/final_external_fuseg_protocol_v1.yaml"


def test_external_protocol_is_frozen_before_FUSeg_access():
    protocol = load_and_validate_external_protocol(PROTOCOL)
    assert protocol["data"]["expected_canonical_units"] == 962
    assert protocol["primary_model"]["id"] == "DRIFT"
    assert protocol["secondary_model"]["id"] == "DRIFT_SAM_BPM"
    assert protocol["primary_model"]["threshold"] == 0.18
    assert protocol["primary_model"]["test_time_augmentation"] is False
    assert set(protocol["excluded_models"]) == PROHIBITED_EXTERNAL_IDS


def test_external_dataset_counts_and_duplicate_policy_are_fixed():
    protocol = load_and_validate_external_protocol(PROTOCOL)
    data = protocol["data"]
    assert data["raw_records"] == 1010
    assert data["excluded_records"] == 24
    assert data["valid_reference_records"] == 986
    assert data["exact_duplicate_groups_after_exclusion"] == 23
    assert data["duplicate_reference_policy"].endswith("weight_unit_once")
    assert data["exact_cross_dataset_groups"] == 0
    assert data["perceptual_cross_dataset_candidates"] == 0


def test_external_protocol_prohibits_adaptation_and_selection():
    protocol = load_and_validate_external_protocol(PROTOCOL)
    lock = protocol["execution_lock"]
    assert lock["fine_tuning_on_FUSeg_allowed"] is False
    assert lock["threshold_reselection_allowed"] is False
    assert lock["checkpoint_reselection_allowed"] is False
    assert lock["model_selection_on_FUSeg_allowed"] is False
    assert lock["test_time_augmentation_allowed"] is False
    assert lock["result_driven_rerun_allowed"] is False


def test_filename_structure_uses_names_only(tmp_path: Path):
    data_root = tmp_path / "data"
    images = data_root / "Dataset/FUSeg_Images"
    masks = data_root / "Dataset/FUSeg_Masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for index in range(1, 1011):
        name = f"{index:04d}.png"
        (images / name).write_bytes(b"not-an-image")
        (masks / name).write_bytes(b"not-a-mask")
    report = audit_fuseg_filename_structure(data_root)
    assert report["image_files"] == 1010
    assert report["mask_files"] == 1010
    assert report["paired_names"] == 1010
    assert report["file_contents_read"] is False


def test_preflight_hashes_artifacts_without_inference(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    data_root = tmp_path / "data"
    images = data_root / "Dataset/FUSeg_Images"
    masks = data_root / "Dataset/FUSeg_Masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for index in range(1, 1011):
        name = f"{index:04d}.png"
        (images / name).write_bytes(b"image")
        (masks / name).write_bytes(b"mask")

    files = {
        "manifest": project / "manifest.csv",
        "exclusions": project / "exclusions.csv",
        "audit": project / "audit.md",
        "primary_config": project / "primary.yaml",
        "primary_checkpoint": project / "runs/DRIFT_seed42/best_checkpoint.pt",
        "sam_config": project / "sam.yaml",
        "sam_checkpoint": project / "sam.pth",
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    source = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    source["data"].update(
        {
            "manifest": "manifest.csv",
            "manifest_sha256": sha256_file(files["manifest"]),
            "exclusions_file": "exclusions.csv",
            "exclusions_sha256": sha256_file(files["exclusions"]),
            "cross_dataset_audit": "audit.md",
            "cross_dataset_audit_sha256": sha256_file(files["audit"]),
        }
    )
    source["primary_model"].update(
        {
            "config": "primary.yaml",
            "run_id": "DRIFT_seed42",
            "run_directory": "runs/DRIFT_seed42",
            "checkpoint": "best_checkpoint.pt",
        }
    )
    source["secondary_model"].update(
        {
            "config": "sam.yaml",
            "sam_checkpoint": "sam.pth",
        }
    )
    protocol_path = project / "protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    output = project / "lock.json"
    report = audit_external_runtime_artifacts(project, data_root, protocol_path, output)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert saved["scientific_file_contents_accessed"] is False
    assert saved["inference_executed"] is False
    assert saved["manifest"]["contents_read"] is False
    assert saved["primary_model"]["id"] == "DRIFT"
    assert saved["secondary_model"]["id"] == "DRIFT_SAM_BPM"
