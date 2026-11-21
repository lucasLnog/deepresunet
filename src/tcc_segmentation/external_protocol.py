"""Validacao e auditoria sem inferencia do protocolo externo FUSeg."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_PRIMARY_ID = "DRIFT"
EXPECTED_SECONDARY_ID = "DRIFT_SAM_BPM"
PROHIBITED_EXTERNAL_IDS = {
    "DRIFTB",
    "D0",
    "P1",
    "A1",
    "P1A1",
    "DRRT",
    "DRIT",
    "DRIF",
    "MedSAM",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_external_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_external_evaluation_access":
        raise ValueError("O protocolo externo ainda nao esta congelado.")

    data = protocol["data"]
    if data["dataset"] != "FUSeg_canonical" or data["split"] != "external_evaluation":
        raise ValueError("Dataset ou papel externo incorreto.")
    expected = {
        "raw_records": 1010,
        "excluded_records": 24,
        "valid_reference_records": 986,
        "expected_canonical_units": 962,
        "exact_duplicate_groups_after_exclusion": 23,
        "exact_cross_dataset_groups": 0,
        "perceptual_cross_dataset_candidates": 0,
    }
    for key, value in expected.items():
        if int(data[key]) != value:
            raise ValueError(f"Valor externo congelado incorreto para {key}.")
    for key in ("manifest_sha256", "exclusions_sha256", "cross_dataset_audit_sha256"):
        if len(str(data[key])) != 64:
            raise ValueError(f"SHA-256 invalido para {key}.")
    if (
        data["duplicate_reference_policy"]
        != "average_metrics_across_references_then_weight_unit_once"
    ):
        raise ValueError("Politica de referencias duplicadas foi alterada.")

    primary = protocol["primary_model"]
    secondary = protocol["secondary_model"]
    if primary["id"] != EXPECTED_PRIMARY_ID or secondary["id"] != EXPECTED_SECONDARY_ID:
        raise ValueError("Modelos externos inesperados.")
    if int(primary["seed"]) != 42 or "seed42" not in primary["run_id"]:
        raise ValueError("DRIFT externo deve permanecer na seed representativa 42.")
    if float(primary["threshold"]) != 0.18 or float(secondary["network_threshold"]) != 0.18:
        raise ValueError("Threshold externo deve permanecer em 0,18.")
    if "validation" not in primary["threshold_source"].casefold():
        raise ValueError("Threshold principal nao veio da validacao DFUC.")
    if secondary["derived_from"] != EXPECTED_PRIMARY_ID or secondary["sam_mode"] != "box_plus_mask":
        raise ValueError("Analise SAM externa foi alterada.")
    if primary["test_time_augmentation"] is not False:
        raise ValueError("TTA deve permanecer desativada.")

    locks = protocol["execution_lock"]
    forbidden_true = (
        "inference_before_preflight_allowed",
        "fine_tuning_on_FUSeg_allowed",
        "threshold_reselection_allowed",
        "checkpoint_reselection_allowed",
        "model_selection_on_FUSeg_allowed",
        "test_time_augmentation_allowed",
        "result_driven_rerun_allowed",
        "panels_selected_by_performance_allowed",
    )
    if any(locks[key] is not False for key in forbidden_true):
        raise ValueError("Uma protecao da avaliacao externa foi removida.")
    if set(protocol["excluded_models"]) != PROHIBITED_EXTERNAL_IDS:
        raise ValueError("Lista de modelos excluidos do FUSeg foi alterada.")
    return protocol


def audit_fuseg_filename_structure(data_root: str | Path) -> dict[str, Any]:
    """Verifica somente nomes/extensoes, sem abrir conteudo de imagem ou mascara."""
    root = Path(data_root).resolve()
    image_dir = root / "Dataset" / "FUSeg_Images"
    mask_dir = root / "Dataset" / "FUSeg_Masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Pastas FUSeg_Images e FUSeg_Masks nao encontradas.")
    images = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    masks = sorted(path.name for path in mask_dir.iterdir() if path.is_file())
    expected = [f"{index:04d}.png" for index in range(1, 1011)]
    if images != expected:
        raise RuntimeError("Nomes das 1.010 imagens FUSeg diferem da estrutura congelada.")
    if masks != expected:
        raise RuntimeError("Nomes das 1.010 mascaras FUSeg diferem da estrutura congelada.")
    return {
        "image_directory": str(image_dir),
        "mask_directory": str(mask_dir),
        "image_files": len(images),
        "mask_files": len(masks),
        "paired_names": len(set(images) & set(masks)),
        "extensions": [".png"],
        "file_contents_read": False,
    }


def audit_external_runtime_artifacts(
    project_root: str | Path,
    data_root: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Sela protocolo, manifesto e pesos sem abrir dados cientificos."""
    root = Path(project_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_and_validate_external_protocol(protocol_file)
    data = protocol["data"]

    manifest = root / data["manifest"]
    exclusions = root / data["exclusions_file"]
    cross_audit = root / data["cross_dataset_audit"]
    for path, expected in (
        (manifest, data["manifest_sha256"]),
        (exclusions, data["exclusions_sha256"]),
        (cross_audit, data["cross_dataset_audit_sha256"]),
    ):
        observed = sha256_file(path)
        if observed.casefold() != str(expected).casefold():
            raise RuntimeError(f"Hash divergente: {path}")

    primary = protocol["primary_model"]
    primary_config = root / primary["config"]
    primary_run = root / primary["run_directory"]
    primary_checkpoint = primary_run / primary["checkpoint"]
    if primary_run.name != primary["run_id"]:
        raise RuntimeError("run_id e pasta DRIFT divergem.")
    for path in (primary_config, primary_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    secondary = protocol["secondary_model"]
    sam_config = root / secondary["config"]
    sam_checkpoint = root / secondary["sam_checkpoint"]
    for path in (sam_config, sam_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    report: dict[str, Any] = {
        "status": "passed",
        "scientific_file_contents_accessed": False,
        "inference_executed": False,
        "protocol_path": str(protocol_file),
        "protocol_sha256": sha256_file(protocol_file),
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
            "contents_read": False,
        },
        "exclusions": {"path": str(exclusions), "sha256": sha256_file(exclusions)},
        "cross_dataset_audit": {"path": str(cross_audit), "sha256": sha256_file(cross_audit)},
        "dataset_structure": audit_fuseg_filename_structure(data_root),
        "primary_model": {
            "id": primary["id"],
            "run_id": primary["run_id"],
            "config": str(primary_config),
            "config_sha256": sha256_file(primary_config),
            "checkpoint": str(primary_checkpoint),
            "checkpoint_sha256": sha256_file(primary_checkpoint),
        },
        "secondary_model": {
            "id": secondary["id"],
            "derived_from": secondary["derived_from"],
            "config": str(sam_config),
            "config_sha256": sha256_file(sam_config),
            "sam_checkpoint": str(sam_checkpoint),
            "sam_checkpoint_sha256": sha256_file(sam_checkpoint),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
