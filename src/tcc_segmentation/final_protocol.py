"""Validação e auditoria sem inferência do protocolo final de teste interno."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_ARM_IDS = {
    "D0",
    "P1",
    "A1",
    "P1A1",
    "DRRT",
    "DRIT",
    "DRIF",
    "DRIFT",
    "DRIFT_SAM_BPM",
    "DRIFTB",
}
EXPECTED_PRIMARY_CONTRASTS = {
    ("P1", "D0"),
    ("A1", "D0"),
    ("DRIT", "DRRT"),
    ("DRIFT", "DRIF"),
    ("DRIFT_SAM_BPM", "DRIFT"),
}
PROHIBITED_PRIMARY_IDS = {"P2", "A2", "SAM_box", "SAM_mask_input", "MedSAM"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_protocol(path: str | Path) -> dict[str, Any]:
    protocol_path = Path(path).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_internal_test_access":
        raise ValueError("O protocolo ainda não está congelado.")
    data = protocol["data"]
    if data["split"] != "internal_test" or data["expected_records"] != 294:
        raise ValueError("Partição final ou quantidade esperada incorreta.")
    if len(data["manifest_sha256"]) != 64:
        raise ValueError("SHA-256 do manifesto de teste inválido.")
    if protocol["representative_run_policy"]["seed"] != 42:
        raise ValueError("A execução representativa deve permanecer na seed 42.")

    arms = protocol["arms"]
    arm_ids = [arm["id"] for arm in arms]
    if len(arm_ids) != len(set(arm_ids)):
        raise ValueError("Há braços duplicados no protocolo.")
    if set(arm_ids) != EXPECTED_ARM_IDS:
        raise ValueError(f"Matriz de braços inesperada: {sorted(arm_ids)}")
    for arm in arms:
        threshold = float(arm["threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold inválido para {arm['id']}: {threshold}")
        if "validation" not in arm["threshold_source"]:
            raise ValueError(f"Threshold de {arm['id']} não veio da validação.")
        if arm["id"] != "DRIFT_SAM_BPM" and "seed42" not in arm["run_id"]:
            raise ValueError(f"{arm['id']} não usa a execução representativa seed 42.")

    primary = {(item["candidate"], item["control"]) for item in protocol["primary_contrasts"]}
    if primary != EXPECTED_PRIMARY_CONTRASTS:
        raise ValueError("Família de contrastes primários foi alterada.")
    if any(identifier in PROHIBITED_PRIMARY_IDS for pair in primary for identifier in pair):
        raise ValueError("Um candidato excluído entrou na família primária.")
    if protocol["statistics"]["multiplicity"] != "Holm_across_five_primary_Dice_contrasts":
        raise ValueError("Correção de multiplicidade primária incorreta.")
    if protocol["execution_lock"]["model_selection_on_test_allowed"] is not False:
        raise ValueError("Seleção pelo teste deve permanecer proibida.")
    return protocol


def audit_runtime_artifacts(
    project_root: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Confere e sela arquivos, sem abrir imagens, máscaras ou calcular métricas."""
    root = Path(project_root).resolve()
    path = Path(protocol_path).resolve()
    protocol = load_and_validate_protocol(path)
    report: dict[str, Any] = {
        "status": "passed",
        "scientific_data_accessed": False,
        "inference_executed": False,
        "protocol_path": str(path),
        "protocol_sha256": sha256_file(path),
        "test_manifest": {},
        "arms": [],
    }

    manifest = root / protocol["data"]["manifest"]
    observed_manifest_hash = sha256_file(manifest)
    expected_manifest_hash = protocol["data"]["manifest_sha256"]
    if observed_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            f"Manifesto de teste divergente: {observed_manifest_hash} != {expected_manifest_hash}"
        )
    report["test_manifest"] = {
        "path": str(manifest),
        "sha256": observed_manifest_hash,
        "contents_read": False,
    }

    seen_checkpoint_paths: set[Path] = set()
    for arm in protocol["arms"]:
        item: dict[str, Any] = {"id": arm["id"], "config": arm["config"]}
        config_path = root / arm["config"]
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        item["config_sha256"] = sha256_file(config_path)
        if arm["id"] == "DRIFT_SAM_BPM":
            sam_path = root / arm["sam_checkpoint"]
            if not sam_path.is_file():
                raise FileNotFoundError(sam_path)
            item.update(
                {
                    "derived_from": arm["derived_from"],
                    "sam_checkpoint": str(sam_path),
                    "sam_checkpoint_sha256": sha256_file(sam_path),
                }
            )
        else:
            run_dir = root / arm["run_directory"]
            checkpoint_path = run_dir / arm["checkpoint"]
            if run_dir.name != arm["run_id"]:
                raise RuntimeError(f"run_id e pasta divergem para {arm['id']}")
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            if checkpoint_path in seen_checkpoint_paths:
                raise RuntimeError(f"Checkpoint repetido indevidamente: {checkpoint_path}")
            seen_checkpoint_paths.add(checkpoint_path)
            item.update(
                {
                    "run_id": arm["run_id"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
        report["arms"].append(item)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
