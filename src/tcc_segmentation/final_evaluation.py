"""Contrato, dados e estatística da avaliação final interna congelada."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .sam_validation import (
    boundary_iou,
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
)
from .final_protocol import load_and_validate_protocol, sha256_file


CONFIRMATION_TOKEN = "YES_I_UNDERSTAND_THIS_OPENS_THE_INTERNAL_TEST"
REQUIRED_MANIFEST_FIELDS = {
    "dataset",
    "canonical_id",
    "image_path",
    "mask_paths",
    "mask_policy",
    "split",
    "height",
    "width",
    "foreground_fraction",
    "foreground_area_bin",
}
METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "boundary_iou",
)


def load_final_test_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_records: int,
) -> list[dict[str, str]]:
    """Abre exclusivamente o manifesto final após a autorização explícita."""
    manifest = Path(path).resolve()
    if manifest.name != "dfuc_test.csv":
        raise ValueError("A avaliação final exige exatamente dfuc_test.csv")
    observed_hash = sha256_file(manifest)
    if observed_hash.casefold() != expected_sha256.casefold():
        raise RuntimeError("Hash do manifesto final difere do protocolo congelado")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_MANIFEST_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colunas ausentes no manifesto final: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_records:
        raise ValueError(f"Esperados {expected_records} casos; encontrados {len(rows)}")
    if {row["split"].strip().casefold() for row in rows} != {"test"}:
        raise ValueError("O manifesto final deve conter somente o split test")
    if {row["dataset"] for row in rows} != {"DFUC2022"}:
        raise ValueError("O teste interno deve usar somente DFUC2022")
    ids = [row["canonical_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs canônicos duplicados no teste interno")
    return rows


def verify_preflight_lock(
    project_root: str | Path,
    protocol_path: str | Path,
    lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_and_validate_protocol(protocol_file)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("status") != "passed":
        raise RuntimeError("O preflight não está aprovado")
    if (
        lock.get("scientific_data_accessed") is not False
        or lock.get("inference_executed") is not False
    ):
        raise RuntimeError("O lacre não representa um preflight limpo")
    if lock.get("protocol_sha256") != sha256_file(protocol_file):
        raise RuntimeError("O protocolo mudou depois do preflight")
    if lock.get("test_manifest", {}).get("sha256") != protocol["data"]["manifest_sha256"]:
        raise RuntimeError("O manifesto do lacre difere do protocolo")

    locked_by_id = {item["id"]: item for item in lock.get("arms", [])}
    if len(locked_by_id) != len(protocol["arms"]):
        raise RuntimeError("Quantidade de braços do lacre incorreta")
    for arm in protocol["arms"]:
        locked = locked_by_id.get(arm["id"])
        if locked is None:
            raise RuntimeError(f"Braço ausente do lacre: {arm['id']}")
        config_path = root / arm["config"]
        if sha256_file(config_path) != locked["config_sha256"]:
            raise RuntimeError(f"Configuração alterada após o preflight: {arm['id']}")
        if arm["id"] == "DRIFT_SAM_BPM":
            sam_path = root / arm["sam_checkpoint"]
            if sha256_file(sam_path) != locked["sam_checkpoint_sha256"]:
                raise RuntimeError("Checkpoint SAM alterado após o preflight")
        else:
            checkpoint = root / arm["run_directory"] / arm["checkpoint"]
            if sha256_file(checkpoint) != locked["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint alterado após o preflight: {arm['id']}")
    return protocol, lock


def deterministic_panel_ids(ids: Sequence[str], count: int = 12) -> list[str]:
    """Seleciona IDs uniformemente na ordem lexical, sem consultar desempenho."""
    ordered = sorted(str(value) for value in ids)
    if count < 1 or len(ordered) < count:
        raise ValueError("Quantidade inválida de casos para o painel")
    indices = np.linspace(0, len(ordered) - 1, num=count, dtype=int)
    return [ordered[int(index)] for index in indices]


def binary_metric_row(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError("Predição e referência devem ser máscaras 2D de mesmo shape")
    tp = int((pred & ref).sum())
    fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum())
    tn = int((~pred & ~ref).sum())

    def divide(num: float, den: float, fallback: float = 1.0) -> float:
        return float(num / den) if den else float(fallback)

    dice = divide(2 * tp, 2 * tp + fp + fn)
    iou = divide(tp, tp + fp + fn)
    precision = divide(tp, tp + fp, 0.0 if tp + fn else 1.0)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    accuracy = divide(tp + tn, tp + tn + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "boundary_iou": boundary_iou(pred, ref, 0.02),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm_id"]), str(row["threshold_policy"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (arm_id, policy), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "arm_id": arm_id,
            "threshold_policy": policy,
            "images": len(group),
            "zero_dice_images": sum(float(row["dice"]) == 0.0 for row in group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        result.append(item)
    return result


def paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    contrast_id: str,
    candidate: str,
    control: str,
    threshold_policy: str,
    family: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["threshold_policy"] == threshold_policy]
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in selected:
        by_arm.setdefault(str(row["arm_id"]), {})[str(row["canonical_id"])] = row
    if candidate not in by_arm or control not in by_arm:
        raise ValueError(f"Braço ausente no contraste {contrast_id}")
    ids = sorted(by_arm[candidate])
    if set(ids) != set(by_arm[control]):
        raise ValueError(f"IDs não pareados no contraste {contrast_id}")
    output: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        differences = [
            float(by_arm[candidate][item][metric]) - float(by_arm[control][item][metric])
            for item in ids
        ]
        low, high = paired_bootstrap_ci(differences, bootstrap_samples, seed + metric_index)
        p_value = (
            paired_sign_flip_pvalue(differences, permutation_samples, seed + 10_000 + metric_index)
            if metric == "dice"
            else None
        )
        output.append(
            {
                "family": family,
                "contrast_id": contrast_id,
                "candidate": candidate,
                "control": control,
                "threshold_policy": threshold_policy,
                "metric": metric,
                "images": len(ids),
                "mean_difference": statistics.fmean(differences),
                "median_difference": statistics.median(differences),
                "ci95_low": low,
                "ci95_high": high,
                "improved": sum(value > 0 for value in differences),
                "equal": sum(value == 0 for value in differences),
                "worsened": sum(value < 0 for value in differences),
                "permutation_p_raw": p_value,
                "permutation_p_holm": None,
            }
        )
    return output


def apply_primary_holm(rows: list[dict[str, Any]]) -> None:
    primary_dice = {
        row["contrast_id"]: float(row["permutation_p_raw"])
        for row in rows
        if row["family"] == "primary" and row["metric"] == "dice"
    }
    if len(primary_dice) != 5:
        raise ValueError("A família primária deve conter cinco contrastes de Dice")
    adjusted = holm_adjust(primary_dice)
    for row in rows:
        if row["family"] == "primary" and row["metric"] == "dice":
            row["permutation_p_holm"] = adjusted[row["contrast_id"]]
