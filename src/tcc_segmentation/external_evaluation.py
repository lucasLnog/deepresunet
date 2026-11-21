"""Contrato e estatistica da avaliacao externa final no FUSeg."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .external_protocol import load_and_validate_external_protocol, sha256_file
from .final_evaluation import METRICS, aggregate_rows, paired_contrast


CONFIRMATION_TOKEN = "YES_I_UNDERSTAND_THIS_OPENS_FUSEG_EXTERNAL_EVALUATION"
REQUIRED_MANIFEST_FIELDS = {
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
}


def load_external_manifest(
    path: str | Path, *, expected_sha256: str, expected_units: int
) -> list[dict[str, str]]:
    """Abre o manifesto FUSeg somente depois da confirmacao explicita."""
    manifest = Path(path).resolve()
    if manifest.name != "fuseg_external_canonical.csv":
        raise ValueError("A avaliacao externa exige fuseg_external_canonical.csv")
    if sha256_file(manifest).casefold() != expected_sha256.casefold():
        raise RuntimeError("Hash do manifesto FUSeg difere do protocolo congelado")
    with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_MANIFEST_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colunas ausentes no manifesto FUSeg: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_units:
        raise ValueError(f"Esperadas {expected_units} unidades; encontradas {len(rows)}")
    if {row["dataset"] for row in rows} != {"FUSeg"}:
        raise ValueError("O manifesto externo deve conter somente FUSeg")
    ids = [row["canonical_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs canonicos duplicados no manifesto externo")
    for row in rows:
        if row["mask_policy"] != "reference_mean":
            raise ValueError("Politica externa deve ser reference_mean")
        masks = [value for value in row["mask_paths"].split("|") if value]
        if len(masks) != int(row["reference_count"]):
            raise ValueError(f"Numero de referencias inconsistente em {row['canonical_id']}")
    return rows


def verify_external_preflight_lock(
    project_root: str | Path,
    protocol_path: str | Path,
    lock_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_and_validate_external_protocol(protocol_file)
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("status") != "passed":
        raise RuntimeError("O preflight externo nao esta aprovado")
    if lock.get("scientific_file_contents_accessed") is not False:
        raise RuntimeError("O preflight externo nao foi cego aos conteudos cientificos")
    if lock.get("inference_executed") is not False:
        raise RuntimeError("O lacre registra inferencia antes da avaliacao")
    if lock.get("protocol_sha256") != sha256_file(protocol_file):
        raise RuntimeError("O protocolo externo mudou depois do preflight")

    data = protocol["data"]
    checks = (
        ("manifest", root / data["manifest"], data["manifest_sha256"]),
        ("exclusions", root / data["exclusions_file"], data["exclusions_sha256"]),
        (
            "cross_dataset_audit",
            root / data["cross_dataset_audit"],
            data["cross_dataset_audit_sha256"],
        ),
    )
    for key, path, expected in checks:
        if lock.get(key, {}).get("sha256") != expected or sha256_file(path) != expected:
            raise RuntimeError(f"Artefato externo alterado apos o preflight: {key}")

    structure = lock.get("dataset_structure", {})
    if any(
        int(structure.get(key, -1)) != 1010 for key in ("image_files", "mask_files", "paired_names")
    ):
        raise RuntimeError("Estrutura FUSeg do lacre esta incompleta")
    if structure.get("file_contents_read") is not False:
        raise RuntimeError("O preflight abriu conteudos do FUSeg")

    primary = protocol["primary_model"]
    locked_primary = lock.get("primary_model", {})
    primary_config = root / primary["config"]
    primary_checkpoint = root / primary["run_directory"] / primary["checkpoint"]
    if locked_primary.get("run_id") != primary["run_id"]:
        raise RuntimeError("Run DRIFT do lacre difere do protocolo")
    if sha256_file(primary_config) != locked_primary.get("config_sha256"):
        raise RuntimeError("Configuracao DRIFT mudou apos o preflight")
    if sha256_file(primary_checkpoint) != locked_primary.get("checkpoint_sha256"):
        raise RuntimeError("Checkpoint DRIFT mudou apos o preflight")

    secondary = protocol["secondary_model"]
    locked_secondary = lock.get("secondary_model", {})
    if sha256_file(root / secondary["config"]) != locked_secondary.get("config_sha256"):
        raise RuntimeError("Configuracao SAM mudou apos o preflight")
    if sha256_file(root / secondary["sam_checkpoint"]) != locked_secondary.get(
        "sam_checkpoint_sha256"
    ):
        raise RuntimeError("Checkpoint SAM mudou apos o preflight")
    return protocol, lock


def average_reference_metric_rows(
    rows: Sequence[Mapping[str, Any]], *, reference_count: int
) -> dict[str, Any]:
    """Resume as referencias de uma imagem antes de dar peso um a unidade."""
    if not rows or len(rows) != reference_count:
        raise ValueError("Quantidade de linhas de referencia inesperada")
    output: dict[str, Any] = {"reference_count": reference_count}
    numeric = ("tp", "fp", "fn", "tn", *METRICS)
    for key in numeric:
        output[key] = statistics.fmean(float(row[key]) for row in rows)
    return output


def bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or samples < 1:
        raise ValueError("Amostra ou numero de reamostragens invalido")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 512):
        size = min(512, samples - start)
        indices = rng.integers(0, len(array), size=(size, len(array)))
        means[start : start + size] = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_rows_with_ci(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    summaries = aggregate_rows(rows)
    for summary_index, summary in enumerate(summaries):
        group = [
            row
            for row in rows
            if row["arm_id"] == summary["arm_id"]
            and row["threshold_policy"] == summary["threshold_policy"]
        ]
        for metric_index, metric in enumerate(METRICS):
            low, high = bootstrap_mean_ci(
                [float(row[metric]) for row in group],
                bootstrap_samples,
                seed + summary_index * 100 + metric_index,
            )
            summary[f"{metric}_ci95_low"] = low
            summary[f"{metric}_ci95_high"] = high
    return summaries


def external_sam_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    return paired_contrast(
        rows,
        contrast_id="DRIFT_SAM_BPM_vs_DRIFT",
        candidate="DRIFT_SAM_BPM",
        control="DRIFT",
        threshold_policy="selected",
        family="secondary_external",
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=seed,
    )
