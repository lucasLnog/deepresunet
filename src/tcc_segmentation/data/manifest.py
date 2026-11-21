"""Leitura estrita dos manifestos permitidos durante desenvolvimento."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
PROTECTED_NAME_MARKERS = ("test", "external", "fuseg")
REQUIRED_FIELDS = frozenset(
    {
        "dataset",
        "canonical_id",
        "image_path",
        "mask_paths",
        "mask_policy",
        "split",
    }
)


class ProtectedPartitionError(RuntimeError):
    """Uso de teste ou avaliacao externa em um fluxo de desenvolvimento."""


class ManifestIntegrityError(RuntimeError):
    """Manifesto ausente, alterado ou nao aprovado pelo split congelado."""


def _reject_protected_path(path: Path) -> None:
    lowered = path.name.casefold()
    if any(marker in lowered for marker in PROTECTED_NAME_MARKERS):
        raise ProtectedPartitionError(
            f"Manifesto protegido bloqueado durante desenvolvimento: {path.name}. "
            "Use somente dfuc_train.csv ou dfuc_validation.csv."
        )


def _validate_columns(fieldnames: Iterable[str] | None) -> None:
    observed = set(fieldnames or [])
    missing = REQUIRED_FIELDS - observed
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {sorted(missing)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_approved_hash(path: Path, split: str) -> None:
    summary_path = path.parent / "split_summary.json"
    if not summary_path.is_file():
        raise ManifestIntegrityError(
            f"Resumo de hashes nao encontrado ao lado do manifesto: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    try:
        expected_hash = summary["manifest_hashes"][split]
    except KeyError as error:
        raise ManifestIntegrityError(f"Hash aprovado ausente para split={split}") from error
    observed_hash = _sha256_file(path)
    if observed_hash != expected_hash:
        raise ManifestIntegrityError(
            f"Manifesto {split} alterado ou nao aprovado. "
            f"Esperado={expected_hash}; observado={observed_hash}"
        )


def load_development_manifest(
    manifest_path: str | Path, expected_split: str | None = None
) -> list[dict[str, str]]:
    """Carrega exclusivamente treino/validacao e rejeita manifestos mistos."""
    path = Path(manifest_path).expanduser().resolve()
    _reject_protected_path(path)
    if expected_split is not None and expected_split not in DEVELOPMENT_SPLITS:
        raise ProtectedPartitionError(f"Split nao permitido no desenvolvimento: {expected_split}")
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _validate_columns(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifesto vazio: {path}")

    observed_splits = {row["split"].strip().casefold() for row in rows}
    forbidden = observed_splits - DEVELOPMENT_SPLITS
    if forbidden:
        raise ProtectedPartitionError(
            f"Manifesto contem particao protegida ou desconhecida: {sorted(forbidden)}"
        )
    if len(observed_splits) != 1:
        raise ValueError(
            f"Manifesto de desenvolvimento deve conter um unico split: {observed_splits}"
        )
    observed_split = next(iter(observed_splits))
    if expected_split is not None and observed_split != expected_split:
        raise ValueError(f"Split observado={observed_split}; esperado={expected_split}")
    _validate_approved_hash(path, observed_split)
    if {row["dataset"] for row in rows} != {"DFUC2022"}:
        raise ProtectedPartitionError("Somente DFUC2022 e permitido no desenvolvimento")
    return rows


def resolve_dataset_path(project_root: str | Path, relative_path: str) -> Path:
    """Resolve um caminho do manifesto sem permitir saida da raiz do projeto."""
    root = Path(project_root).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Caminho fora da raiz do projeto: {relative_path}") from error
    return candidate
