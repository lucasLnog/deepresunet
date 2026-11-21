#!/usr/bin/env python3
"""Verifica hashes e isolamento dos splits sem abrir imagens ou mascaras."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.build_canonical_manifests import sha256_file
    from scripts.create_frozen_splits import SPLIT_NAMES, append_csv, read_csv
except ModuleNotFoundError:  # Execucao direta
    from build_canonical_manifests import sha256_file
    from create_frozen_splits import SPLIT_NAMES, append_csv, read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("manifests/splits"))
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Nao registrar a verificacao (reservado a testes automatizados).",
    )
    return parser.parse_args()


def frozen_hash(freeze_path: Path) -> str:
    text = freeze_path.read_text(encoding="utf-8")
    match = re.search(r"\*\*SHA-256:\*\* `([0-9a-f]{64})`", text)
    if not match:
        raise ValueError(f"Hash nao encontrado em {freeze_path}")
    return match.group(1)


def verify(split_dir: Path) -> dict[str, object]:
    summary = json.loads((split_dir / "split_summary.json").read_text(encoding="utf-8"))
    rows_by_split = {name: read_csv(split_dir / f"dfuc_{name}.csv") for name in SPLIT_NAMES}
    observed_hashes = {name: sha256_file(split_dir / f"dfuc_{name}.csv") for name in SPLIT_NAMES}

    for name in SPLIT_NAMES:
        expected_count = int(summary["target_counts"][name])
        if len(rows_by_split[name]) != expected_count:
            raise AssertionError(
                f"Contagem de {name}: {len(rows_by_split[name])}; esperada: {expected_count}"
            )
        if observed_hashes[name] != summary["manifest_hashes"][name]:
            raise AssertionError(f"Hash de {name} difere do resumo congelado")
        if any(row["split"] != name for row in rows_by_split[name]):
            raise AssertionError(f"Rotulo de split incorreto no manifesto {name}")

    freeze_path = split_dir / "TEST_FROZEN.md"
    if freeze_path.exists() and observed_hashes["test"] != frozen_hash(freeze_path):
        raise AssertionError("Hash do teste difere do registro TEST_FROZEN.md")

    overlap: dict[str, int] = {}
    for field in ("canonical_id", "image_group_sha256"):
        sets = {name: {row[field] for row in rows_by_split[name]} for name in SPLIT_NAMES}
        for index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[index + 1 :]:
                key = f"{field}:{left}:{right}"
                overlap[key] = len(sets[left] & sets[right])
                if overlap[key]:
                    raise AssertionError(f"Sobreposicao detectada em {key}")

    return {
        "status": "ok",
        "counts": {name: len(rows_by_split[name]) for name in SPLIT_NAMES},
        "hashes": observed_hashes,
        "overlap_counts": overlap,
        "images_or_masks_opened": False,
    }


def main() -> int:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    result = verify(split_dir)
    if not args.no_log:
        append_csv(
            split_dir / "TEST_ACCESS_LOG.csv",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "manifest_integrity_check",
                "purpose": "verify_frozen_hashes_and_split_isolation",
                "authorized_test_evaluation": False,
                "notes": "Metadata only; no image, mask, inference, or metric accessed",
            },
            ("timestamp", "action", "purpose", "authorized_test_evaluation", "notes"),
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
