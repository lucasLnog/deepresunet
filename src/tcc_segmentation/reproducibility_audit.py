"""Read-only helpers for the final reproducibility audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def grouped_unique_counts(
    rows: Iterable[Mapping[str, str]], group_fields: tuple[str, ...], id_field: str
) -> dict[tuple[str, ...], int]:
    groups: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, set()).add(row[id_field])
    return {key: len(values) for key, values in groups.items()}


def hash_matches(path: Path, expected: str) -> bool:
    return path.exists() and sha256(path).lower() == expected.lower()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
