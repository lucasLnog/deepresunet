#!/usr/bin/env python3
"""Constroi unidades canonicas a partir do inventario auditado.

O programa nao copia nem modifica imagens ou mascaras. Para o DFUC2022, cada
imagem unica torna-se uma unidade e as mascaras duplicadas recebem a politica
`logical_or`. Para o FUSeg, referencias vazias listadas sao excluidas e as
variantes validas da mesma imagem permanecem associadas a uma unica unidade.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    from scripts.audit_datasets import count_components_8, mask_to_gray_array
except ModuleNotFoundError:  # Execucao direta: python scripts/build_canonical_manifests.py
    from audit_datasets import count_components_8, mask_to_gray_array


CANONICAL_FIELDS = (
    "dataset",
    "canonical_id",
    "image_group_sha256",
    "image_path",
    "source_count",
    "source_stems",
    "mask_paths",
    "mask_policy",
    "reference_count",
    "width",
    "height",
    "foreground_pixels",
    "total_pixels",
    "foreground_fraction",
    "connected_components_8",
    "reference_foreground_fractions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("results/audit/2026-08-25/dataset_inventory.csv"),
    )
    parser.add_argument("--policy", type=Path, default=Path("configs/data_policy.yaml"))
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=Path("configs/data_quality_exclusions.csv"),
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path("."), help="Raiz para paths relativos."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("manifests/canonical"),
    )
    return parser.parse_args()


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith(('"', "'")) and value.endswith(('"', "'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Le o subconjunto de YAML usado pela politica sem dependencia externa."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        content = line_without_comment.strip()
        if ":" not in content:
            raise ValueError(f"YAML nao suportado na linha {line_number}: {raw_line}")
        key, value = content.split(":", 1)

        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key.strip()] = parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))

    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def project_relative(path_value: str, project_root: Path) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"Path fora da raiz do projeto: {path}") from None


def load_binary_mask(path: str, threshold: int) -> np.ndarray:
    with Image.open(path) as source:
        source.load()
        gray, _ = mask_to_gray_array(source)
    return np.asarray(gray) >= threshold


def canonical_id(dataset: str, stems: Sequence[str]) -> str:
    return f"{dataset}_{min(stems)}"


def summarize_masks(
    absolute_mask_paths: Sequence[str], threshold: int, policy: str
) -> dict[str, Any]:
    masks = [load_binary_mask(path, threshold) for path in absolute_mask_paths]
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"Mascaras do grupo com shapes diferentes: {shapes}")

    fractions = [float(mask.mean()) for mask in masks]
    if policy == "logical_or":
        representative = np.logical_or.reduce(masks)
    elif policy == "reference_mean":
        representative = masks[0]
    else:
        raise ValueError(f"Politica de mascara desconhecida: {policy}")

    total_pixels = int(representative.size)
    if policy == "reference_mean":
        foreground_pixels: int | float = statistics.fmean(float(mask.sum()) for mask in masks)
        foreground_fraction = statistics.fmean(fractions)
        components = statistics.fmean(float(count_components_8(mask)) for mask in masks)
    else:
        foreground_pixels = int(representative.sum())
        foreground_fraction = float(representative.mean())
        components = int(count_components_8(representative))

    return {
        "foreground_pixels": foreground_pixels,
        "total_pixels": total_pixels,
        "foreground_fraction": foreground_fraction,
        "connected_components_8": components,
        "reference_foreground_fractions": "|".join(f"{value:.12f}" for value in fractions),
    }


def build_dataset_units(
    dataset: str,
    inventory: Sequence[dict[str, str]],
    excluded_stems: set[str],
    project_root: Path,
    threshold: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in inventory
        if row["dataset"] == dataset and row["status"] == "ok" and row["stem"] not in excluded_stems
    ]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eligible:
        groups[row["image_pixel_sha256"]].append(row)

    units: list[dict[str, Any]] = []
    for image_hash, members in sorted(
        groups.items(), key=lambda item: min(row["stem"] for row in item[1])
    ):
        members = sorted(members, key=lambda row: row["stem"])
        stems = [row["stem"] for row in members]
        absolute_masks = [row["mask_path"] for row in members]
        policy = "logical_or" if dataset == "DFUC2022" else "reference_mean"
        mask_summary = summarize_masks(absolute_masks, threshold, policy)
        width = int(members[0]["image_width"])
        height = int(members[0]["image_height"])

        units.append(
            {
                "dataset": dataset,
                "canonical_id": canonical_id(dataset, stems),
                "image_group_sha256": image_hash,
                "image_path": project_relative(members[0]["image_path"], project_root),
                "source_count": len(members),
                "source_stems": "|".join(stems),
                "mask_paths": "|".join(
                    project_relative(path, project_root) for path in absolute_masks
                ),
                "mask_policy": policy,
                "reference_count": len(absolute_masks),
                "width": width,
                "height": height,
                **mask_summary,
            }
        )

    return units


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_units(
    units: Sequence[dict[str, Any]],
    dataset: str,
    expected_units: int,
    expected_sources: int,
) -> None:
    if len(units) != expected_units:
        raise AssertionError(f"{dataset}: esperado {expected_units} unidades, obtido {len(units)}")
    if sum(int(row["source_count"]) for row in units) != expected_sources:
        raise AssertionError(f"{dataset}: contagem de fontes inesperada")
    ids = [str(row["canonical_id"]) for row in units]
    hashes = [str(row["image_group_sha256"]) for row in units]
    if len(set(ids)) != len(ids):
        raise AssertionError(f"{dataset}: canonical_id duplicado")
    if len(set(hashes)) != len(hashes):
        raise AssertionError(f"{dataset}: grupo de imagem duplicado")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    inventory_path = args.inventory.resolve()
    policy_path = args.policy.resolve()
    exclusions_path = args.exclusions.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = load_simple_yaml(policy_path)
    inventory = read_csv(inventory_path)
    exclusions = read_csv(exclusions_path)
    excluded_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in exclusions:
        excluded_by_dataset[row["dataset"]].add(row["stem"])

    threshold = int(policy["global"]["mask_binarization_threshold"])
    dfuc_units = build_dataset_units(
        "DFUC2022",
        inventory,
        excluded_by_dataset["DFUC2022"],
        project_root,
        threshold,
    )
    fuseg_units = build_dataset_units(
        "FUSeg",
        inventory,
        excluded_by_dataset["FUSeg"],
        project_root,
        threshold,
    )

    validate_units(
        dfuc_units,
        "DFUC2022",
        int(policy["DFUC2022"]["expected_canonical_units"]),
        int(policy["DFUC2022"]["raw_records"]),
    )
    validate_units(
        fuseg_units,
        "FUSeg",
        int(policy["FUSeg"]["expected_unique_evaluation_units"]),
        int(policy["FUSeg"]["valid_reference_records_after_exclusion"]),
    )

    dfuc_path = output_dir / "dfuc2022_canonical.csv"
    fuseg_path = output_dir / "fuseg_external_canonical.csv"
    write_csv(dfuc_path, dfuc_units)
    write_csv(fuseg_path, fuseg_units)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mask_threshold": threshold,
        "raw_data_modified": False,
        "DFUC2022": {
            "units": len(dfuc_units),
            "source_records": sum(int(row["source_count"]) for row in dfuc_units),
            "multi_reference_units": sum(int(row["reference_count"]) > 1 for row in dfuc_units),
            "mask_policy": "logical_or",
        },
        "FUSeg": {
            "units": len(fuseg_units),
            "valid_reference_records": sum(int(row["source_count"]) for row in fuseg_units),
            "excluded_records": len(excluded_by_dataset["FUSeg"]),
            "multi_reference_units": sum(int(row["reference_count"]) > 1 for row in fuseg_units),
            "mask_policy": "reference_mean",
        },
        "files": {
            "dfuc2022_canonical.csv": sha256_file(dfuc_path),
            "fuseg_external_canonical.csv": sha256_file(fuseg_path),
            "data_policy.yaml": sha256_file(policy_path),
            "data_quality_exclusions.csv": sha256_file(exclusions_path),
            "dataset_inventory.csv": sha256_file(inventory_path),
        },
    }
    (output_dir / "canonical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
