#!/usr/bin/env python3
"""Seleciona casos fixos de validação sem consultar resultados de modelos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("configs/visualization_cases_v1.csv"))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    manifest = read_rows(root / "manifests/splits/dfuc_validation.csv")
    inventory = read_rows(root / "results/audit/2026-08-25/dataset_inventory.csv")
    luminance_by_stem = {
        row["stem"]: float(row["image_mean_luminance"])
        for row in inventory
        if row["dataset"] == "DFUC2022" and row["status"] == "ok"
    }
    candidates = []
    for row in manifest:
        first_stem = row["source_stems"].split("|")[0]
        candidates.append(
            {
                "canonical_id": row["canonical_id"],
                "image_path": row["image_path"],
                "mask_paths": row["mask_paths"],
                "foreground_area_bin": int(row["foreground_area_bin"]),
                "foreground_fraction": float(row["foreground_fraction"]),
                "component_group": row["component_group"],
                "connected_components_8": int(row["connected_components_8"]),
                "mean_luminance": luminance_by_stem[first_stem],
            }
        )

    selected: list[dict] = []
    selected_ids: set[str] = set()
    for area_bin in range(5):
        group = sorted(
            (row for row in candidates if row["foreground_area_bin"] == area_bin),
            key=lambda row: (row["mean_luminance"], row["canonical_id"]),
        )
        for role, row in (("dark", group[0]), ("bright", group[-1])):
            chosen = {**row, "selection_role": f"area_{area_bin}_{role}"}
            selected.append(chosen)
            selected_ids.add(row["canonical_id"])

    multiple = sorted(
        (
            row
            for row in candidates
            if row["component_group"] == "multiple" and row["canonical_id"] not in selected_ids
        ),
        key=lambda row: (
            -row["connected_components_8"],
            row["foreground_fraction"],
            row["canonical_id"],
        ),
    )
    for index, row in enumerate(multiple[:2], start=1):
        selected.append({**row, "selection_role": f"multiple_component_{index}"})

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "panel_order",
        "selection_role",
        "canonical_id",
        "image_path",
        "mask_paths",
        "foreground_area_bin",
        "foreground_fraction",
        "component_group",
        "connected_components_8",
        "mean_luminance",
    )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for order, row in enumerate(selected, start=1):
            writer.writerow({"panel_order": order, **row})
    print(output)
    print(f"Casos fixados sem métricas de modelo: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
