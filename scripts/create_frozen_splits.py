#!/usr/bin/env python3
"""Gera splits estratificados e congela o manifesto de teste DFUC2022."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from scripts.build_canonical_manifests import load_simple_yaml, sha256_file
except ModuleNotFoundError:  # Execucao direta
    from build_canonical_manifests import load_simple_yaml, sha256_file


SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=Path("configs/split_policy.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Raiz do projeto.")
    parser.add_argument("--output-dir", type=Path, default=Path("manifests/splits"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fields: Sequence[str]) -> None:
    """Acrescenta um evento sem apagar o historico existente."""
    write_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def largest_remainder_targets(total: int, ratios: Sequence[float]) -> list[int]:
    quotas = [total * ratio for ratio in ratios]
    targets = [math.floor(quota) for quota in quotas]
    remaining = total - sum(targets)
    order = sorted(
        range(len(ratios)), key=lambda index: (quotas[index] - targets[index], -index), reverse=True
    )
    for index in order[:remaining]:
        targets[index] += 1
    return targets


def quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    if bins < 2:
        raise ValueError("foreground_quantile_bins deve ser >= 2")
    return [
        float(value)
        for value in np.quantile(
            np.asarray(values, dtype=np.float64), np.linspace(0.0, 1.0, bins + 1)
        )
    ]


def add_strata(
    rows: Sequence[dict[str, str]], bins: int
) -> tuple[list[dict[str, Any]], list[float]]:
    fractions = [float(row["foreground_fraction"]) for row in rows]
    edges = quantile_edges(fractions, bins)
    internal_edges = np.asarray(edges[1:-1], dtype=np.float64)
    enriched: list[dict[str, Any]] = []

    for row in rows:
        fraction = float(row["foreground_fraction"])
        area_bin = int(np.searchsorted(internal_edges, fraction, side="right"))
        component_group = "multiple" if float(row["connected_components_8"]) > 1.0 else "single"
        enriched.append(
            {
                **row,
                "foreground_area_bin": area_bin,
                "component_group": component_group,
                "stratum": f"area_{area_bin}__{component_group}",
            }
        )
    return enriched, edges


def allocate_stratum_counts(
    stratum_sizes: dict[str, int], ratios: Sequence[float], global_targets: Sequence[int]
) -> dict[str, list[int]]:
    allocations: dict[str, list[int]] = {}
    fractions: dict[tuple[str, int], float] = {}
    row_deficits: dict[str, int] = {}

    for stratum, size in sorted(stratum_sizes.items()):
        quotas = [size * ratio for ratio in ratios]
        floors = [math.floor(quota) for quota in quotas]
        allocations[stratum] = floors
        row_deficits[stratum] = size - sum(floors)
        for split_index, quota in enumerate(quotas):
            fractions[(stratum, split_index)] = quota - floors[split_index]

    current_columns = [
        sum(counts[index] for counts in allocations.values()) for index in range(len(ratios))
    ]
    column_deficits = [
        int(global_targets[index] - current_columns[index]) for index in range(len(ratios))
    ]

    candidates = sorted(
        fractions,
        key=lambda item: (fractions[item], item[0], -item[1]),
        reverse=True,
    )
    used_cells: set[tuple[str, int]] = set()
    for stratum, split_index in candidates:
        if row_deficits[stratum] > 0 and column_deficits[split_index] > 0:
            allocations[stratum][split_index] += 1
            row_deficits[stratum] -= 1
            column_deficits[split_index] -= 1
            used_cells.add((stratum, split_index))

    for stratum in sorted(row_deficits):
        while row_deficits[stratum] > 0:
            available = [
                index
                for index, deficit in enumerate(column_deficits)
                if deficit > 0 and (stratum, index) not in used_cells
            ]
            if not available:
                raise AssertionError("Nao foi possivel fechar a matriz de alocacao")
            split_index = max(available, key=lambda index: column_deficits[index])
            allocations[stratum][split_index] += 1
            row_deficits[stratum] -= 1
            column_deficits[split_index] -= 1
            used_cells.add((stratum, split_index))

    if any(column_deficits):
        raise AssertionError(f"Deficits globais restantes: {column_deficits}")
    return allocations


def stable_rank(seed: int, canonical_id: str) -> str:
    return hashlib.sha256(f"{seed}|{canonical_id}".encode("utf-8")).hexdigest()


def assign_splits(
    rows: Sequence[dict[str, Any]], ratios: Sequence[float], seed: int
) -> tuple[list[dict[str, Any]], list[int]]:
    global_targets = largest_remainder_targets(len(rows), ratios)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["stratum"])].append(row)
    allocations = allocate_stratum_counts(
        {stratum: len(members) for stratum, members in grouped.items()},
        ratios,
        global_targets,
    )

    assigned: list[dict[str, Any]] = []
    for stratum, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: stable_rank(seed, str(row["canonical_id"])))
        cursor = 0
        for split_index, split_name in enumerate(SPLIT_NAMES):
            count = allocations[stratum][split_index]
            for row in ordered[cursor : cursor + count]:
                assigned.append(
                    {
                        **row,
                        "split": split_name,
                        "split_seed": seed,
                        "split_rank_sha256": stable_rank(seed, str(row["canonical_id"])),
                    }
                )
            cursor += count
        if cursor != len(ordered):
            raise AssertionError(f"Stratum incompleto: {stratum}")

    counts = Counter(row["split"] for row in assigned)
    observed = [counts[name] for name in SPLIT_NAMES]
    if observed != global_targets:
        raise AssertionError(f"Contagens {observed} diferem do alvo {global_targets}")
    return assigned, global_targets


def describe_split(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fractions = np.asarray([float(row["foreground_fraction"]) for row in rows], dtype=np.float64)
    return {
        "count": len(rows),
        "foreground_fraction": {
            "min": float(fractions.min()),
            "q1": float(np.quantile(fractions, 0.25)),
            "median": float(np.quantile(fractions, 0.50)),
            "q3": float(np.quantile(fractions, 0.75)),
            "max": float(fractions.max()),
            "mean": float(fractions.mean()),
        },
        "component_groups": dict(sorted(Counter(row["component_group"] for row in rows).items())),
        "area_bins": dict(
            sorted(
                Counter(str(row["foreground_area_bin"]) for row in rows).items(),
                key=lambda item: int(item[0]),
            )
        ),
    }


def validate_no_overlap(assigned: Sequence[dict[str, Any]]) -> None:
    ids_by_split = {
        name: {row["canonical_id"] for row in assigned if row["split"] == name}
        for name in SPLIT_NAMES
    }
    hashes_by_split = {
        name: {row["image_group_sha256"] for row in assigned if row["split"] == name}
        for name in SPLIT_NAMES
    }
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if ids_by_split[left] & ids_by_split[right]:
                raise AssertionError(f"canonical_id compartilhado entre {left} e {right}")
            if hashes_by_split[left] & hashes_by_split[right]:
                raise AssertionError(f"imagem compartilhada entre {left} e {right}")


def protect_existing_frozen_test(test_path: Path, expected_hash: str) -> bool:
    """Impede a substituicao silenciosa de um manifesto de teste congelado."""
    if not test_path.exists():
        return False
    observed_hash = sha256_file(test_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            "O teste congelado existente difere da divisao deterministica esperada. "
            f"Esperado={expected_hash}; observado={observed_hash}. "
            "A execucao foi interrompida para preservar a integridade experimental."
        )
    return True


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    policy_path = args.policy.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_simple_yaml(policy_path)

    source_path = (project_root / str(policy["source_manifest"])).resolve()
    source_rows = read_csv(source_path)
    ratios = [
        float(policy["ratios"]["train"]),
        float(policy["ratios"]["validation"]),
        float(policy["ratios"]["test"]),
    ]
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-12):
        raise ValueError(f"Ratios nao somam 1: {ratios}")
    seed = int(policy["seed"])
    bins = int(policy["stratification"]["foreground_quantile_bins"])

    enriched, edges = add_strata(source_rows, bins)
    assigned, targets = assign_splits(enriched, ratios, seed)
    validate_no_overlap(assigned)

    source_fields = list(source_rows[0].keys())
    extra_fields = [
        "foreground_area_bin",
        "component_group",
        "stratum",
        "split",
        "split_seed",
        "split_rank_sha256",
    ]
    fields = source_fields + extra_fields
    combined_path = output_dir / "dfuc_split_manifest.csv"
    write_csv(
        combined_path,
        sorted(assigned, key=lambda row: (row["split"], row["canonical_id"])),
        fields,
    )

    split_paths: dict[str, Path] = {}
    split_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for split_name in SPLIT_NAMES:
        path = output_dir / f"dfuc_{split_name}.csv"
        split_rows = sorted(
            (row for row in assigned if row["split"] == split_name),
            key=lambda row: row["canonical_id"],
        )
        split_paths[split_name] = path
        split_rows_by_name[split_name] = split_rows

    # Calcula o hash que a divisao deterministica produziria antes de tocar no
    # teste existente. O arquivo temporario fica dentro da pasta de saida.
    expected_test_path = output_dir / ".dfuc_test.expected.csv"
    write_csv(expected_test_path, split_rows_by_name["test"], fields)
    expected_test_hash = sha256_file(expected_test_path)
    test_was_already_frozen = protect_existing_frozen_test(split_paths["test"], expected_test_hash)
    expected_test_path.unlink()

    for split_name, path in split_paths.items():
        if split_name == "test" and test_was_already_frozen:
            continue
        write_csv(path, split_rows_by_name[split_name], fields)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_path.relative_to(project_root).as_posix()),
        "source_manifest_sha256": sha256_file(source_path),
        "policy_sha256": sha256_file(policy_path),
        "seed": seed,
        "ratios": dict(zip(SPLIT_NAMES, ratios)),
        "target_counts": dict(zip(SPLIT_NAMES, targets)),
        "foreground_quantile_edges": edges,
        "stratification": "foreground quintile x single/multiple components",
        "patient_level_split": False,
        "patient_level_limitation": "patient identifiers unavailable",
        "overlap_checks": {
            "canonical_id_overlap": 0,
            "exact_image_hash_overlap": 0,
        },
        "splits": {
            name: describe_split([row for row in assigned if row["split"] == name])
            for name in SPLIT_NAMES
        },
        "manifest_hashes": {
            "combined": sha256_file(combined_path),
            **{name: sha256_file(path) for name, path in split_paths.items()},
        },
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    test_hash = summary["manifest_hashes"]["test"]
    freeze_path = output_dir / "TEST_FROZEN.md"
    freeze_text = f"""# Teste DFUC2022 congelado

**Criado em:** {datetime.now().astimezone().isoformat(timespec='seconds')}  
**Manifesto:** `manifests/splits/dfuc_test.csv`  
**SHA-256:** `{test_hash}`  
**Amostras:** {targets[2]}  
**Seed do split:** {seed}

O teste nao pode ser usado para escolher arquitetura, backbone, pre-processamento,
augmentation, loss, learning rate, epoca, checkpoint, threshold ou configuracao do SAM.

Antes da avaliacao final, o hash deve ser recalculado e comparado ao valor acima.
Qualquer acesso deve ser registrado em `TEST_ACCESS_LOG.csv`.
"""
    if not freeze_path.exists():
        freeze_path.write_text(freeze_text, encoding="utf-8")
    access_log = output_dir / "TEST_ACCESS_LOG.csv"
    if not test_was_already_frozen:
        append_csv(
            access_log,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "manifest_created_and_frozen",
                "purpose": "split_construction_before_model_development",
                "authorized_test_evaluation": False,
                "notes": "No model inference or metric evaluation performed",
            },
            ("timestamp", "action", "purpose", "authorized_test_evaluation", "notes"),
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
