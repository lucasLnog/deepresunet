"""Compara A0, A1 e A2 por imagem na validação congelada.

Calcula diferenças pareadas, IC 95% bootstrap e teste bilateral de randomização por
inversão de sinais. O ajuste de Holm abrange os dois contrastes Dice pré-registrados
contra A0. Não acessa imagens, treino, teste ou FUSeg.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


METRICS = ("dice", "iou", "precision", "recall")
ARMS = ("A0", "A1", "A2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.lower()}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--analysis-role", choices=("primary", "sensitivity"), required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_rows(path: Path) -> tuple[list[str], dict[str, dict[str, float]], float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 294:
        raise ValueError(f"{path}: esperado 294 linhas, encontrado {len(raw)}")
    ids = [row["canonical_id"] for row in raw]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: IDs duplicados")
    thresholds = {float(row["threshold"]) for row in raw}
    if len(thresholds) != 1:
        raise ValueError(f"{path}: mais de um threshold")
    rows = {row["canonical_id"]: {metric: float(row[metric]) for metric in METRICS} for row in raw}
    return ids, rows, thresholds.pop()


def percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def bootstrap_ci(differences: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def permutation_pvalue(differences: list[float], samples: int, seed: int) -> float:
    rng = random.Random(seed)
    observed = abs(statistics.fmean(differences))
    extreme = 0
    for _ in range(samples):
        permuted = abs(
            sum(value if rng.getrandbits(1) else -value for value in differences) / len(differences)
        )
        if permuted >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def main() -> int:
    args = parse_args()
    paths = {arm: getattr(args, arm.lower()).resolve() for arm in ARMS}
    loaded = {arm: load_rows(path) for arm, path in paths.items()}
    reference_ids = loaded["A0"][0]
    for arm in ("A1", "A2"):
        if loaded[arm][0] != reference_ids:
            raise ValueError(f"A ordem/identidade de {arm} difere de A0")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_summary = {}
    for arm in ARMS:
        _, rows, threshold = loaded[arm]
        arm_summary[arm] = {
            "threshold": threshold,
            **{
                metric: statistics.fmean(rows[item][metric] for item in reference_ids)
                for metric in METRICS
            },
            "zero_dice": sum(rows[item]["dice"] == 0.0 for item in reference_ids),
        }

    comparisons = (("A1", "A0"), ("A2", "A0"), ("A1", "A2"))
    pairwise_rows: list[dict[str, object]] = []
    primary_raw_p: dict[str, float] = {}
    for comparison_index, (left, right) in enumerate(comparisons):
        for metric_index, metric in enumerate(METRICS):
            differences = [
                loaded[left][1][item][metric] - loaded[right][1][item][metric]
                for item in reference_ids
            ]
            low, high = bootstrap_ci(
                differences,
                args.bootstrap_samples,
                args.seed + comparison_index * 100 + metric_index,
            )
            row: dict[str, object] = {
                "comparison": f"{left}-{right}",
                "metric": metric,
                "mean_difference": statistics.fmean(differences),
                "median_difference": statistics.median(differences),
                "ci95_low": low,
                "ci95_high": high,
                "improved": sum(value > 0 for value in differences),
                "equal": sum(value == 0 for value in differences),
                "worsened": sum(value < 0 for value in differences),
                "permutation_p_raw": None,
                "permutation_p_holm": None,
            }
            if right == "A0" and metric == "dice":
                name = f"{left}-{right}"
                p_value = permutation_pvalue(
                    differences,
                    args.permutation_samples,
                    args.seed + 10_000 + comparison_index,
                )
                row["permutation_p_raw"] = p_value
                primary_raw_p[name] = p_value
            pairwise_rows.append(row)

    adjusted = holm_adjust(primary_raw_p)
    for row in pairwise_rows:
        if row["metric"] == "dice" and row["comparison"] in adjusted:
            row["permutation_p_holm"] = adjusted[str(row["comparison"])]

    with (output_dir / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)

    summary = {
        "status": "completed",
        "partition": "validation",
        "statistical_unit": "canonical_image",
        "records": len(reference_ids),
        "bootstrap": {
            "method": "paired_percentile",
            "samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "permutation_test": {
            "method": "paired_sign_flip_two_sided",
            "samples": args.permutation_samples,
        },
        "analysis_role": args.analysis_role,
        "multiplicity": "Holm for A1-A0 and A2-A0 Dice contrasts",
        "seed": args.seed,
        "arms": arm_summary,
        "sources": {arm: {"file": str(paths[arm]), "sha256": sha256(paths[arm])} for arm in ARMS},
        "protected_partitions_accessed": False,
    }
    (output_dir / "paired_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
