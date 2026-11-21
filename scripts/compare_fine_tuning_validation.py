"""Compara DR-IFT, DR-IF e, opcionalmente, DR-IT na validação congelada.

O contraste primário é DR-IFT menos DR-IF. DR-IFT menos DR-IT é secundário,
pois verifica se o fine-tuning progressivo supera o treinamento integral iniciado
com pesos ImageNet. O script opera apenas sobre CSVs por imagem da validação.
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


METRICS = ("dice", "iou", "precision", "recall", "balanced_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drif", type=Path, required=True)
    parser.add_argument("--drift", type=Path, required=True)
    parser.add_argument("--drit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> tuple[list[str], dict[str, dict[str, float]], float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 294:
        raise ValueError(f"{path}: esperado 294 registros, obtido {len(raw)}")
    ids = [row["canonical_id"] for row in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: IDs duplicados")
    thresholds = {float(row["threshold"]) for row in raw}
    if len(thresholds) != 1:
        raise ValueError(f"{path}: esperado um único threshold")
    rows = {row["canonical_id"]: {metric: float(row[metric]) for metric in METRICS} for row in raw}
    return ids, rows, thresholds.pop()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def infer(
    differences: list[float], seed: int, bootstrap_samples: int, permutation_samples: int
) -> dict[str, float | int]:
    rng = random.Random(seed)
    n = len(differences)
    bootstrap_means = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(bootstrap_samples)
    ]
    observed = abs(statistics.fmean(differences))
    extreme = 0
    for _ in range(permutation_samples):
        permuted = abs(sum(value if rng.getrandbits(1) else -value for value in differences) / n)
        extreme += permuted >= observed - 1e-15
    return {
        "mean_difference": statistics.fmean(differences),
        "median_difference": statistics.median(differences),
        "ci95_low": percentile(bootstrap_means, 0.025),
        "ci95_high": percentile(bootstrap_means, 0.975),
        "improved": sum(value > 0 for value in differences),
        "equal": sum(value == 0 for value in differences),
        "worsened": sum(value < 0 for value in differences),
        "permutation_p_two_sided": (extreme + 1) / (permutation_samples + 1),
    }


def main() -> int:
    args = parse_args()
    paths = {"DR-IF": args.drif.resolve(), "DR-IFT": args.drift.resolve()}
    if args.drit:
        paths["DR-IT"] = args.drit.resolve()
    loaded = {arm: load(path) for arm, path in paths.items()}
    reference_ids = loaded["DR-IF"][0]
    for arm, (ids, _, _) in loaded.items():
        if ids != reference_ids:
            raise ValueError(f"ordem/identidade das imagens difere em {arm}")

    comparisons = [("DR-IFT", "DR-IF", "primary")]
    if "DR-IT" in loaded:
        comparisons.append(("DR-IFT", "DR-IT", "secondary"))

    arms = {}
    for arm, (_, rows, threshold) in loaded.items():
        arms[arm] = {
            "threshold": threshold,
            **{
                metric: statistics.fmean(rows[item][metric] for item in reference_ids)
                for metric in METRICS
            },
            "zero_dice_images": sum(rows[item]["dice"] == 0 for item in reference_ids),
        }

    results = {}
    for comparison_index, (left, right, role) in enumerate(comparisons):
        key = f"{left}_minus_{right}"
        results[key] = {
            "role": role,
            "metrics": {
                metric: infer(
                    [
                        loaded[left][1][item][metric] - loaded[right][1][item][metric]
                        for item in reference_ids
                    ],
                    args.seed + comparison_index * 100 + metric_index,
                    args.bootstrap_samples,
                    args.permutation_samples,
                )
                for metric_index, metric in enumerate(METRICS)
            },
        }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "completed",
        "partition": "validation",
        "statistical_unit": "canonical_image",
        "records": len(reference_ids),
        "primary_contrast": "DR-IFT_minus_DR-IF",
        "secondary_contrast": "DR-IFT_minus_DR-IT" if "DR-IT" in loaded else None,
        "primary_metric": "dice",
        "bootstrap": {
            "method": "paired_percentile",
            "samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "permutation": {
            "method": "paired_sign_flip_two_sided",
            "samples": args.permutation_samples,
        },
        "multiplicity": "single_prespecified_primary_contrast; secondary_is_exploratory",
        "arms": arms,
        "results": results,
        "sources": {
            arm: {"path": str(path), "sha256": sha256(path)} for arm, path in paths.items()
        },
        "protected_partitions_accessed": False,
    }
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (output_dir / "paired_differences_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["canonical_id"] + [
            f"{metric}_{left}_minus_{right}" for left, right, _ in comparisons for metric in METRICS
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in reference_ids:
            row: dict[str, str | float] = {"canonical_id": item}
            for left, right, _ in comparisons:
                for metric in METRICS:
                    row[f"{metric}_{left}_minus_{right}"] = (
                        loaded[left][1][item][metric] - loaded[right][1][item][metric]
                    )
            writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
