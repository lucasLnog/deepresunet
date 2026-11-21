"""Compara P1+A1 com D0, P1 e A1 na validação congelada.

O contraste confirmatório pré-registrado é P1+A1 menos A1 em Dice macro por
imagem. Comparações secundárias e o contraste de interação são descritivos. O
script não acessa imagens, checkpoints, teste interno ou FUSeg.
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
ARMS = ("D0", "P1", "A1", "P1A1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.lower()}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-role", choices=("primary", "sensitivity"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> tuple[list[str], dict[str, dict[str, float]], float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 294:
        raise ValueError(f"{path}: esperado 294 linhas, encontrado {len(raw)}")
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
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(differences: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = [sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
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


def describe(
    comparison: str,
    metric: str,
    differences: list[float],
    samples: int,
    seed: int,
    *,
    primary: bool = False,
    permutation_samples: int = 50_000,
) -> dict[str, object]:
    low, high = bootstrap_ci(differences, samples, seed)
    return {
        "comparison": comparison,
        "metric": metric,
        "mean_difference": statistics.fmean(differences),
        "median_difference": statistics.median(differences),
        "ci95_low": low,
        "ci95_high": high,
        "improved": sum(value > 0 for value in differences),
        "equal": sum(value == 0 for value in differences),
        "worsened": sum(value < 0 for value in differences),
        "permutation_p_two_sided": (
            permutation_pvalue(differences, permutation_samples, seed + 50_000) if primary else None
        ),
        "inferential_role": "primary" if primary else "descriptive",
    }


def main() -> int:
    args = parse_args()
    paths = {arm: getattr(args, arm.lower()).resolve() for arm in ARMS}
    loaded = {arm: load(path) for arm, path in paths.items()}
    ids = loaded["D0"][0]
    for arm in ARMS[1:]:
        if loaded[arm][0] != ids:
            raise ValueError(f"Ordem ou identidade de {arm} difere de D0")

    rows: list[dict[str, object]] = []
    contrasts = (("P1A1", "A1"), ("P1A1", "P1"), ("P1A1", "D0"))
    for contrast_index, (left, right) in enumerate(contrasts):
        for metric_index, metric in enumerate(METRICS):
            differences = [
                loaded[left][1][item][metric] - loaded[right][1][item][metric] for item in ids
            ]
            rows.append(
                describe(
                    f"{left}-{right}",
                    metric,
                    differences,
                    args.bootstrap_samples,
                    args.seed + contrast_index * 100 + metric_index,
                    primary=(left, right, metric) == ("P1A1", "A1", "dice"),
                    permutation_samples=args.permutation_samples,
                )
            )

    interaction = [
        (loaded["P1A1"][1][item]["dice"] - loaded["P1"][1][item]["dice"])
        - (loaded["A1"][1][item]["dice"] - loaded["D0"][1][item]["dice"])
        for item in ids
    ]
    rows.append(
        describe(
            "(P1A1-P1)-(A1-D0)",
            "dice",
            interaction,
            args.bootstrap_samples,
            args.seed + 900,
        )
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_comparisons.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arms = {}
    for arm in ARMS:
        _, data, threshold = loaded[arm]
        arms[arm] = {
            "threshold": threshold,
            **{metric: statistics.fmean(data[item][metric] for item in ids) for metric in METRICS},
            "zero_dice": sum(data[item]["dice"] == 0.0 for item in ids),
        }
    primary = next(
        row for row in rows if row["comparison"] == "P1A1-A1" and row["metric"] == "dice"
    )
    summary = {
        "status": "completed",
        "partition": "validation",
        "statistical_unit": "canonical_image",
        "records": len(ids),
        "analysis_role": args.analysis_role,
        "primary_contrast": "P1A1-A1",
        "primary_metric": "macro_dice_per_image",
        "bootstrap": {
            "method": "paired_percentile",
            "samples": args.bootstrap_samples,
            "confidence": 0.95,
        },
        "permutation": {
            "method": "paired_sign_flip_two_sided",
            "samples": args.permutation_samples,
        },
        "primary_result": primary,
        "arms": arms,
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
