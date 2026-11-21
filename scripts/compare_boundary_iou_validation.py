#!/usr/bin/env python3
"""Compara Boundary IoU pareado de DR-IFT-B menos DR-IFT nas três seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 294:
        raise ValueError(f"{path}: esperadas 294 imagens")
    ids = [row["canonical_id"] for row in rows]
    return ids, {row["canonical_id"]: float(row["boundary_iou"]) for row in rows}


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from tcc_segmentation.sam_validation import (
        paired_bootstrap_ci,
        paired_sign_flip_pvalue,
    )

    input_root = args.input_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in (42, 1337, 2026):
        base_ids, baseline = load(
            input_root / f"seed{seed}/DRIFT/validation_boundary_per_image.csv"
        )
        cand_ids, candidate = load(
            input_root / f"seed{seed}/DRIFTB/validation_boundary_per_image.csv"
        )
        if base_ids != cand_ids:
            raise ValueError(f"seed {seed}: IDs ou ordem diferentes")
        differences = [candidate[item] - baseline[item] for item in base_ids]
        low, high = paired_bootstrap_ci(differences, 10_000, 20260917 + seed)
        per_seed.append(
            {
                "seed": seed,
                "DR-IFT_boundary_iou": statistics.fmean(baseline.values()),
                "DR-IFT-B_boundary_iou": statistics.fmean(candidate.values()),
                "difference": statistics.fmean(differences),
                "ci95_low": low,
                "ci95_high": high,
                "permutation_p_two_sided": paired_sign_flip_pvalue(
                    differences, 50_000, 20261917 + seed
                ),
                "improved_images": sum(value > 0 for value in differences),
                "equal_images": sum(value == 0 for value in differences),
                "worsened_images": sum(value < 0 for value in differences),
            }
        )
    with (output / "boundary_per_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    differences = [float(row["difference"]) for row in per_seed]
    summary = {
        "status": "completed",
        "partition": "validation",
        "comparison": "DR-IFT-B_minus_DR-IFT",
        "metric": "Boundary IoU",
        "boundary_dilation_ratio": 0.02,
        "seeds": [42, 1337, 2026],
        "mean_difference_across_seeds": statistics.fmean(differences),
        "sample_sd_difference_across_seeds": statistics.stdev(differences),
        "seeds_favoring_DR-IFT-B": sum(value > 0 for value in differences),
        "per_seed": per_seed,
        "role": "secondary_descriptive; no model or threshold selection",
        "protected_partitions_accessed": False,
    }
    (output / "boundary_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
