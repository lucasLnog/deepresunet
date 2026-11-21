#!/usr/bin/env python3
"""Audita e resume os resultados pareados de Boundary IoU da avaliação de Boundary IoU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


SEEDS = (42, 1337, 2026)
ARMS = ("DRIFT", "DRIFTB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--day14-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (seed, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[seed] = min(1.0, running)
    return adjusted


def main() -> int:
    args = parse_args()
    source = args.input_root.resolve()
    prior = args.day14_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prior_rows = {int(row["seed"]): row for row in read_csv(prior)}
    if set(prior_rows) != set(SEEDS):
        raise ValueError("Resultados anteriores não contêm as três seeds esperadas")

    indexed: dict[tuple[int, str], dict[str, dict[str, float]]] = {}
    order: list[str] | None = None
    integrity: dict[str, dict] = {}
    per_seed: list[dict] = []
    raw_p: dict[int, float] = {}

    comparison = read_json(source / "comparison/boundary_comparison_summary.json")
    if comparison.get("partition") != "validation":
        raise ValueError("Partição inesperada no resumo comparativo")
    if comparison.get("protected_partitions_accessed") is not False:
        raise ValueError("Resumo indica acesso a partição protegida")
    if not math.isclose(float(comparison["boundary_dilation_ratio"]), 0.02):
        raise ValueError("Faixa de Boundary IoU diferente de 2%")

    comparison_by_seed = {int(row["seed"]): row for row in comparison["per_seed"]}
    for seed in SEEDS:
        seed_integrity: dict[str, object] = {}
        arm_rows: dict[str, list[dict[str, str]]] = {}
        for arm in ARMS:
            csv_path = source / f"seed{seed}" / arm / "validation_boundary_per_image.csv"
            json_path = source / f"seed{seed}" / arm / "boundary_summary.json"
            rows = read_csv(csv_path)
            summary = read_json(json_path)
            ids = [row["canonical_id"] for row in rows]
            if len(rows) != 294 or len(ids) != len(set(ids)):
                raise ValueError(f"seed {seed}/{arm}: quantidade ou IDs inválidos")
            if order is None:
                order = ids
            elif ids != order:
                raise ValueError(f"seed {seed}/{arm}: IDs ou ordem diferentes")
            if summary.get("status") != "completed" or summary.get("partition") != "validation":
                raise ValueError(f"seed {seed}/{arm}: execução incompleta ou partição incorreta")
            if summary.get("dice_reproduction_passed") is not True:
                raise ValueError(f"seed {seed}/{arm}: Dice não reproduzido")
            if summary.get("protected_partitions_accessed") is not False:
                raise ValueError(f"seed {seed}/{arm}: partição protegida acessada")
            if int(summary["records"]) != 294 or int(summary["seed"]) != seed:
                raise ValueError(f"seed {seed}/{arm}: metadados inconsistentes")
            expected_experiment = "DR-IFT" if arm == "DRIFT" else "DR-IFT-B"
            if summary["experiment_id"] != expected_experiment:
                raise ValueError(f"seed {seed}/{arm}: experimento incorreto")
            boundary_mean = statistics.fmean(float(row["boundary_iou"]) for row in rows)
            dice_mean = statistics.fmean(float(row["dice"]) for row in rows)
            if not math.isclose(boundary_mean, float(summary["boundary_iou_mean"]), abs_tol=1e-12):
                raise ValueError(f"seed {seed}/{arm}: média Boundary IoU inconsistente")
            prior_key = "DR-IFT_dice" if arm == "DRIFT" else "DR-IFT-B_dice"
            if not math.isclose(dice_mean, float(prior_rows[seed][prior_key]), abs_tol=1e-10):
                raise ValueError(f"seed {seed}/{arm}: Dice difere da ablação de perímetro")
            threshold_key = "DR-IFT_threshold" if arm == "DRIFT" else "DR-IFT-B_threshold"
            if not math.isclose(
                float(summary["threshold"]), float(prior_rows[seed][threshold_key])
            ):
                raise ValueError(
                    f"seed {seed}/{arm}: limiar difere do registrado na ablação de perímetro"
                )
            arm_rows[arm] = rows
            indexed[(seed, arm)] = {
                row["canonical_id"]: {
                    "dice": float(row["dice"]),
                    "boundary_iou": float(row["boundary_iou"]),
                }
                for row in rows
            }
            seed_integrity[arm] = {
                "run_id": summary["run_id"],
                "checkpoint_epoch": int(summary["checkpoint_epoch"]),
                "threshold": float(summary["threshold"]),
                "records": len(rows),
                "dice_reproduction_passed": True,
                "csv_sha256": sha256(csv_path),
            }

        differences = [
            float(candidate["boundary_iou"]) - float(baseline["boundary_iou"])
            for baseline, candidate in zip(arm_rows["DRIFT"], arm_rows["DRIFTB"])
        ]
        provided = comparison_by_seed[seed]
        recalculated = statistics.fmean(differences)
        if not math.isclose(recalculated, float(provided["difference"]), abs_tol=1e-12):
            raise ValueError(f"seed {seed}: diferença média inconsistente")
        raw_p[seed] = float(provided["permutation_p_two_sided"])
        per_seed.append(
            {
                "seed": seed,
                "baseline_boundary_iou": float(provided["DR-IFT_boundary_iou"]),
                "candidate_boundary_iou": float(provided["DR-IFT-B_boundary_iou"]),
                "difference": recalculated,
                "ci95_low": float(provided["ci95_low"]),
                "ci95_high": float(provided["ci95_high"]),
                "permutation_p_raw": raw_p[seed],
                "improved_images": sum(value > 0 for value in differences),
                "equal_images": sum(value == 0 for value in differences),
                "worsened_images": sum(value < 0 for value in differences),
            }
        )
        integrity[str(seed)] = seed_integrity

    adjusted = holm_adjust(raw_p)
    for row in per_seed:
        row["permutation_p_holm_exploratory"] = adjusted[int(row["seed"])]

    assert order is not None
    consensus = {"improved_all_three": 0, "worsened_all_three": 0, "mixed_or_equal": 0}
    majority = {"improved_at_least_two": 0, "worsened_at_least_two": 0, "other": 0}
    averaged_differences: list[float] = []
    for canonical_id in order:
        values = [
            indexed[(seed, "DRIFTB")][canonical_id]["boundary_iou"]
            - indexed[(seed, "DRIFT")][canonical_id]["boundary_iou"]
            for seed in SEEDS
        ]
        averaged_differences.append(statistics.fmean(values))
        positives = sum(value > 0 for value in values)
        negatives = sum(value < 0 for value in values)
        if positives == 3:
            consensus["improved_all_three"] += 1
        elif negatives == 3:
            consensus["worsened_all_three"] += 1
        else:
            consensus["mixed_or_equal"] += 1
        if positives >= 2:
            majority["improved_at_least_two"] += 1
        elif negatives >= 2:
            majority["worsened_at_least_two"] += 1
        else:
            majority["other"] += 1

    baseline_mean = statistics.fmean(row["baseline_boundary_iou"] for row in per_seed)
    candidate_mean = statistics.fmean(row["candidate_boundary_iou"] for row in per_seed)
    mean_difference = statistics.fmean(row["difference"] for row in per_seed)
    result = {
        "status": "audited",
        "source_zip_sha256": sha256(args.source_zip.resolve()) if args.source_zip else None,
        "partition": "validation",
        "records_per_run": 294,
        "seeds": list(SEEDS),
        "boundary_dilation_ratio": 0.02,
        "integrity": integrity,
        "per_seed": per_seed,
        "across_seed_descriptive": {
            "baseline_mean_boundary_iou": baseline_mean,
            "candidate_mean_boundary_iou": candidate_mean,
            "mean_absolute_difference": mean_difference,
            "sample_sd_seed_differences": statistics.stdev(row["difference"] for row in per_seed),
            "relative_difference_percent": 100.0 * mean_difference / baseline_mean,
            "seeds_favoring_candidate": sum(row["difference"] > 0 for row in per_seed),
        },
        "image_consistency_across_seeds": {
            **consensus,
            **majority,
            "mean_of_seed_averaged_image_differences": statistics.fmean(averaged_differences),
            "median_of_seed_averaged_image_differences": statistics.median(averaged_differences),
        },
        "interpretation": {
            "boundary_signal": "small_positive_consistent_across_seed_means",
            "seed_specific_uncertainty": "only_seed1337_ci_excludes_zero",
            "holm_note": "exploratory_only; no seed p-value remains below 0.05",
            "selection_role": "secondary_descriptive_only",
            "recommended_primary_model": "DR-IFT",
            "recommended_status_candidate": "DR-IFT-B remains exploratory",
        },
        "protected_partitions_accessed": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "per_seed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    print(json.dumps(result["across_seed_descriptive"], ensure_ascii=False, indent=2))
    print(json.dumps(result["image_consistency_across_seeds"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
