#!/usr/bin/env python3
"""Audita e compara DR-IFT-B com DR-IFT nas três seeds de validação."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

import yaml
import numpy as np


METRICS = ("dice", "iou", "precision", "recall", "balanced_accuracy")
SEEDS = (42, 1337, 2026)
EXPECTED_LOSS = {
    "name": "bce_dice_plus_soft_perimeter",
    "bce_weight": 0.5,
    "dice_weight": 0.5,
    "region_weight": 0.9,
    "perimeter_weight": 0.1,
    "smooth": 1.0,
    "perimeter_epsilon": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=50_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def infer(differences: list[float], seed: int, bootstrap: int, permutations: int):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_values = []
    for start in range(0, bootstrap, 1000):
        size = min(1000, bootstrap - start)
        indices = rng.integers(0, n, size=(size, n))
        boot_values.extend(values[indices].mean(axis=1).tolist())
    observed = abs(float(values.mean()))
    extreme = 0
    for start in range(0, permutations, 1000):
        size = min(1000, permutations - start)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
    return {
        "mean_difference": statistics.fmean(differences),
        "median_difference": statistics.median(differences),
        "ci95_low": percentile(boot_values, 0.025),
        "ci95_high": percentile(boot_values, 0.975),
        "improved_images": sum(item > 0 for item in differences),
        "equal_images": sum(item == 0 for item in differences),
        "worsened_images": sum(item < 0 for item in differences),
        "permutation_p_two_sided": (extreme + 1) / (permutations + 1),
    }


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 294:
        raise ValueError(f"{path}: esperado 294 registros, obtido {len(raw)}")
    ids = [row["canonical_id"] for row in raw]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: IDs duplicados")
    thresholds = {float(row["threshold"]) for row in raw}
    if len(thresholds) != 1:
        raise ValueError(f"{path}: threshold inconsistente")
    indexed = {
        row["canonical_id"]: {
            **{metric: float(row[metric]) for metric in METRICS},
            "foreground_area_bin": int(row["foreground_area_bin"]),
        }
        for row in raw
    }
    return ids, indexed, thresholds.pop()


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def locate_candidate(root: Path, seed: int) -> Path:
    directories = sorted((root / f"seed{seed}" / "DRIFTB").glob(f"DRIFTB_seed{seed}_*"))
    if len(directories) != 1:
        raise ValueError(f"seed {seed}: esperada uma execução, encontradas {len(directories)}")
    return directories[0]


def baseline_directory(root: Path, seed: int) -> Path:
    if seed == 42:
        return root / "results/incoming/day10_20260903/DRIFT"
    return root / f"results/incoming/day11_20260914/seed{seed}/DRIFT"


def audit_config(candidate: dict[str, Any], baseline: dict[str, Any], seed: int):
    if candidate.get("loss") != EXPECTED_LOSS:
        raise ValueError(f"seed {seed}: loss fora do contrato")
    if int(candidate.get("seed", -1)) != seed:
        raise ValueError(f"seed {seed}: config registra outra seed")
    ignored = {
        "protocol_amendment",
        "stage",
        "experiment_id",
        "scientific_role",
        "loss",
        "boundary_ablation",
        "runtime",
    }
    differing = [
        key for key in baseline if key not in ignored and candidate.get(key) != baseline[key]
    ]
    if differing:
        raise ValueError(f"seed {seed}: fatores não autorizados diferem: {differing}")
    return {"only_loss_changed": True, "ignored_identity_fields": sorted(ignored)}


def analyze_pair(
    baseline_csv: Path,
    candidate_csv: Path,
    *,
    inference_seed: int,
    bootstrap: int,
    permutations: int,
):
    base_ids, baseline, base_threshold = load_rows(baseline_csv)
    candidate_ids, candidate, candidate_threshold = load_rows(candidate_csv)
    if base_ids != candidate_ids:
        raise ValueError("Identidade ou ordem das 294 imagens difere")
    metrics = {}
    for index, metric in enumerate(METRICS):
        differences = [candidate[item][metric] - baseline[item][metric] for item in base_ids]
        metrics[metric] = {
            "DR-IFT": statistics.fmean(baseline[item][metric] for item in base_ids),
            "DR-IFT-B": statistics.fmean(candidate[item][metric] for item in base_ids),
            **infer(
                differences,
                inference_seed + index,
                bootstrap,
                permutations,
            ),
        }
    area_bins = []
    for area_bin in sorted({candidate[item]["foreground_area_bin"] for item in base_ids}):
        selected = [item for item in base_ids if candidate[item]["foreground_area_bin"] == area_bin]
        area_bins.append(
            {
                "foreground_area_bin": area_bin,
                "images": len(selected),
                "DR-IFT_dice": statistics.fmean(baseline[item]["dice"] for item in selected),
                "DR-IFT-B_dice": statistics.fmean(candidate[item]["dice"] for item in selected),
                "difference_dice": statistics.fmean(
                    candidate[item]["dice"] - baseline[item]["dice"] for item in selected
                ),
            }
        )
    return {
        "baseline_threshold": base_threshold,
        "candidate_threshold": candidate_threshold,
        "metrics": metrics,
        "area_bins": area_bins,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    candidate_root = args.candidate_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    per_seed = []
    seed_details = {}
    integrity = {}
    all_area_rows = []
    for seed in SEEDS:
        candidate_dir = locate_candidate(candidate_root, seed)
        baseline_dir = baseline_directory(root, seed)
        required = (
            "config_used.yaml",
            "final_summary.json",
            "training_summary.json",
            "history.csv",
            "threshold_selection.json",
            "validation_per_image_selected.csv",
            "validation_per_image_at_0_5.csv",
            "augmentation_safety.json",
        )
        missing = [name for name in required if not (candidate_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"seed {seed}: ausentes {missing}")
        candidate_summary = json.loads(
            (candidate_dir / "final_summary.json").read_text(encoding="utf-8")
        )
        baseline_summary = json.loads(
            (baseline_dir / "final_summary.json").read_text(encoding="utf-8")
        )
        if candidate_summary.get("protected_partitions_accessed") is not False:
            raise ValueError(f"seed {seed}: partição protegida possivelmente acessada")
        if candidate_summary.get("status") != "completed":
            raise ValueError(f"seed {seed}: execução incompleta")
        config_audit = audit_config(
            load_yaml(candidate_dir / "config_used.yaml"),
            load_yaml(baseline_dir / "config_used.yaml"),
            seed,
        )
        safety = json.loads(
            (candidate_dir / "augmentation_safety.json").read_text(encoding="utf-8")
        )
        if int(safety["identity_fallbacks"]) != 0:
            raise ValueError(f"seed {seed}: augmentation usou fallback")

        selected = analyze_pair(
            baseline_dir / "validation_per_image_selected.csv",
            candidate_dir / "validation_per_image_selected.csv",
            inference_seed=20260917 + seed,
            bootstrap=args.bootstrap_samples,
            permutations=args.permutation_samples,
        )
        fixed = analyze_pair(
            baseline_dir / "validation_per_image_at_0_5.csv",
            candidate_dir / "validation_per_image_at_0_5.csv",
            inference_seed=20261917 + seed,
            bootstrap=args.bootstrap_samples,
            permutations=args.permutation_samples,
        )
        detail = {
            "selected_thresholds": selected,
            "fixed_threshold_0_5": fixed,
            "candidate_best_epoch": int(candidate_summary["training"]["best_epoch"]),
            "candidate_epochs_executed": int(candidate_summary["training"]["epochs_executed"]),
            "baseline_best_epoch": int(baseline_summary["training"]["best_epoch"]),
            "baseline_epochs_executed": int(baseline_summary["training"]["epochs_executed"]),
        }
        seed_details[str(seed)] = detail
        per_seed.append(
            {
                "seed": seed,
                "DR-IFT_threshold": selected["baseline_threshold"],
                "DR-IFT-B_threshold": selected["candidate_threshold"],
                "DR-IFT_dice": selected["metrics"]["dice"]["DR-IFT"],
                "DR-IFT-B_dice": selected["metrics"]["dice"]["DR-IFT-B"],
                "selected_difference_dice": selected["metrics"]["dice"]["mean_difference"],
                "selected_ci95_low": selected["metrics"]["dice"]["ci95_low"],
                "selected_ci95_high": selected["metrics"]["dice"]["ci95_high"],
                "fixed_0_5_difference_dice": fixed["metrics"]["dice"]["mean_difference"],
                "fixed_0_5_ci95_low": fixed["metrics"]["dice"]["ci95_low"],
                "fixed_0_5_ci95_high": fixed["metrics"]["dice"]["ci95_high"],
            }
        )
        for row in selected["area_bins"]:
            all_area_rows.append({"seed": seed, **row})
        integrity[str(seed)] = {
            **config_audit,
            "run_id": candidate_summary["run_id"],
            "records": 294,
            "augmentation_identity_fallbacks": int(safety["identity_fallbacks"]),
            "candidate_csv_sha256": sha256(candidate_dir / "validation_per_image_selected.csv"),
            "baseline_csv_sha256": sha256(baseline_dir / "validation_per_image_selected.csv"),
        }

    deltas = [float(row["selected_difference_dice"]) for row in per_seed]
    fixed_deltas = [float(row["fixed_0_5_difference_dice"]) for row in per_seed]
    decision = {
        "mean_selected_dice_delta": statistics.fmean(deltas),
        "sample_sd_selected_dice_delta": statistics.stdev(deltas),
        "seeds_won_selected": sum(value > 0 for value in deltas),
        "mean_fixed_0_5_dice_delta": statistics.fmean(fixed_deltas),
        "sample_sd_fixed_0_5_dice_delta": statistics.stdev(fixed_deltas),
        "seeds_won_fixed_0_5": sum(value > 0 for value in fixed_deltas),
        "minimum_mean_selected_delta": 0.005,
        "minimum_seed_wins": 2,
    }
    decision["advance"] = bool(
        decision["mean_selected_dice_delta"] >= decision["minimum_mean_selected_delta"]
        and decision["seeds_won_selected"] >= decision["minimum_seed_wins"]
    )
    decision["threshold_lower_bound_hits"] = sum(
        math.isclose(float(row["DR-IFT-B_threshold"]), 0.05) for row in per_seed
    )

    summary = {
        "status": "completed",
        "partition": "validation",
        "comparison": "DR-IFT-B_minus_DR-IFT",
        "classification": "post_registration_exploratory_ablation",
        "statistical_unit": "canonical_image_within_seed; seed_for_training_variability",
        "seeds": list(SEEDS),
        "integrity": integrity,
        "per_seed": seed_details,
        "decision": decision,
        "inference_note": (
            "Paired image bootstrap intervals are conditional on each training run; "
            "three seeds support descriptive mean/SD, not asymptotic inference over seeds."
        ),
        "protected_partitions_accessed": False,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "per_seed_results.csv", per_seed)
    write_csv(output / "area_bin_results.csv", all_area_rows)

    try:
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7, 5))
        for row in per_seed:
            axis.plot(
                ["DR-IFT", "DR-IFT-B"],
                [row["DR-IFT_dice"], row["DR-IFT-B_dice"]],
                marker="o",
                label=f"seed {row['seed']}",
            )
        axis.set(ylabel="Dice macro por imagem", title="Validação — efeito da loss de perímetro")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "dice_by_seed.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 5))
        for index, row in enumerate(per_seed):
            delta = float(row["selected_difference_dice"])
            low = float(row["selected_ci95_low"])
            high = float(row["selected_ci95_high"])
            axis.errorbar(delta, index, xerr=[[delta - low], [high - delta]], fmt="o")
        axis.axvline(0, color="black", linestyle="--")
        axis.axvline(0.005, color="gray", linestyle=":", label="ganho médio mínimo")
        axis.set(
            yticks=range(len(per_seed)),
            yticklabels=[f"seed {row['seed']}" for row in per_seed],
            xlabel="DR-IFT-B − DR-IFT (Dice)",
            title="Diferença pareada por seed — threshold selecionado",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "dice_effects_by_seed.png", dpi=180)
        plt.close(figure)
    except (ImportError, AttributeError) as error:
        (output / "figure_generation_skipped.txt").write_text(
            f"Matplotlib indisponível neste ambiente: {error}\n", encoding="utf-8"
        )

    print(json.dumps({"decision": decision, "per_seed": per_seed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
