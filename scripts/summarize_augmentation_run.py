#!/usr/bin/env python3
"""Audita um run A1/A2 e produz comparação descritiva pareada com D0.

Usa somente artefatos de validação já gerados. Não acessa imagens, treino, teste ou
FUSeg. Inferência conjunta e correção de multiplicidade ficam para a comparação final
após a disponibilidade simultânea de A1 e A2.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ("dice", "iou", "precision", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--d0-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def indexed_metrics(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    rows = read_csv(path)
    ids = [row["canonical_id"] for row in rows]
    if len(ids) != 294 or len(set(ids)) != 294:
        raise ValueError(f"{path}: esperado 294 IDs canônicos únicos")
    return ids, {
        row["canonical_id"]: {metric: float(row[metric]) for metric in METRICS} for row in rows
    }


def paired_descriptive(
    ids: list[str], candidate: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]
) -> dict[str, object]:
    output: dict[str, object] = {}
    for metric in METRICS:
        differences = [candidate[item][metric] - baseline[item][metric] for item in ids]
        output[metric] = {
            "mean_difference": statistics.fmean(differences),
            "median_difference": statistics.median(differences),
            "improved_images": sum(value > 0 for value in differences),
            "equal_images": sum(value == 0 for value in differences),
            "worsened_images": sum(value < 0 for value in differences),
        }
    return output


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    d0_dir = args.d0_dir.resolve()
    final = read_json(run_dir / "final_summary.json")
    safety = read_json(run_dir / "augmentation_safety.json")
    environment = read_json(run_dir / "environment_report.json")
    history = read_csv(run_dir / "history.csv")
    if len(history) != int(final["training"]["epochs_executed"]):
        raise ValueError("Histórico não reconcilia com epochs_executed")
    expected_samples = int(environment["train_records"]) * len(history)
    if int(safety["samples_seen"]) != expected_samples:
        raise ValueError("samples_seen não reconcilia treino × épocas")
    if any(
        int(row.get("augmentation_samples_seen", 0)) != int(environment["train_records"])
        for row in history
    ):
        raise ValueError("Uma ou mais épocas não viram todas as unidades de treino")

    best_epoch = int(final["training"]["best_epoch"])
    best_row = next(row for row in history if int(row["epoch"]) == best_epoch)
    minimum_validation_loss_row = min(history, key=lambda row: float(row["validation_loss"]))
    candidate_ids, candidate = indexed_metrics(run_dir / "validation_per_image_selected.csv")
    d0_ids, baseline = indexed_metrics(d0_dir / "validation_per_image_selected.csv")
    if candidate_ids != d0_ids:
        raise ValueError("Identidade ou ordem da validação difere do D0")

    selection = final["selection"]
    selected_metrics = selection["metrics_selected"]["macro"]
    fixed_metrics = selection["metrics_at_0_5"]["macro"]
    d0_final = read_json(d0_dir / "final_summary.json")
    d0_selection = d0_final["selection"]
    result = {
        "status": "validated",
        "run_id": final["run_id"],
        "experiment_id": final["experiment_id"],
        "seed": final["seed"],
        "records": {
            "train": environment["train_records"],
            "validation": environment["validation_records"],
            "history_epochs": len(history),
        },
        "training": {
            "best_epoch": best_epoch,
            "best_validation_dice_at_0_5": float(best_row["validation_dice_at_0_5"]),
            "minimum_validation_loss_epoch": int(minimum_validation_loss_row["epoch"]),
            "minimum_validation_loss": float(minimum_validation_loss_row["validation_loss"]),
            "final_train_loss": float(history[-1]["train_loss"]),
            "final_validation_loss": float(history[-1]["validation_loss"]),
            "stopped_early": bool(final["training"]["stopped_early"]),
        },
        "threshold": {
            "selected": float(selection["threshold"]),
            "macro_dice_at_0_5": float(fixed_metrics["dice"]["mean"]),
            "macro_dice_selected": float(selected_metrics["dice"]["mean"]),
            "dice_gain_from_calibration": float(
                selected_metrics["dice"]["mean"] - fixed_metrics["dice"]["mean"]
            ),
            "precision_change": float(
                selected_metrics["precision"]["mean"] - fixed_metrics["precision"]["mean"]
            ),
            "recall_change": float(
                selected_metrics["recall"]["mean"] - fixed_metrics["recall"]["mean"]
            ),
        },
        "safety": {
            **safety,
            "empty_rejection_rate": int(safety["empty_mask_rejections"])
            / int(safety["samples_seen"]),
        },
        "descriptive_vs_d0": {
            "at_0_5": {
                "candidate_dice": float(fixed_metrics["dice"]["mean"]),
                "d0_dice": float(d0_selection["metrics_at_0_5"]["macro"]["dice"]["mean"]),
                "difference": float(
                    fixed_metrics["dice"]["mean"]
                    - d0_selection["metrics_at_0_5"]["macro"]["dice"]["mean"]
                ),
            },
            "at_each_selected_threshold": {
                "candidate_threshold": float(selection["threshold"]),
                "d0_threshold": float(d0_selection["threshold"]),
                "candidate_dice": float(selected_metrics["dice"]["mean"]),
                "d0_dice": float(d0_selection["metrics_selected"]["macro"]["dice"]["mean"]),
                "difference": float(
                    selected_metrics["dice"]["mean"]
                    - d0_selection["metrics_selected"]["macro"]["dice"]["mean"]
                ),
                "paired_by_image": paired_descriptive(candidate_ids, candidate, baseline),
            },
            "inferential_status": "deferred_until_A1_and_A2_are_jointly_available",
        },
        "partitions_accessed_by_analysis": ["validation_artifacts"],
        "protected_partitions_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
