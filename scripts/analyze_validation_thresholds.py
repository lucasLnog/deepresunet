#!/usr/bin/env python3
"""Seleciona um threshold global usando apenas o melhor checkpoint e a validação."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


def build_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    if not 0 < start <= stop < 1 or step <= 0:
        raise ValueError("Grid de threshold inválido")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] > stop + 1e-9:
        values.pop()
    return values


def select_threshold(curve: Sequence[dict[str, Any]], plateau_tolerance: float) -> dict[str, Any]:
    if not curve:
        raise ValueError("Curva de threshold vazia")
    if plateau_tolerance < 0:
        raise ValueError("plateau_tolerance deve ser não negativo")
    best_dice = max(float(row["dice_mean"]) for row in curve)
    eligible = [row for row in curve if float(row["dice_mean"]) >= best_dice - plateau_tolerance]
    selected = min(
        eligible,
        key=lambda row: (abs(float(row["threshold"]) - 0.5), float(row["threshold"])),
    )
    return {
        "selected_threshold": float(selected["threshold"]),
        "selected_dice_mean": float(selected["dice_mean"]),
        "raw_max_dice_mean": best_dice,
        "eligible_thresholds": [float(row["threshold"]) for row in eligible],
        "plateau_tolerance": plateau_tolerance,
        "tie_break": "closest_to_0.5_then_lower",
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Não é possível escrever CSV vazio")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else project_root
    run_dir = args.run_dir if args.run_dir.is_absolute() else project_root / args.run_dir
    config_path = run_dir / "config_snapshot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(project_root))
    from scripts.run_development_baseline import validate_baseline_config

    validate_baseline_config(config)
    sys.path.insert(0, str(project_root / "src"))
    import torch

    from tcc_segmentation.data import DevelopmentDataset, ResizePad, create_development_dataloader
    from tcc_segmentation.models import build_segmentation_model
    from tcc_segmentation.training import load_training_checkpoint
    from tcc_segmentation.training.metrics import aggregate_metric_rows, metrics_to_rows
    from tcc_segmentation.training.reporting import write_json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bool(config["training"]["require_cuda"]) and device.type != "cuda":
        raise RuntimeError("CUDA obrigatória para análise completa de threshold")
    amp = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    data_config = config["data"]
    transform = ResizePad(height=int(data_config["height"]), width=int(data_config["width"]))
    validation_manifest = project_root / str(data_config["validation_manifest"])
    dataset = DevelopmentDataset(
        validation_manifest,
        data_root,
        expected_split="validation",
        transform=transform,
        mask_threshold=int(data_config["mask_threshold"]),
    )
    loader = create_development_dataloader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        seed=int(config["training"]["seed"]),
        num_workers=int(data_config["num_workers"]),
    )
    model = build_segmentation_model(config["model"]).to(device)
    checkpoint = load_training_checkpoint(
        run_dir / "checkpoints/best.pt",
        model=model,
        map_location=device,
        restore_rng=False,
    )
    if checkpoint["config"] != config:
        raise RuntimeError("Configuração do checkpoint difere do snapshot da execução")
    model.eval()

    threshold_config = config["threshold_analysis"]
    thresholds = build_threshold_grid(
        float(threshold_config["start"]),
        float(threshold_config["stop"]),
        float(threshold_config["step"]),
    )
    rows_by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in thresholds
    }
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True) >= 0.5
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(images)
            probabilities = torch.sigmoid(logits.float()).flatten(start_dim=1)
            references = targets.flatten(start_dim=1)
            canonical_ids = list(batch["canonical_id"])
            for threshold in thresholds:
                predictions = probabilities >= threshold
                tp = (predictions & references).sum(dim=1).float()
                fp = (predictions & ~references).sum(dim=1).float()
                fn = (~predictions & references).sum(dim=1).float()
                tn = (~predictions & ~references).sum(dim=1).float()

                def divide(numerator, denominator, empty_value: float):
                    fallback = torch.full_like(numerator, empty_value)
                    return torch.where(
                        denominator > 0, numerator / denominator.clamp_min(1.0), fallback
                    )

                recall = divide(tp, tp + fn, 1.0)
                specificity = divide(tn, tn + fp, 1.0)
                metrics = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "dice": divide(2 * tp, 2 * tp + fp + fn, 1.0),
                    "iou": divide(tp, tp + fp + fn, 1.0),
                    "precision": torch.where(
                        (tp + fp) > 0,
                        tp / (tp + fp).clamp_min(1.0),
                        torch.where(
                            (tp + fn) > 0,
                            torch.zeros_like(tp),
                            torch.ones_like(tp),
                        ),
                    ),
                    "recall": recall,
                    "specificity": specificity,
                    "accuracy": divide(tp + tn, tp + tn + fp + fn, 1.0),
                    "balanced_accuracy": 0.5 * (recall + specificity),
                }
                rows_by_threshold[threshold].extend(metrics_to_rows(metrics, canonical_ids))

    curve = []
    summaries = {}
    for threshold in thresholds:
        summary = aggregate_metric_rows(rows_by_threshold[threshold])
        summaries[str(threshold)] = summary
        curve.append(
            {
                "threshold": threshold,
                **{
                    f"{metric}_mean": summary[metric]["mean"]
                    for metric in (
                        "dice",
                        "iou",
                        "precision",
                        "recall",
                        "specificity",
                        "accuracy",
                        "balanced_accuracy",
                    )
                },
            }
        )
    selection = select_threshold(curve, float(threshold_config["plateau_tolerance"]))
    selected_threshold = selection["selected_threshold"]
    write_csv(run_dir / "threshold_curve.csv", curve)
    write_csv(
        run_dir / "per_image/selected_threshold_validation.csv",
        rows_by_threshold[selected_threshold],
    )
    result = {
        "status": "completed",
        "run_id": checkpoint["run_id"],
        "scientific_result": True,
        "development_only": True,
        "partition": "validation",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection_metric": "mean_per_image_dice",
        **selection,
        "metrics_at_selected_threshold": summaries[str(selected_threshold)],
        "metrics_at_0.5": summaries["0.5"],
        "thresholds_evaluated": len(thresholds),
        "protected_partitions_accessed": False,
    }
    write_json(run_dir / "threshold_selection.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
