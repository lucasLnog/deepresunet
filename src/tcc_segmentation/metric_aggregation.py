"""Agregacoes reproduziveis que nao dependem de PyTorch."""

from __future__ import annotations

from statistics import mean, median, stdev
from typing import Any, Sequence


def aggregate_global_pixel_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Agrega contagens por imagem e calcula metricas micro por pixels."""
    if not rows:
        raise ValueError("Nao e possivel agregar uma lista vazia")

    required = {"tp", "fp", "fn", "tn", "dice"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Linha {index} sem colunas obrigatorias: {sorted(missing)}")

    tp = sum(int(float(row["tp"])) for row in rows)
    fp = sum(int(float(row["fp"])) for row in rows)
    fn = sum(int(float(row["fn"])) for row in rows)
    tn = sum(int(float(row["tn"])) for row in rows)

    def divide(numerator: float, denominator: float, empty_value: float) -> float:
        return float(numerator / denominator) if denominator > 0 else empty_value

    dice_micro = divide(2 * tp, 2 * tp + fp + fn, 1.0)
    iou_micro = divide(tp, tp + fp + fn, 1.0)
    precision_micro = divide(tp, tp + fp, 1.0)
    recall_micro = divide(tp, tp + fn, 1.0)
    specificity_micro = divide(tn, tn + fp, 1.0)
    accuracy_micro = divide(tp + tn, tp + tn + fp + fn, 1.0)
    balanced_accuracy_micro = 0.5 * (recall_micro + specificity_micro)

    dice_values = [float(row["dice"]) for row in rows]
    zero_rows = [row for row in rows if float(row["dice"]) == 0.0]
    zero_target_pixels = sum(int(float(row["tp"])) + int(float(row["fn"])) for row in zero_rows)
    all_target_pixels = sum(int(float(row["tp"])) + int(float(row["fn"])) for row in rows)
    macro_mean = mean(dice_values)

    return {
        "images": len(rows),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "micro": {
            "dice": dice_micro,
            "iou": iou_micro,
            "precision": precision_micro,
            "recall": recall_micro,
            "specificity": specificity_micro,
            "accuracy": accuracy_micro,
            "balanced_accuracy": balanced_accuracy_micro,
        },
        "macro_dice": {
            "mean": macro_mean,
            "median": median(dice_values),
            "std": stdev(dice_values) if len(rows) > 1 else 0.0,
        },
        "aggregation_gap": {"dice_micro_minus_macro": dice_micro - macro_mean},
        "zero_dice_cases": {
            "images": len(zero_rows),
            "image_fraction": len(zero_rows) / len(rows),
            "target_pixels": zero_target_pixels,
            "target_pixel_fraction": divide(zero_target_pixels, all_target_pixels, 0.0),
        },
    }
