"""Metricas binarias calculadas separadamente para cada imagem."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch


METRIC_NAMES = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def _divide(numerator: torch.Tensor, denominator: torch.Tensor, empty_value: float) -> torch.Tensor:
    value = numerator / denominator.clamp_min(1.0)
    fallback = torch.full_like(value, float(empty_value))
    return torch.where(denominator > 0, value, fallback)


@torch.no_grad()
def binary_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, torch.Tensor]:
    """Retorna um vetor por metrica, com um elemento por imagem."""
    if logits.shape != targets.shape:
        raise ValueError(f"Shapes diferentes: {logits.shape} != {targets.shape}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold deve estar estritamente entre 0 e 1")

    predictions = torch.sigmoid(logits.float()) >= threshold
    references = targets >= 0.5
    predictions = predictions.flatten(start_dim=1)
    references = references.flatten(start_dim=1)

    tp = (predictions & references).sum(dim=1).float()
    fp = (predictions & ~references).sum(dim=1).float()
    fn = (~predictions & references).sum(dim=1).float()
    tn = (~predictions & ~references).sum(dim=1).float()

    dice = _divide(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0)
    iou = _divide(tp, tp + fp + fn, empty_value=1.0)
    # Se a referência contém lesão e a predição é vazia, precisão é definida como 0
    # neste estudo: a falha completa não deve elevar a média. O valor 1 fica reservado
    # ao caso em que referência e predição são ambas vazias.
    precision_denominator = tp + fp
    precision = torch.where(
        precision_denominator > 0,
        tp / precision_denominator.clamp_min(1.0),
        torch.where(
            (tp + fn) > 0,
            torch.zeros_like(tp),
            torch.ones_like(tp),
        ),
    )
    recall = _divide(tp, tp + fn, empty_value=1.0)
    specificity = _divide(tn, tn + fp, empty_value=1.0)
    accuracy = _divide(tp + tn, tp + tn + fp + fn, empty_value=1.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def metrics_to_rows(
    metrics: dict[str, torch.Tensor], canonical_ids: Sequence[str]
) -> list[dict[str, Any]]:
    if not canonical_ids:
        return []
    expected = len(canonical_ids)
    if any(int(values.shape[0]) != expected for values in metrics.values()):
        raise ValueError("Numero de IDs difere do numero de metricas")
    rows: list[dict[str, Any]] = []
    for index, canonical_id in enumerate(canonical_ids):
        row: dict[str, Any] = {"canonical_id": str(canonical_id)}
        for name, values in metrics.items():
            value = values[index].detach().cpu().item()
            row[name] = int(value) if name in {"tp", "fp", "fn", "tn"} else float(value)
        rows.append(row)
    return rows


def aggregate_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Nao e possivel agregar uma lista vazia")
    result: dict[str, Any] = {"images": len(rows)}
    for metric in METRIC_NAMES:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result
