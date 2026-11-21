"""Métricas transparentes para estudar união OR de máscaras duplicadas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def load_binary_mask(path: str | Path, threshold: int = 128) -> np.ndarray:
    with Image.open(path) as source:
        values = np.asarray(source.convert("L"), dtype=np.uint8)
    return values >= threshold


def analyze_mask_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float | int]:
    """Quantifica concordância e efeito de A OR B para duas máscaras booleanas."""
    if mask_a.shape != mask_b.shape:
        raise ValueError("As máscaras precisam ter o mesmo shape")
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    area_a = int(a.sum())
    area_b = int(b.sum())
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    disagreement = int(np.logical_xor(a, b).sum())
    denominator = area_a + area_b
    dice = 1.0 if denominator == 0 else 2.0 * intersection / denominator
    iou = 1.0 if union == 0 else intersection / union
    larger = max(area_a, area_b)
    mean_area = (area_a + area_b) / 2.0
    return {
        "area_a": area_a,
        "area_b": area_b,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "disagreement_pixels": disagreement,
        "dice_between_references": float(dice),
        "iou_between_references": float(iou),
        "or_added_over_larger_pixels": union - larger,
        "or_added_over_larger_fraction": (0.0 if larger == 0 else (union - larger) / larger),
        "or_added_over_mean_fraction": (0.0 if mean_area == 0 else (union - mean_area) / mean_area),
    }


def combine_masks_or(paths: Sequence[str | Path], threshold: int = 128) -> np.ndarray:
    if not paths:
        raise ValueError("Ao menos uma máscara é necessária")
    masks = [load_binary_mask(path, threshold=threshold) for path in paths]
    if any(mask.shape != masks[0].shape for mask in masks[1:]):
        raise ValueError("Máscaras do grupo possuem shapes diferentes")
    return np.logical_or.reduce(masks)
