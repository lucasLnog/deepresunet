"""Leitura e composicao das referencias binarias canônicas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def load_binary_mask(path: str | Path, threshold: int = 128) -> np.ndarray:
    """Carrega uma mascara como matriz uint8 estritamente em {0, 1}."""
    with Image.open(path) as source:
        grayscale = np.asarray(source.convert("L"), dtype=np.uint8)
    return (grayscale >= threshold).astype(np.uint8)


def load_canonical_mask(
    paths: Sequence[str | Path], policy: str, threshold: int = 128
) -> np.ndarray:
    """Aplica a politica de referencia definida antes dos splits."""
    if not paths:
        raise ValueError("Unidade canonica sem mascara")
    masks = [load_binary_mask(path, threshold) for path in paths]
    shapes = {mask.shape for mask in masks}
    if len(shapes) != 1:
        raise ValueError(f"Mascaras da unidade possuem dimensoes diferentes: {shapes}")
    if policy != "logical_or":
        raise ValueError(f"Politica {policy!r} nao e permitida no dataloader de desenvolvimento")
    return np.logical_or.reduce(masks).astype(np.uint8)
