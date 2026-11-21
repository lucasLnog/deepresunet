"""Transformacoes deterministicas compartilhadas por imagem e mascara."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ResizePad:
    """Redimensiona preservando aspecto e completa com padding central."""

    height: int
    width: int
    image_fill: tuple[int, int, int] = (0, 0, 0)
    mask_fill: int = 0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height e width devem ser positivos")

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Imagem RGB esperada; shape observado={image.shape}")
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise ValueError("Imagem e mascara devem ter dimensoes espaciais iguais")

        source_height, source_width = mask.shape
        scale = min(self.width / source_width, self.height / source_height)
        resized_width = max(1, min(self.width, round(source_width * scale)))
        resized_height = max(1, min(self.height, round(source_height * scale)))

        image_pil = Image.fromarray(image, mode="RGB").resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        mask_pil = Image.fromarray(mask, mode="L").resize(
            (resized_width, resized_height), resample=Image.Resampling.NEAREST
        )

        image_canvas = Image.new("RGB", (self.width, self.height), self.image_fill)
        mask_canvas = Image.new("L", (self.width, self.height), self.mask_fill)
        left = (self.width - resized_width) // 2
        top = (self.height - resized_height) // 2
        image_canvas.paste(image_pil, (left, top))
        mask_canvas.paste(mask_pil, (left, top))
        return (
            np.asarray(image_canvas, dtype=np.uint8),
            (np.asarray(mask_canvas, dtype=np.uint8) > 0).astype(np.uint8),
        )
