"""Pré-processamentos de cor determinísticos usados nas ablações da comparação de pré-processamento."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


ImageTransform = Callable[[np.ndarray], np.ndarray]
PairTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _validate_rgb_uint8(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Imagem RGB uint8 esperada; " f"dtype={image.dtype}, shape={image.shape}")


def shades_of_gray(image: np.ndarray, minkowski_p: float = 6.0) -> np.ndarray:
    """Aplica constância de cor Shades-of-Gray e devolve RGB uint8.

    A iluminação de cada canal é estimada pela norma de Minkowski. O vetor é
    normalizado e cada canal é escalado em direção a um iluminante acromático.
    """

    _validate_rgb_uint8(image)
    if not np.isfinite(minkowski_p) or minkowski_p <= 0:
        raise ValueError("minkowski_p deve ser positivo e finito")

    values = image.astype(np.float64) / 255.0
    illuminant = np.mean(np.power(values, minkowski_p), axis=(0, 1))
    illuminant = np.power(illuminant, 1.0 / minkowski_p)
    norm = float(np.linalg.norm(illuminant))
    if norm <= np.finfo(np.float64).eps:
        return image.copy()

    illuminant /= norm
    neutral = 1.0 / np.sqrt(3.0)
    scales = neutral / np.maximum(illuminant, np.finfo(np.float64).eps)
    corrected = np.clip(values * scales[None, None, :], 0.0, 1.0)
    return np.rint(corrected * 255.0).astype(np.uint8)


def clahe_l_star(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Aplica CLAHE somente à luminosidade L* e retorna RGB uint8."""

    _validate_rgb_uint8(image)
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("clip_limit deve ser positivo e finito")
    if len(tile_grid_size) != 2 or any(int(value) <= 0 for value in tile_grid_size):
        raise ValueError("tile_grid_size deve conter dois inteiros positivos")

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - exercitado no Colab
        raise RuntimeError("OpenCV é necessário para CLAHE em L*") from error

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    corrected = lab.copy()
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=tuple(int(value) for value in tile_grid_size),
    )
    corrected[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB)


def build_color_preprocessor(
    name: str,
    *,
    minkowski_p: float = 6.0,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> ImageTransform:
    """Constrói P0, P1 ou P2 sem permitir combinações implícitas."""

    normalized = str(name).strip().casefold().replace("-", "_")
    if normalized in {"p0", "none", "rgb"}:
        return lambda image: image.copy()
    if normalized in {"p1", "shades_of_gray"}:
        return lambda image: shades_of_gray(image, minkowski_p=minkowski_p)
    if normalized in {"p2", "clahe_l_star", "clahe_l"}:
        return lambda image: clahe_l_star(
            image, clip_limit=clip_limit, tile_grid_size=tile_grid_size
        )
    raise ValueError(f"Pré-processamento desconhecido: {name}")


@dataclass(frozen=True)
class ImageOnlyPairTransform:
    """Aplica uma transformação somente à imagem, preservando a máscara."""

    transform: ImageTransform

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.transform(image), mask


@dataclass(frozen=True)
class ComposePair:
    """Compõe transformações pareadas na ordem declarada."""

    transforms: Sequence[PairTransform]

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_image, current_mask = image, mask
        for transform in self.transforms:
            current_image, current_mask = transform(current_image, current_mask)
        return current_image, current_mask
