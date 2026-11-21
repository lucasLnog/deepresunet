from __future__ import annotations

import numpy as np
import pytest

from tcc_segmentation.data.preprocessing import (
    ComposePair,
    ImageOnlyPairTransform,
    build_color_preprocessor,
    clahe_l_star,
    shades_of_gray,
)


def sample_rgb() -> np.ndarray:
    y, x = np.mgrid[0:32, 0:48]
    return np.stack([40 + 2 * x, 25 + 3 * y, 15 + x + y], axis=-1).clip(0, 255).astype(np.uint8)


def test_shades_of_gray_is_deterministic_and_preserves_contract() -> None:
    image = sample_rgb()
    first = shades_of_gray(image, minkowski_p=6)
    second = shades_of_gray(image, minkowski_p=6)
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert np.array_equal(first, second)
    assert not np.array_equal(first, image)


def test_shades_of_gray_preserves_black_image() -> None:
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    assert np.array_equal(shades_of_gray(image), image)


def test_clahe_l_star_is_deterministic_and_preserves_contract() -> None:
    pytest.importorskip("cv2")
    image = sample_rgb()
    first = clahe_l_star(image, clip_limit=2.0, tile_grid_size=(8, 8))
    second = clahe_l_star(image, clip_limit=2.0, tile_grid_size=(8, 8))
    assert first.dtype == np.uint8
    assert first.shape == image.shape
    assert np.array_equal(first, second)


def test_pair_transform_never_changes_mask() -> None:
    image = sample_rgb()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[5:10, 7:15] = 1
    transform = ComposePair([ImageOnlyPairTransform(build_color_preprocessor("P1", minkowski_p=6))])
    transformed_image, transformed_mask = transform(image, mask)
    assert not np.array_equal(transformed_image, image)
    assert np.array_equal(transformed_mask, mask)


def test_unknown_preprocessing_is_rejected() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        build_color_preprocessor("P1_plus_P2")
