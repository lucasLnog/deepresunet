"""Testes unitarios das funcoes criticas da auditoria."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from scripts.audit_datasets import count_components_8, dhash64, phash64


class ConnectedComponentsTests(unittest.TestCase):
    def test_empty_mask_has_zero_components(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        self.assertEqual(count_components_8(mask), 0)

    def test_diagonal_pixels_are_connected_with_connectivity_8(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 1] = True
        mask[2, 2] = True
        self.assertEqual(count_components_8(mask), 1)

    def test_distant_regions_are_separate(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[1:3, 1:3] = True
        mask[5, 5] = True
        self.assertEqual(count_components_8(mask), 2)

    def test_bridge_unifies_previous_labels(self) -> None:
        mask = np.zeros((5, 7), dtype=bool)
        mask[1, 1] = True
        mask[1, 5] = True
        mask[2, 1:6] = True
        self.assertEqual(count_components_8(mask), 1)


class PerceptualHashTests(unittest.TestCase):
    def test_hashes_are_stable_and_64_bit(self) -> None:
        pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
        image = Image.fromarray(pixels, mode="L").convert("RGB")
        self.assertEqual(dhash64(image), dhash64(image.copy()))
        self.assertEqual(phash64(image), phash64(image.copy()))
        self.assertEqual(len(dhash64(image)), 16)
        self.assertEqual(len(phash64(image)), 16)


if __name__ == "__main__":
    unittest.main()
