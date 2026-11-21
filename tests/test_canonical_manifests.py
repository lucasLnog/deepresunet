"""Testes de integracao da preparacao canonica."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from scripts.build_canonical_manifests import load_binary_mask, load_simple_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = PROJECT_ROOT / "manifests" / "canonical"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class CanonicalManifestTests(unittest.TestCase):
    def test_policy_parser(self) -> None:
        policy = load_simple_yaml(PROJECT_ROOT / "configs" / "data_policy.yaml")
        self.assertEqual(policy["global"]["mask_binarization_threshold"], 128)
        self.assertEqual(policy["DFUC2022"]["expected_canonical_units"], 1961)
        self.assertEqual(policy["FUSeg"]["expected_unique_evaluation_units"], 962)

    def test_canonical_counts_and_source_totals(self) -> None:
        dfuc = read_csv(CANONICAL_DIR / "dfuc2022_canonical.csv")
        fuseg = read_csv(CANONICAL_DIR / "fuseg_external_canonical.csv")
        self.assertEqual(len(dfuc), 1961)
        self.assertEqual(sum(int(row["source_count"]) for row in dfuc), 2000)
        self.assertEqual(sum(int(row["source_count"]) > 1 for row in dfuc), 39)
        self.assertEqual(len(fuseg), 962)
        self.assertEqual(sum(int(row["source_count"]) for row in fuseg), 986)
        self.assertEqual(sum(int(row["source_count"]) > 1 for row in fuseg), 23)

    @unittest.skipUnless(
        (PROJECT_ROOT / "Dataset").is_dir(),
        "Este teste compara os manifestos com máscaras do conjunto local.",
    )
    def test_dfuc_or_summary_matches_source_masks(self) -> None:
        dfuc = read_csv(CANONICAL_DIR / "dfuc2022_canonical.csv")
        duplicate = next(row for row in dfuc if int(row["source_count"]) > 1)
        masks = [
            load_binary_mask(str(PROJECT_ROOT / path), 128)
            for path in duplicate["mask_paths"].split("|")
        ]
        union = np.logical_or.reduce(masks)
        self.assertEqual(int(union.sum()), int(float(duplicate["foreground_pixels"])))
        self.assertAlmostEqual(float(union.mean()), float(duplicate["foreground_fraction"]))


if __name__ == "__main__":
    unittest.main()
