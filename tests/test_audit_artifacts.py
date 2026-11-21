"""Testes de integracao dos artefatos produzidos pela auditoria local."""

from __future__ import annotations

import csv
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "results" / "audit" / "2026-08-25"
AUDIT_TABLES = ("dataset_inventory.csv", "exact_duplicate_groups.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@unittest.skipUnless(
    all((AUDIT_DIR / name).is_file() for name in AUDIT_TABLES),
    "As tabelas históricas da auditoria são distribuídas com os resultados locais.",
)
class AuditArtifactTests(unittest.TestCase):
    def test_inventory_counts_and_integrity(self) -> None:
        rows = read_csv(AUDIT_DIR / "dataset_inventory.csv")
        by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_dataset[row["dataset"]].append(row)

        self.assertEqual(len(by_dataset["DFUC2022"]), 2000)
        self.assertEqual(len(by_dataset["FUSeg"]), 1010)
        self.assertTrue(all(row["status"] == "ok" for row in rows))
        self.assertTrue(all(row["dimensions_match"] == "True" for row in rows))

    def test_duplicate_group_counts(self) -> None:
        rows = read_csv(AUDIT_DIR / "exact_duplicate_groups.csv")
        groups: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            groups[row["dataset"]].add(row["group_id"])

        self.assertEqual(len(groups["DFUC2022"]), 39)
        self.assertEqual(len(groups["FUSeg"]), 24)

    def test_exclusions_match_empty_fuseg_masks(self) -> None:
        inventory = read_csv(AUDIT_DIR / "dataset_inventory.csv")
        exclusions = read_csv(PROJECT_ROOT / "configs" / "data_quality_exclusions.csv")
        empty_stems = {
            row["stem"]
            for row in inventory
            if row["dataset"] == "FUSeg" and row["mask_empty"] == "True"
        }
        excluded_stems = {row["stem"] for row in exclusions if row["dataset"] == "FUSeg"}

        self.assertEqual(len(empty_stems), 24)
        self.assertEqual(excluded_stems, empty_stems)


if __name__ == "__main__":
    unittest.main()
