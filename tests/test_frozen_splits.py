from pathlib import Path
from shutil import copyfile

from scripts.create_frozen_splits import (
    SPLIT_NAMES,
    add_strata,
    assign_splits,
    read_csv,
    validate_no_overlap,
)
from scripts.verify_frozen_split import verify


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "manifests" / "splits"


def test_frozen_split_integrity() -> None:
    result = verify(SPLIT_DIR)
    assert result["status"] == "ok"
    assert result["counts"] == {"train": 1373, "validation": 294, "test": 294}
    assert all(value == 0 for value in result["overlap_counts"].values())
    assert result["images_or_masks_opened"] is False


def test_assignment_is_deterministic() -> None:
    rows = read_csv(ROOT / "manifests" / "canonical" / "dfuc2022_canonical.csv")
    enriched, _ = add_strata(rows, 5)
    first, _ = assign_splits(enriched, (0.70, 0.15, 0.15), 42)
    second, _ = assign_splits(enriched, (0.70, 0.15, 0.15), 42)
    first_map = {row["canonical_id"]: row["split"] for row in first}
    second_map = {row["canonical_id"]: row["split"] for row in second}
    assert first_map == second_map
    validate_no_overlap(first)


def test_frozen_splits_can_be_verified_without_local_audit_records(tmp_path: Path) -> None:
    manifest_names = [f"dfuc_{name}.csv" for name in SPLIT_NAMES]
    for name in [*manifest_names, "split_summary.json"]:
        copyfile(SPLIT_DIR / name, tmp_path / name)

    result = verify(tmp_path)

    assert result["status"] == "ok"
    assert result["counts"]["test"] == 294
