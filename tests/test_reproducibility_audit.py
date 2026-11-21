from pathlib import Path

from tcc_segmentation.reproducibility_audit import grouped_unique_counts, hash_matches, sha256


def test_sha256_and_match(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("TCC", encoding="utf-8")
    digest = sha256(path)
    assert len(digest) == 64
    assert hash_matches(path, digest.upper())


def test_grouped_unique_counts() -> None:
    rows = [
        {"arm": "A", "policy": "selected", "id": "1"},
        {"arm": "A", "policy": "selected", "id": "2"},
        {"arm": "A", "policy": "selected", "id": "2"},
        {"arm": "B", "policy": "fixed", "id": "1"},
    ]
    counts = grouped_unique_counts(rows, ("arm", "policy"), "id")
    assert counts == {("A", "selected"): 2, ("B", "fixed"): 1}
