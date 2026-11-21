from pathlib import Path

import pytest

from tcc_segmentation.thesis_reporting import (
    latex_escape,
    matching_rows,
    require_one,
    write_latex_table,
)


def test_matching_and_require_one() -> None:
    rows = [{"dataset": "A", "arm": "x"}, {"dataset": "A", "arm": "y"}]
    assert matching_rows(rows, dataset="A", arm="x") == [{"dataset": "A", "arm": "x"}]
    assert require_one(rows, arm="y")["dataset"] == "A"
    with pytest.raises(ValueError):
        require_one(rows, dataset="A")


def test_latex_escape() -> None:
    assert latex_escape("DRIFT_SAM & 95%") == r"DRIFT\_SAM \& 95\%"


def test_write_latex_table(tmp_path: Path) -> None:
    path = tmp_path / "table.tex"
    write_latex_table(
        path, ["Método", "Dice"], [["DR-IFT", "0.7682"]], "Resultado", "tab:resultado"
    )
    text = path.read_text(encoding="utf-8")
    assert "\\begin{table}" in text
    assert "DR-IFT & 0.7682" in text
    assert "\\label{tab:resultado}" in text
