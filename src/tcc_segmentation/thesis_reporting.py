"""Build tables and figures from the final experimental results.

This module only reshapes already approved result tables.  It deliberately contains
no inference, model selection, threshold search, or statistical recomputation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def matching_rows(rows: Iterable[Mapping[str, str]], **criteria: str) -> list[dict[str, str]]:
    return [
        dict(row) for row in rows if all(row.get(key) == value for key, value in criteria.items())
    ]


def require_one(rows: Iterable[Mapping[str, str]], **criteria: str) -> dict[str, str]:
    matches = matching_rows(rows, **criteria)
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def as_float(row: Mapping[str, str], key: str) -> float:
    return float(row[key])


def fmt_metric(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def write_latex_table(
    path: Path,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(item) for item in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(latex_escape(item) for item in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
