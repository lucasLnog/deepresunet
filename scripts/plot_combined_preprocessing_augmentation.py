"""Gera SVG da comparação D0, P1, A1 e P1+A1 na validação."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-summary", type=Path, required=True)
    parser.add_argument("--selected-comparisons", type=Path, required=True)
    parser.add_argument("--fixed-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def label(x: float, y: float, value: str, **attrs: object) -> str:
    attributes = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(val))}"' for key, val in attrs.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(value)}</text>'


def main() -> int:
    args = parse_args()
    selected = json.loads(args.selected_summary.read_text(encoding="utf-8"))
    fixed = json.loads(args.fixed_summary.read_text(encoding="utf-8"))
    with args.selected_comparisons.open("r", encoding="utf-8-sig", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    contrasts = {row["comparison"]: row for row in comparison_rows if row["metric"] == "dice"}

    width, height = 1380, 640
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202020}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.4}</style>",
        label(
            width / 2,
            36,
            "Combinação P1+A1 — validação congelada, seed 42",
            text_anchor="middle",
            font_size=24,
            font_weight="bold",
        ),
    ]

    left, top, chart_w, chart_h = 80, 105, 620, 410
    minimum, maximum = 0.62, 0.71
    y = lambda value: top + chart_h - (value - minimum) / (maximum - minimum) * chart_h
    parts.append(
        label(
            left + chart_w / 2,
            76,
            "Dice macro por imagem (n=294)",
            text_anchor="middle",
            font_size=18,
        )
    )
    for tick in (0.62, 0.64, 0.66, 0.68, 0.70):
        yy = y(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{yy}" x2="{left + chart_w}" y2="{yy}"/>',
                label(left - 10, yy + 5, f"{tick:.2f}", text_anchor="end", font_size=12),
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}"/>',
        ]
    )
    arms = ("D0", "P1", "A1", "P1A1")
    names = ("D0", "P1", "A1", "P1+A1")
    colors = ("#777777", "#5B8E7D", "#2374AB", "#6A4C93")
    centers = (left + 85, left + 235, left + 385, left + 535)
    bar_width = 42
    for arm, name, color, center in zip(arms, names, colors, centers):
        for offset, source, opacity in ((-23, selected, 1.0), (23, fixed, 0.40)):
            value = float(source["arms"][arm]["dice"])
            yy = y(value)
            parts.append(
                f'<rect x="{center + offset - bar_width / 2}" y="{yy}" width="{bar_width}" '
                f'height="{top + chart_h - yy}" fill="{color}" fill-opacity="{opacity}" stroke="{color}"/>'
            )
            if offset < 0:
                parts.append(
                    label(
                        center + offset,
                        yy - 8,
                        f"{value:.3f}",
                        text_anchor="middle",
                        font_size=12,
                        font_weight="bold",
                    )
                )
        parts.append(
            label(
                center,
                top + chart_h + 30,
                name,
                text_anchor="middle",
                font_size=15,
                font_weight="bold",
            )
        )
    parts.extend(
        [
            '<rect x="145" y="570" width="16" height="16" fill="#2374AB"/>',
            label(168, 583, "threshold próprio", font_size=12),
            '<rect x="335" y="570" width="16" height="16" fill="#2374AB" fill-opacity="0.40" stroke="#2374AB"/>',
            label(358, 583, "threshold 0,5", font_size=12),
        ]
    )

    right, plot_w = 850, 450
    x_min, x_max = -0.025, 0.030
    x = lambda value: right + (value - x_min) / (x_max - x_min) * plot_w
    parts.append(
        label(
            right + plot_w / 2,
            76,
            "Contrastes pareados — IC 95% bootstrap",
            text_anchor="middle",
            font_size=18,
        )
    )
    for tick in (-0.02, -0.01, 0.00, 0.01, 0.02, 0.03):
        xx = x(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{xx}" y1="{top}" x2="{xx}" y2="445"/>',
                label(xx, 470, f"{tick:+.2f}", text_anchor="middle", font_size=12),
            ]
        )
    parts.append(
        f'<line x1="{x(0)}" y1="{top}" x2="{x(0)}" y2="445" stroke="#222" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    items = (
        (205, "P1A1-A1", "P1+A1 − A1", "#6A4C93"),
        (345, "(P1A1-P1)-(A1-D0)", "interação descritiva", "#B56576"),
    )
    for row_y, key, name, color in items:
        row = contrasts[key]
        mean, low, high = (
            float(row[field]) for field in ("mean_difference", "ci95_low", "ci95_high")
        )
        parts.extend(
            [
                label(
                    right - 18, row_y + 6, name, text_anchor="end", font_size=15, font_weight="bold"
                ),
                f'<line x1="{x(low)}" y1="{row_y}" x2="{x(high)}" y2="{row_y}" stroke="{color}" stroke-width="6"/>',
                f'<line x1="{x(low)}" y1="{row_y - 10}" x2="{x(low)}" y2="{row_y + 10}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{x(high)}" y1="{row_y - 10}" x2="{x(high)}" y2="{row_y + 10}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x(mean)}" cy="{row_y}" r="8" fill="{color}" stroke="white" stroke-width="2"/>',
                label(x(high) + 8, row_y + 5, f"{mean:+.3f}", font_size=13, font_weight="bold"),
            ]
        )
    parts.extend(
        [
            label(
                right + plot_w / 2,
                520,
                "P1+A1 tem a maior média, mas não demonstrou superar A1.",
                text_anchor="middle",
                font_size=14,
                font_weight="bold",
            ),
            label(
                right + plot_w / 2,
                548,
                "Contraste primário: p=0,305; IC inclui zero.",
                text_anchor="middle",
                font_size=13,
            ),
            label(
                right + plot_w / 2,
                578,
                "Triagem em uma seed; robustez multisseed ainda necessária.",
                text_anchor="middle",
                font_size=12,
            ),
        ]
    )

    parts.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
