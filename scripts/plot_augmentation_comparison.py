"""Gera um SVG da comparação D0, A1 e A2 na validação congelada.

Usa somente a biblioteca padrão para evitar dependências gráficas locais. Não abre
imagens, checkpoints, teste interno ou FUSeg.
"""

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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dice_contrasts(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["comparison"]: {
            key: float(row[key]) for key in ("mean_difference", "ci95_low", "ci95_high")
        }
        for row in rows
        if row["metric"] == "dice" and row["comparison"] in {"A1-A0", "A2-A0"}
    }


def text(x: float, y: float, value: str, **attributes: object) -> str:
    attrs = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(val))}"' for key, val in attributes.items()
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{html.escape(value)}</text>'


def main() -> int:
    args = parse_args()
    selected = load_json(args.selected_summary)
    fixed = load_json(args.fixed_summary)
    contrasts = load_dice_contrasts(args.selected_comparisons)

    width, height = 1320, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#202020}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.4}</style>",
        text(
            width / 2,
            36,
            "Triagem de data augmentation — seed 42",
            text_anchor="middle",
            font_size=24,
            font_weight="bold",
        ),
    ]

    left, top, panel_w, panel_h = 85, 95, 520, 400
    minimum, maximum = 0.60, 0.70
    y = lambda value: top + panel_h - (value - minimum) / (maximum - minimum) * panel_h
    parts.append(
        text(
            left + panel_w / 2,
            72,
            "Desempenho na validação congelada (n = 294)",
            text_anchor="middle",
            font_size=18,
        )
    )
    for tick in (0.60, 0.62, 0.64, 0.66, 0.68, 0.70):
        yy = y(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{yy}" x2="{left + panel_w}" y2="{yy}"/>',
                text(left - 10, yy + 5, f"{tick:.2f}", text_anchor="end", font_size=13),
            ]
        )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + panel_h}" x2="{left + panel_w}" y2="{top + panel_h}"/>',
        ]
    )

    arms = ("A0", "A1", "A2")
    labels = ("D0", "A1", "A2")
    subtitles = ("sem augmentation", "geométrico", "geométrico + fotométrico")
    colors = ("#777777", "#2374AB", "#E07A1F")
    centers = (left + 105, left + 260, left + 415)
    bar_width = 46
    for arm, label, subtitle, color, center in zip(arms, labels, subtitles, colors, centers):
        for offset, source, opacity in ((-25, selected, 1.0), (25, fixed, 0.42)):
            value = source["arms"][arm]["dice"]
            yy = y(value)
            parts.append(
                f'<rect x="{center + offset - bar_width / 2}" y="{yy}" width="{bar_width}" '
                f'height="{top + panel_h - yy}" fill="{color}" fill-opacity="{opacity}" stroke="{color}"/>'
            )
            if offset < 0:
                parts.append(
                    text(
                        center + offset,
                        yy - 8,
                        f"{value:.3f}",
                        text_anchor="middle",
                        font_size=12,
                        font_weight="bold",
                    )
                )
        parts.extend(
            [
                text(
                    center,
                    top + panel_h + 28,
                    label,
                    text_anchor="middle",
                    font_size=15,
                    font_weight="bold",
                ),
                text(center, top + panel_h + 48, subtitle, text_anchor="middle", font_size=11),
            ]
        )
    parts.extend(
        [
            '<rect x="100" y="555" width="16" height="16" fill="#2374AB"/>',
            text(123, 568, "threshold escolhido na validação", font_size=12),
            '<rect x="340" y="555" width="16" height="16" fill="#2374AB" fill-opacity="0.42" stroke="#2374AB"/>',
            text(363, 568, "threshold 0,5", font_size=12),
        ]
    )

    right, right_top, right_w = 740, 130, 500
    x_min, x_max = -0.012, 0.060
    x = lambda value: right + (value - x_min) / (x_max - x_min) * right_w
    parts.append(
        text(
            right + right_w / 2,
            72,
            "Ganho sobre D0 — IC 95% bootstrap pareado",
            text_anchor="middle",
            font_size=18,
        )
    )
    for tick in (-0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        xx = x(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{xx}" y1="{right_top}" x2="{xx}" y2="430"/>',
                text(xx, 452, f"{tick:+.2f}", text_anchor="middle", font_size=12),
            ]
        )
    parts.append(
        f'<line x1="{x(0)}" y1="{right_top}" x2="{x(0)}" y2="430" stroke="#222" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    for row_y, name, label, color in (
        (220, "A1-A0", "A1 − D0", colors[1]),
        (345, "A2-A0", "A2 − D0", colors[2]),
    ):
        item = contrasts[name]
        low, mean, high = item["ci95_low"], item["mean_difference"], item["ci95_high"]
        parts.extend(
            [
                text(
                    right - 20,
                    row_y + 6,
                    label,
                    text_anchor="end",
                    font_size=16,
                    font_weight="bold",
                ),
                f'<line x1="{x(low)}" y1="{row_y}" x2="{x(high)}" y2="{row_y}" stroke="{color}" stroke-width="6"/>',
                f'<line x1="{x(low)}" y1="{row_y - 10}" x2="{x(low)}" y2="{row_y + 10}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{x(high)}" y1="{row_y - 10}" x2="{x(high)}" y2="{row_y + 10}" stroke="{color}" stroke-width="3"/>',
                f'<circle cx="{x(mean)}" cy="{row_y}" r="8" fill="{color}" stroke="white" stroke-width="2"/>',
                text(x(high) + 10, row_y + 5, f"{mean:+.3f}", font_size=14, font_weight="bold"),
            ]
        )
    parts.extend(
        [
            text(
                right + right_w / 2,
                488,
                "Diferença média de Dice macro por imagem",
                text_anchor="middle",
                font_size=14,
            ),
            text(
                right + right_w / 2,
                535,
                "A1: IC acima de zero; A2: IC cruza zero",
                text_anchor="middle",
                font_size=13,
                font_weight="bold",
            ),
            text(
                right + right_w / 2,
                558,
                "Triagem em uma seed; confirmação multisseed será posterior.",
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
