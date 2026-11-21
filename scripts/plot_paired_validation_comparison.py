"""Gera um painel SVG da comparação pareada de Dice, sem dependências externas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def dice_rows(path: Path) -> dict[str, dict[str, float | int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["metric"] == "dice"]
    return {
        row["comparison"]: {
            "mean": float(row["mean_difference"]),
            "low": float(row["ci95_low"]),
            "high": float(row["ci95_high"]),
            "improved": int(row["improved"]),
            "equal": int(row["equal"]),
            "worsened": int(row["worsened"]),
        }
        for row in rows
    }


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    args = parse_args()
    selected = dice_rows(args.selected)
    fixed = dice_rows(args.fixed)
    contrasts = ("P1-D0", "P2-D0")
    labels = {"P1-D0": "P1 − D0", "P2-D0": "P2 − D0"}
    width, height = 1500, 650
    left_x0, left_x1 = 170, 720
    min_effect, max_effect = -0.035, 0.03

    def effect_x(value: float) -> float:
        return left_x0 + (value - min_effect) * (left_x1 - left_x0) / (max_effect - min_effect)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:25px;font-weight:700}.subtitle{font-size:20px;font-weight:700}.axis{font-size:15px}.label{font-size:17px}.small{font-size:14px}</style>",
        '<text x="750" y="38" text-anchor="middle" class="title">Comparação pareada D0 × P1 × P2 — validação congelada</text>',
        '<text x="445" y="82" text-anchor="middle" class="subtitle">Diferença média de Dice e IC 95%</text>',
        '<text x="1135" y="82" text-anchor="middle" class="subtitle">Resultado por imagem — threshold selecionado</text>',
    ]

    zero_x = effect_x(0.0)
    svg.append(
        f'<line x1="{zero_x:.1f}" y1="105" x2="{zero_x:.1f}" y2="500" stroke="#222" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    for tick in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
        x = effect_x(tick)
        svg.extend(
            [
                f'<line x1="{x:.1f}" y1="500" x2="{x:.1f}" y2="507" stroke="#333"/>',
                f'<text x="{x:.1f}" y="529" text-anchor="middle" class="axis">{tick:+.2f}</text>',
            ]
        )
    svg.append(
        '<text x="445" y="570" text-anchor="middle" class="label">Diferença média pareada de Dice</text>'
    )

    color_sets = (
        (selected, "#1769aa", -13, "Threshold selecionado"),
        (fixed, "#d95f02", 13, "Threshold 0,5"),
    )
    for row_index, contrast in enumerate(contrasts):
        y = 220 + row_index * 190
        svg.append(
            f'<text x="150" y="{y + 6}" text-anchor="end" class="label">{labels[contrast]}</text>'
        )
        for data, color, offset, _ in color_sets:
            item = data[contrast]
            cy = y + offset
            x_low, x_high, x_mean = (
                effect_x(float(item["low"])),
                effect_x(float(item["high"])),
                effect_x(float(item["mean"])),
            )
            svg.extend(
                [
                    f'<line x1="{x_low:.1f}" y1="{cy}" x2="{x_high:.1f}" y2="{cy}" stroke="{color}" stroke-width="4"/>',
                    f'<line x1="{x_low:.1f}" y1="{cy-8}" x2="{x_low:.1f}" y2="{cy+8}" stroke="{color}" stroke-width="3"/>',
                    f'<line x1="{x_high:.1f}" y1="{cy-8}" x2="{x_high:.1f}" y2="{cy+8}" stroke="{color}" stroke-width="3"/>',
                    f'<circle cx="{x_mean:.1f}" cy="{cy}" r="7" fill="{color}"/>',
                ]
            )
    for index, (_, color, _, label) in enumerate(color_sets):
        x = 245 + index * 240
        svg.extend(
            [
                f'<circle cx="{x}" cy="615" r="7" fill="{color}"/>',
                f'<text x="{x+14}" y="620" class="small">{esc(label)}</text>',
            ]
        )

    bar_width = 150
    scale = 330 / 294
    stack_colors = (
        ("improved", "#2ca25f", "Melhorou"),
        ("equal", "#bdbdbd", "Empatou"),
        ("worsened", "#de2d26", "Piorou"),
    )
    for index, contrast in enumerate(contrasts):
        x = 965 + index * 260
        bottom = 500.0
        for key, color, _ in stack_colors:
            value = int(selected[contrast][key])
            segment_height = value * scale
            bottom -= segment_height
            svg.append(
                f'<rect x="{x}" y="{bottom:.1f}" width="{bar_width}" height="{segment_height:.1f}" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{x+bar_width/2}" y="{bottom+segment_height/2+5:.1f}" text-anchor="middle" class="label" fill="white">{value}</text>'
            )
        svg.append(
            f'<text x="{x+bar_width/2}" y="530" text-anchor="middle" class="label">{labels[contrast]}</text>'
        )
    for index, (_, color, label) in enumerate(stack_colors):
        x = 880 + index * 165
        svg.extend(
            [
                f'<rect x="{x}" y="590" width="18" height="18" fill="{color}"/>',
                f'<text x="{x+25}" y="604" class="small">{label}</text>',
            ]
        )
    svg.append(
        '<text x="850" y="300" transform="rotate(-90 850 300)" text-anchor="middle" class="label">Número de imagens (n=294)</text>'
    )
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
