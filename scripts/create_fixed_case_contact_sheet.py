#!/usr/bin/env python3
"""Cria contato visual dos casos de validação fixados antes das comparações."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--cases", type=Path, default=Path("configs/visualization_cases_v1.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/auditoria_metodologica/fixed_validation_cases.png"),
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    cases_path = args.cases if args.cases.is_absolute() else root / args.cases
    with cases_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = (
        ImageFont.truetype(str(font_path), 16)
        if font_path.exists()
        else ImageFont.load_default(size=15)
    )
    tile_width, image_height, caption_height = 320, 240, 55
    columns = 4
    lines = (len(rows) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, lines * (image_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        image = Image.open(root / row["image_path"]).convert("RGB")
        mask_arrays = []
        for relative_path in row["mask_paths"].split("|"):
            mask_arrays.append(np.asarray(Image.open(root / relative_path).convert("L")) >= 128)
        mask = np.logical_or.reduce(mask_arrays)
        overlay = np.asarray(image).copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray([255, 0, 0])).astype(np.uint8)
        tile = Image.fromarray(overlay).resize(
            (tile_width, image_height), Image.Resampling.BILINEAR
        )
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + caption_height)
        sheet.paste(tile, (x, y))
        draw.text(
            (x + 7, y + image_height + 5),
            f"{row['canonical_id']} | {row['selection_role']}",
            fill="#172554",
            font=font,
        )
        draw.text(
            (x + 7, y + image_height + 28),
            f"área={float(row['foreground_fraction']):.3%} | L={float(row['mean_luminance']):.1f}",
            fill="#475569",
            font=font,
        )

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
