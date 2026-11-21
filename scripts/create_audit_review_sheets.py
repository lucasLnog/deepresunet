#!/usr/bin/env python3
"""Gera folhas visuais para revisar duplicatas exatas e mascaras vazias."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("results/audit/2026-08-25"),
        help="Diretorio que contem os CSVs gerados pela auditoria.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretorio das folhas; por padrao, <audit-dir>/review_sheets.",
    )
    parser.add_argument("--groups-per-page", type=int, default=6)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def overlay_mask(image_path: str, mask_path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image_source:
        image = image_source.convert("RGB")
    with Image.open(mask_path) as mask_source:
        mask = mask_source.convert("L")

    image = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    mask = ImageOps.contain(mask, size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.putalpha(mask.point(lambda value: 115 if value > 127 else 0))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def plain_image(image_path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image_source:
        image = image_source.convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def duplicate_groups(rows: Sequence[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(sorted(groups.items()))


def create_duplicate_pages(
    rows: Sequence[dict[str, str]], output_dir: Path, groups_per_page: int
) -> list[Path]:
    groups = duplicate_groups(rows)
    output_paths: list[Path] = []
    thumb_size = (220, 170)
    cell_width = 240
    row_height = 225

    for dataset in sorted({row["dataset"] for row in rows}):
        selected_groups = [
            (group_id, members)
            for group_id, members in groups.items()
            if members[0]["dataset"] == dataset
        ]
        max_members = max(len(members) for _, members in selected_groups)
        columns = 1 + max_members

        for page_index in range(math.ceil(len(selected_groups) / groups_per_page)):
            page_groups = selected_groups[
                page_index * groups_per_page : (page_index + 1) * groups_per_page
            ]
            canvas = Image.new(
                "RGB", (columns * cell_width, len(page_groups) * row_height + 35), "white"
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 10),
                f"{dataset} - duplicatas exatas - pagina {page_index + 1}",
                fill="black",
            )

            for local_row, (group_id, members) in enumerate(page_groups):
                y = 35 + local_row * row_height
                original = plain_image(members[0]["image_path"], thumb_size)
                canvas.paste(original, (10, y + 20))
                draw.text((10, y), f"{group_id} original", fill="black")

                for member_index, member in enumerate(members, start=1):
                    overlay = overlay_mask(member["image_path"], member["mask_path"], thumb_size)
                    x = member_index * cell_width + 10
                    canvas.paste(overlay, (x, y + 20))
                    draw.text((x, y), f"mask {member['stem']}", fill="black")

            output_path = output_dir / f"duplicates_{dataset}_page_{page_index + 1:02d}.png"
            canvas.save(output_path)
            output_paths.append(output_path)

    return output_paths


def create_empty_mask_pages(inventory: Sequence[dict[str, str]], output_dir: Path) -> list[Path]:
    empty_rows = [row for row in inventory if row["mask_empty"] == "True"]
    if not empty_rows:
        return []

    columns, rows_per_page = 4, 4
    items_per_page = columns * rows_per_page
    cell_width, cell_height = 270, 245
    thumb_size = (250, 190)
    paths: list[Path] = []

    for page_index in range(math.ceil(len(empty_rows) / items_per_page)):
        page_rows = empty_rows[page_index * items_per_page : (page_index + 1) * items_per_page]
        canvas = Image.new("RGB", (columns * cell_width, rows_per_page * cell_height + 35), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Mascaras vazias - pagina {page_index + 1}", fill="black")

        for index, record in enumerate(page_rows):
            column = index % columns
            row_index = index // columns
            x = column * cell_width + 10
            y = row_index * cell_height + 45
            image = plain_image(record["image_path"], thumb_size)
            canvas.paste(image, (x, y))
            draw.text((x, y + 195), f"{record['dataset']} {record['stem']}", fill="black")

        output_path = output_dir / f"empty_masks_page_{page_index + 1:02d}.png"
        canvas.save(output_path)
        paths.append(output_path)

    return paths


def main() -> int:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else audit_dir / "review_sheets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    duplicate_rows = read_csv(audit_dir / "exact_duplicate_groups.csv")
    inventory = read_csv(audit_dir / "dataset_inventory.csv")
    duplicate_pages = create_duplicate_pages(duplicate_rows, output_dir, args.groups_per_page)
    empty_pages = create_empty_mask_pages(inventory, output_dir)

    print(f"Folhas de duplicatas: {len(duplicate_pages)}")
    print(f"Folhas de mascaras vazias: {len(empty_pages)}")
    print(f"Saida: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
