#!/usr/bin/env python3
"""Analisa e visualiza OR em duplicatas DFUC usando somente treino/validação."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "q1": float(np.quantile(array, 0.25)),
        "q3": float(np.quantile(array, 0.75)),
    }


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    result = image.copy()
    result[mask] = (0.55 * result[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(result)


def create_panel(root: Path, rows: list[dict], path: Path) -> None:
    columns = 6
    tile_width, tile_height, caption_height = 280, 210, 54
    sheet = Image.new(
        "RGB", (columns * tile_width, len(rows) * (tile_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/segoeui.ttf")
    font = (
        ImageFont.truetype(str(font_path), 15)
        if font_path.exists()
        else ImageFont.load_default(size=14)
    )
    headings = ("Imagem", "Máscara A", "Máscara B", "Interseção", "Discordância", "OR final")
    for row_index, row in enumerate(rows):
        image = np.asarray(Image.open(root / row["image_path"]).convert("RGB"))
        mask_paths = [root / item for item in row["mask_paths"].split("|")]
        from tcc_segmentation.data.duplicate_union import load_binary_mask

        mask_a, mask_b = (load_binary_mask(item) for item in mask_paths)
        intersection = mask_a & mask_b
        union = mask_a | mask_b
        disagreement = np.zeros_like(image)
        disagreement[mask_a & ~mask_b] = (255, 0, 255)
        disagreement[mask_b & ~mask_a] = (0, 255, 255)
        displays = (
            Image.fromarray(image),
            overlay(image, mask_a, (255, 0, 0)),
            overlay(image, mask_b, (0, 90, 255)),
            overlay(image, intersection, (0, 200, 70)),
            overlay(image, mask_a & ~mask_b, (255, 0, 255)),
            overlay(image, union, (255, 190, 0)),
        )
        # Acrescenta em ciano a região exclusiva de B na coluna de discordância.
        displays = list(displays)
        disagreement_overlay = np.asarray(displays[4]).copy()
        only_b = mask_b & ~mask_a
        disagreement_overlay[only_b] = (
            0.55 * image[only_b] + 0.45 * np.asarray([0, 255, 255])
        ).astype(np.uint8)
        displays[4] = Image.fromarray(disagreement_overlay)
        y = row_index * (tile_height + caption_height)
        for column, display in enumerate(displays):
            tile = display.resize((tile_width, tile_height), Image.Resampling.BILINEAR)
            x = column * tile_width
            sheet.paste(tile, (x, y))
            draw.text((x + 6, y + tile_height + 5), headings[column], fill="#172554", font=font)
        draw.text(
            (tile_width + 6, y + tile_height + 28),
            f"{row['canonical_id']} | Dice A×B={row['dice_between_references']:.3f}",
            fill="#475569",
            font=font,
        )
        draw.text(
            (4 * tile_width + 6, y + tile_height + 28),
            f"OR vs maior: +{100 * row['or_added_over_larger_fraction']:.2f}%",
            fill="#7f1d1d",
            font=font,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/data_preparation/or_union_analysis_train_validation"),
    )
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    sys.path.insert(0, str(root / "src"))
    from tcc_segmentation.data.duplicate_union import analyze_mask_pair, load_binary_mask

    # Lê exclusivamente manifestos permitidos. O manifesto de teste não é aberto.
    manifest_rows = []
    for split in ("train", "validation"):
        rows = read_csv(root / f"manifests/splits/dfuc_{split}.csv")
        if any(row["split"] != split for row in rows):
            raise RuntimeError(f"Manifesto {split} contém outra partição")
        manifest_rows.extend(rows)

    results = []
    for row in manifest_rows:
        mask_paths = row["mask_paths"].split("|")
        if len(mask_paths) != 2:
            continue
        mask_a = load_binary_mask(root / mask_paths[0])
        mask_b = load_binary_mask(root / mask_paths[1])
        results.append(
            {
                "canonical_id": row["canonical_id"],
                "split": row["split"],
                "source_stems": row["source_stems"],
                "image_path": row["image_path"],
                "mask_paths": row["mask_paths"],
                **analyze_mask_pair(mask_a, mask_b),
            }
        )
    if not results:
        raise RuntimeError("Nenhum grupo duplicado acessível foi encontrado")

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "or_union_per_group.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    ordered = sorted(results, key=lambda row: row["dice_between_references"])
    representatives = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    panel_path = output / "or_union_examples.png"
    create_panel(root, representatives, panel_path)
    report = {
        "status": "completed",
        "purpose": "didactic_audit_of_duplicate_mask_OR",
        "scientific_model_result": False,
        "dataset": "DFUC2022",
        "partitions_accessed": ["train", "validation"],
        "protected_test_accessed": False,
        "groups_analyzed": len(results),
        "known_dfuc_duplicate_groups_total_from_pre_split_audit": 39,
        "groups_not_reopened_because_assigned_to_test": 39 - len(results),
        "dice_between_references": summarize([row["dice_between_references"] for row in results]),
        "or_added_over_larger_fraction": summarize(
            [row["or_added_over_larger_fraction"] for row in results]
        ),
        "or_added_over_mean_fraction": summarize(
            [row["or_added_over_mean_fraction"] for row in results]
        ),
        "representative_cases": [row["canonical_id"] for row in representatives],
        "interpretation": (
            "OR inclui qualquer pixel marcado por pelo menos uma referência; reduz "
            "omissão entre anotações, mas pode ampliar a máscara canônica."
        ),
    }
    json_path = output / "or_union_summary.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(panel_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
