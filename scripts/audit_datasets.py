#!/usr/bin/env python3
"""Audita os datasets DFUC2022 e FUSeg sem modificar os dados brutos.

Saidas principais:
    - dataset_inventory.csv
    - unpaired_files.csv
    - exact_duplicate_groups.csv
    - perceptual_duplicate_pairs.csv
    - duplicate_mask_consistency.csv
    - resolution_counts.csv
    - summary.json
    - AUDITORIA_DATASETS.md
    - foreground_fraction_histogram.png
    - sample_montage.png

O script usa apenas NumPy e Pillow para facilitar a execucao local e no Google
Colab. Duplicatas perceptuais sao candidatas para revisao, nao exclusoes
automaticas. Nenhum split e criado por este programa.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    images_dir: str
    masks_dir: str


DATASETS = (
    DatasetSpec(
        name="DFUC2022",
        images_dir="DFUC2022_train_images",
        masks_dir="DFUC2022_train_masks",
    ),
    DatasetSpec(
        name="FUSeg",
        images_dir="FUSeg_Images",
        masks_dir="FUSeg_Masks",
    ),
)


INVENTORY_FIELDS = (
    "dataset",
    "stem",
    "image_path",
    "mask_path",
    "status",
    "error",
    "image_width",
    "image_height",
    "image_mode_original",
    "mask_width",
    "mask_height",
    "mask_mode_original",
    "dimensions_match",
    "mask_channels_equal",
    "mask_unique_count",
    "mask_unique_values_preview",
    "mask_native_binary",
    "mask_empty",
    "mask_full",
    "foreground_pixels",
    "total_pixels",
    "foreground_fraction",
    "connected_components_8",
    "image_mean_r",
    "image_mean_g",
    "image_mean_b",
    "image_mean_luminance",
    "image_raw_sha256",
    "image_pixel_sha256",
    "mask_raw_sha256",
    "mask_binary_sha256",
    "dhash64",
    "phash64",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria reprodutivel de DFUC2022 e FUSeg.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset"),
        help="Diretorio que contem as quatro pastas dos datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/audit/latest"),
        help="Diretorio de saida da auditoria.",
    )
    parser.add_argument(
        "--phash-distance",
        type=int,
        default=4,
        help="Distancia Hamming maxima de pHash para candidato perceptual.",
    )
    parser.add_argument(
        "--dhash-distance",
        type=int,
        default=6,
        help="Distancia Hamming maxima de dHash para candidato perceptual.",
    )
    parser.add_argument(
        "--skip-perceptual",
        action="store_true",
        help="Pula a busca O(n^2) de candidatos perceptuais.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, prefix: str = "") -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.astype(bool).ravel():
        value = (value << 1) | int(bit)
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def dhash64(image_rgb: Image.Image) -> str:
    gray = image_rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    return bits_to_hex(pixels[:, 1:] > pixels[:, :-1])


def dct_matrix(size: int = 32) -> np.ndarray:
    x = np.arange(size, dtype=np.float64)
    k = x[:, None]
    matrix = np.cos((math.pi / size) * (x + 0.5) * k)
    matrix[0, :] *= math.sqrt(1.0 / size)
    matrix[1:, :] *= math.sqrt(2.0 / size)
    return matrix


DCT_32 = dct_matrix(32)


def phash64(image_rgb: Image.Image) -> str:
    gray = image_rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    coefficients = DCT_32 @ pixels @ DCT_32.T
    low_frequency = coefficients[:8, :8]
    median = float(np.median(low_frequency.ravel()[1:]))
    return bits_to_hex(low_frequency > median)


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Diretorio nao encontrado: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def index_by_stem(paths: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem].append(path)
    return dict(index)


def mask_to_gray_array(mask: Image.Image) -> tuple[np.ndarray, bool | None]:
    raw = np.asarray(mask)
    channels_equal: bool | None = None

    if raw.ndim == 2:
        gray = raw
    elif raw.ndim == 3:
        rgb = raw[..., :3]
        channels_equal = bool(
            np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])
        )
        gray = np.asarray(mask.convert("L"))
    else:
        raise ValueError(f"Mascara com shape nao suportado: {raw.shape}")

    return np.asarray(gray), channels_equal


def count_components_8(binary: np.ndarray) -> int:
    """Conta componentes 8-conectados usando rotulos de segmentos por linha."""

    parent: list[int] = []

    def make_label() -> int:
        label = len(parent)
        parent.append(label)
        return label

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    previous: list[tuple[int, int, int]] = []

    for row in np.asarray(binary, dtype=np.uint8):
        padded = np.pad(row, (1, 1), mode="constant", constant_values=0)
        transitions = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        current: list[tuple[int, int, int]] = []

        for start, end in zip(starts.tolist(), ends.tolist()):
            overlaps = [
                label
                for prev_start, prev_end, label in previous
                if prev_end >= start - 1 and prev_start <= end + 1
            ]
            if overlaps:
                label = overlaps[0]
                for other in overlaps[1:]:
                    union(label, other)
            else:
                label = make_label()
            current.append((start, end, label))

        previous = current

    return len({find(label) for label in range(len(parent))})


def unique_preview(values: np.ndarray, limit: int = 16) -> str:
    if values.size <= limit:
        return ";".join(str(int(value)) for value in values)
    head = ";".join(str(int(value)) for value in values[: limit // 2])
    tail = ";".join(str(int(value)) for value in values[-limit // 2 :])
    return f"{head};...;{tail}"


def audit_pair(dataset: str, stem: str, image_path: Path, mask_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in INVENTORY_FIELDS}
    row.update(
        {
            "dataset": dataset,
            "stem": stem,
            "image_path": image_path.as_posix(),
            "mask_path": mask_path.as_posix(),
            "status": "ok",
        }
    )

    try:
        image_raw_sha = sha256_file(image_path)
        mask_raw_sha = sha256_file(mask_path)

        with Image.open(image_path) as image_source:
            image_source.load()
            image_mode = image_source.mode
            image_rgb = image_source.convert("RGB")

        with Image.open(mask_path) as mask_source:
            mask_source.load()
            mask_mode = mask_source.mode
            mask_size = mask_source.size
            mask_gray, mask_channels_equal = mask_to_gray_array(mask_source)

        image_array = np.asarray(image_rgb, dtype=np.uint8)
        unique_values = np.unique(mask_gray)
        binary = mask_gray > 127
        foreground_pixels = int(binary.sum())
        total_pixels = int(binary.size)
        mean_rgb = image_array.reshape(-1, 3).mean(axis=0)
        mean_luminance = float(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2])

        native_binary_values = {0, 1, 255}
        row.update(
            {
                "image_width": image_rgb.width,
                "image_height": image_rgb.height,
                "image_mode_original": image_mode,
                "mask_width": mask_size[0],
                "mask_height": mask_size[1],
                "mask_mode_original": mask_mode,
                "dimensions_match": image_rgb.size == mask_size,
                "mask_channels_equal": mask_channels_equal,
                "mask_unique_count": int(unique_values.size),
                "mask_unique_values_preview": unique_preview(unique_values),
                "mask_native_binary": set(map(int, unique_values)).issubset(native_binary_values),
                "mask_empty": foreground_pixels == 0,
                "mask_full": foreground_pixels == total_pixels,
                "foreground_pixels": foreground_pixels,
                "total_pixels": total_pixels,
                "foreground_fraction": foreground_pixels / total_pixels,
                "connected_components_8": count_components_8(binary),
                "image_mean_r": float(mean_rgb[0]),
                "image_mean_g": float(mean_rgb[1]),
                "image_mean_b": float(mean_rgb[2]),
                "image_mean_luminance": mean_luminance,
                "image_raw_sha256": image_raw_sha,
                "image_pixel_sha256": sha256_array(image_array, prefix="RGB"),
                "mask_raw_sha256": mask_raw_sha,
                "mask_binary_sha256": sha256_array(binary.astype(np.uint8), prefix="MASK"),
                "dhash64": dhash64(image_rgb),
                "phash64": phash64(image_rgb),
            }
        )
    except (OSError, ValueError, UnidentifiedImageError) as error:
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"

    return row


def collect_pairs(
    dataset_root: Path, spec: DatasetSpec
) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    image_index = index_by_stem(list_images(dataset_root / spec.images_dir))
    mask_index = index_by_stem(list_images(dataset_root / spec.masks_dir))
    all_stems = sorted(set(image_index) | set(mask_index))
    pairs: list[tuple[str, Path, Path]] = []
    unpaired: list[dict[str, str]] = []

    for stem in all_stems:
        images = image_index.get(stem, [])
        masks = mask_index.get(stem, [])
        if len(images) == 1 and len(masks) == 1:
            pairs.append((stem, images[0], masks[0]))
            continue

        if not images:
            issue = "mask_without_image"
        elif not masks:
            issue = "image_without_mask"
        else:
            issue = "ambiguous_duplicate_stem"
        unpaired.append(
            {
                "dataset": spec.name,
                "stem": stem,
                "issue": issue,
                "image_paths": "|".join(path.as_posix() for path in images),
                "mask_paths": "|".join(path.as_posix() for path in masks),
            }
        )

    return pairs, unpaired


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def exact_duplicate_groups(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[str(row["image_pixel_sha256"])].append(row)

    output: list[dict[str, Any]] = []
    group_number = 0
    for pixel_hash, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        group_number += 1
        group_id = f"EXACT-{group_number:04d}"
        datasets = sorted({str(member["dataset"]) for member in members})
        for member in members:
            output.append(
                {
                    "group_id": group_id,
                    "group_size": len(members),
                    "scope": "cross_dataset" if len(datasets) > 1 else "within_dataset",
                    "datasets": "|".join(datasets),
                    "dataset": member["dataset"],
                    "stem": member["stem"],
                    "image_path": member["image_path"],
                    "mask_path": member["mask_path"],
                    "image_pixel_sha256": pixel_hash,
                    "mask_binary_sha256": member["mask_binary_sha256"],
                }
            )
    return output


def perceptual_pairs(
    rows: Sequence[dict[str, Any]], phash_limit: int, dhash_limit: int
) -> list[dict[str, Any]]:
    valid = [row for row in rows if row["status"] == "ok"]
    candidates: list[dict[str, Any]] = []

    for left, right in itertools.combinations(valid, 2):
        if left["image_pixel_sha256"] == right["image_pixel_sha256"]:
            continue
        phash_distance = hamming_hex(str(left["phash64"]), str(right["phash64"]))
        if phash_distance > phash_limit:
            continue
        dhash_distance = hamming_hex(str(left["dhash64"]), str(right["dhash64"]))
        if dhash_distance > dhash_limit:
            continue
        candidates.append(
            {
                "scope": (
                    "cross_dataset" if left["dataset"] != right["dataset"] else "within_dataset"
                ),
                "dataset_a": left["dataset"],
                "stem_a": left["stem"],
                "image_path_a": left["image_path"],
                "dataset_b": right["dataset"],
                "stem_b": right["stem"],
                "image_path_b": right["image_path"],
                "phash_distance": phash_distance,
                "dhash_distance": dhash_distance,
                "requires_visual_review": True,
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            int(row["phash_distance"]),
            int(row["dhash_distance"]),
            str(row["image_path_a"]),
            str(row["image_path_b"]),
        ),
    )


def binary_mask_from_path(path: str) -> np.ndarray:
    with Image.open(path) as mask:
        mask.load()
        gray, _ = mask_to_gray_array(mask)
    return gray > 127


def mask_consistency(exact_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        groups[str(row["group_id"])].append(row)

    output: list[dict[str, Any]] = []
    for group_id, members in groups.items():
        for left, right in itertools.combinations(members, 2):
            left_mask = binary_mask_from_path(str(left["mask_path"]))
            right_mask = binary_mask_from_path(str(right["mask_path"]))
            same_shape = left_mask.shape == right_mask.shape
            if same_shape:
                intersection = int(np.logical_and(left_mask, right_mask).sum())
                left_area = int(left_mask.sum())
                right_area = int(right_mask.sum())
                union = int(np.logical_or(left_mask, right_mask).sum())
                denominator = left_area + right_area
                dice = 1.0 if denominator == 0 else (2.0 * intersection / denominator)
                iou = 1.0 if union == 0 else (intersection / union)
            else:
                dice = ""
                iou = ""
            output.append(
                {
                    "group_id": group_id,
                    "dataset_a": left["dataset"],
                    "stem_a": left["stem"],
                    "mask_path_a": left["mask_path"],
                    "dataset_b": right["dataset"],
                    "stem_b": right["stem"],
                    "mask_path_b": right["mask_path"],
                    "same_shape": same_shape,
                    "same_binary_mask": left["mask_binary_sha256"] == right["mask_binary_sha256"],
                    "mask_dice": dice,
                    "mask_iou": iou,
                }
            )
    return output


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def dataset_summary(
    name: str,
    rows: Sequence[dict[str, Any]],
    unpaired: Sequence[dict[str, str]],
) -> dict[str, Any]:
    subset = [row for row in rows if row["dataset"] == name]
    valid = [row for row in subset if row["status"] == "ok"]
    fractions = [float(row["foreground_fraction"]) for row in valid]
    components = [int(row["connected_components_8"]) for row in valid]
    resolutions = Counter(f"{row['image_width']}x{row['image_height']}" for row in valid)

    return {
        "paired_records": len(subset),
        "valid_records": len(valid),
        "invalid_records": len(subset) - len(valid),
        "unpaired_or_ambiguous": sum(1 for issue in unpaired if issue["dataset"] == name),
        "dimension_mismatches": sum(row["dimensions_match"] is False for row in valid),
        "non_native_binary_masks": sum(row["mask_native_binary"] is False for row in valid),
        "empty_masks": sum(row["mask_empty"] is True for row in valid),
        "full_masks": sum(row["mask_full"] is True for row in valid),
        "foreground_fraction": {
            "min": min(fractions) if fractions else None,
            "q1": percentile(fractions, 0.25),
            "median": percentile(fractions, 0.50),
            "q3": percentile(fractions, 0.75),
            "max": max(fractions) if fractions else None,
            "mean": statistics.fmean(fractions) if fractions else None,
        },
        "connected_components_8": {
            "min": min(components) if components else None,
            "median": percentile([float(value) for value in components], 0.50),
            "max": max(components) if components else None,
            "mean": statistics.fmean(components) if components else None,
        },
        "resolution_counts": dict(sorted(resolutions.items())),
    }


def resolution_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter(
        (str(row["dataset"]), int(row["image_width"]), int(row["image_height"]))
        for row in rows
        if row["status"] == "ok"
    )
    return [
        {"dataset": dataset, "width": width, "height": height, "count": count}
        for (dataset, width, height), count in sorted(counter.items())
    ]


def create_histogram(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    width, height = 1000, 560
    margin_left, margin_right = 85, 30
    margin_top, margin_bottom = 55, 80
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    bins = np.linspace(0.0, 1.0, 41)
    colors = {"DFUC2022": (35, 105, 180), "FUSeg": (220, 95, 55)}
    histograms: dict[str, np.ndarray] = {}

    for dataset in colors:
        values = [
            float(row["foreground_fraction"])
            for row in rows
            if row["status"] == "ok" and row["dataset"] == dataset
        ]
        histograms[dataset] = np.histogram(values, bins=bins)[0]

    max_count = max(int(hist.max()) for hist in histograms.values()) or 1
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_left, 18), "Distribuicao da proporcao de pixels de ferida", fill="black")
    draw.line(
        (margin_left, margin_top, margin_left, margin_top + chart_height), fill="black", width=2
    )
    draw.line(
        (
            margin_left,
            margin_top + chart_height,
            margin_left + chart_width,
            margin_top + chart_height,
        ),
        fill="black",
        width=2,
    )

    bin_width_pixels = chart_width / (len(bins) - 1)
    for dataset_index, (dataset, hist) in enumerate(histograms.items()):
        color = colors[dataset]
        offset = dataset_index * (bin_width_pixels / 2)
        bar_width = max(1, int(bin_width_pixels / 2) - 1)
        for index, count in enumerate(hist.tolist()):
            bar_height = (count / max_count) * chart_height
            x0 = margin_left + index * bin_width_pixels + offset
            y0 = margin_top + chart_height - bar_height
            draw.rectangle((x0, y0, x0 + bar_width, margin_top + chart_height), fill=color)

    for tick in range(0, 11):
        x = margin_left + chart_width * (tick / 10)
        draw.line((x, margin_top + chart_height, x, margin_top + chart_height + 5), fill="black")
        draw.text((x - 10, margin_top + chart_height + 10), f"{tick / 10:.1f}", fill="black")

    draw.text(
        (margin_left + chart_width / 2 - 80, height - 30), "Fracao de foreground", fill="black"
    )
    legend_x = width - 220
    for index, (dataset, color) in enumerate(colors.items()):
        y = 20 + index * 22
        draw.rectangle((legend_x, y, legend_x + 15, y + 12), fill=color)
        draw.text((legend_x + 22, y), dataset, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def select_quantile_rows(rows: Sequence[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    subset = sorted(
        (row for row in rows if row["status"] == "ok" and row["dataset"] == dataset),
        key=lambda row: float(row["foreground_fraction"]),
    )
    if not subset:
        return []
    indexes = [round((len(subset) - 1) * quantile) for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return [subset[index] for index in indexes]


def create_sample_montage(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    selected: list[dict[str, Any]] = []
    for dataset in ("DFUC2022", "FUSeg"):
        selected.extend(select_quantile_rows(rows, dataset))

    cell_width, cell_height = 260, 300
    columns = 5
    rows_count = max(1, math.ceil(len(selected) / columns))
    canvas = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)

    for index, record in enumerate(selected):
        column = index % columns
        row_index = index // columns
        x = column * cell_width
        y = row_index * cell_height

        with Image.open(str(record["image_path"])) as image_source:
            image = image_source.convert("RGB")
        with Image.open(str(record["mask_path"])) as mask_source:
            mask = mask_source.convert("L")

        fitted = ImageOps.contain(image, (240, 240), Image.Resampling.LANCZOS)
        fitted_mask = ImageOps.contain(mask, (240, 240), Image.Resampling.NEAREST)
        overlay = Image.new("RGBA", fitted.size, (255, 0, 0, 0))
        alpha = fitted_mask.point(lambda value: 110 if value > 127 else 0)
        overlay.putalpha(alpha)
        composite = Image.alpha_composite(fitted.convert("RGBA"), overlay).convert("RGB")
        canvas.paste(composite, (x + 10, y + 10))
        label = (
            f"{record['dataset']} {record['stem']}  "
            f"area={float(record['foreground_fraction']):.3f}"
        )
        draw.text((x + 10, y + 255), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.3f}%"


def create_markdown_report(
    output_path: Path,
    dataset_summaries: dict[str, dict[str, Any]],
    exact_rows: Sequence[dict[str, Any]],
    perceptual: Sequence[dict[str, Any]],
    mask_comparisons: Sequence[dict[str, Any]],
    thresholds: dict[str, int],
) -> None:
    exact_group_ids = {str(row["group_id"]) for row in exact_rows}
    exact_cross_ids = {
        str(row["group_id"]) for row in exact_rows if row["scope"] == "cross_dataset"
    }
    differing_masks = sum(
        comparison["same_binary_mask"] is False for comparison in mask_comparisons
    )
    perceptual_cross = sum(pair["scope"] == "cross_dataset" for pair in perceptual)

    lines = [
        "# Auditoria dos datasets DFUC2022 e FUSeg",
        "",
        f"**Gerado em:** {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## Escopo",
        "",
        "Auditoria somente leitura. Nenhum split foi criado e nenhuma imagem foi removida.",
        "Candidatos perceptuais exigem revisao visual antes de qualquer decisao.",
        "",
        "## Resumo por dataset",
        "",
        "| Dataset | Pares validos | Problemas de par | Erros de leitura | Dimensoes divergentes | Mascaras nao binarias | Mascaras vazias | Foreground mediano |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for dataset, summary in dataset_summaries.items():
        lines.append(
            "| {dataset} | {valid} | {unpaired} | {invalid} | {mismatch} | "
            "{non_binary} | {empty} | {median} |".format(
                dataset=dataset,
                valid=summary["valid_records"],
                unpaired=summary["unpaired_or_ambiguous"],
                invalid=summary["invalid_records"],
                mismatch=summary["dimension_mismatches"],
                non_binary=summary["non_native_binary_masks"],
                empty=summary["empty_masks"],
                median=format_percent(summary["foreground_fraction"]["median"]),
            )
        )

    lines.extend(
        [
            "",
            "## Duplicatas e sobreposicao",
            "",
            f"- grupos de imagens exatamente iguais por pixels decodificados: **{len(exact_group_ids)}**;",
            f"- grupos exatos que atravessam DFUC2022 e FUSeg: **{len(exact_cross_ids)}**;",
            f"- comparacoes de mascaras divergentes dentro de imagens exatas: **{differing_masks}**;",
            f"- pares perceptuais candidatos: **{len(perceptual)}**;",
            f"- candidatos perceptuais entre datasets: **{perceptual_cross}**;",
            f"- limites perceptuais: pHash <= {thresholds['phash_distance']} e dHash <= {thresholds['dhash_distance']}.",
            "",
            "## Interpretacao obrigatoria",
            "",
            "- Duplicatas exatas devem permanecer no mesmo grupo durante a divisao.",
            "- Pares perceptuais nao devem ser removidos automaticamente; precisam de revisao visual.",
            "- Sem IDs de paciente, nao e possivel garantir independencia por paciente.",
            "- Mascaras nao nativamente binarias serao binarizadas somente no pipeline posterior, com regra documentada.",
            "- O FUSeg continua reservado para avaliacao externa e nao participa da escolha de configuracoes.",
            "",
            "## Artefatos",
            "",
            "- `dataset_inventory.csv`: uma linha por par imagem-mascara;",
            "- `unpaired_files.csv`: arquivos ausentes ou stems ambiguos;",
            "- `exact_duplicate_groups.csv`: grupos exatos;",
            "- `perceptual_duplicate_pairs.csv`: candidatos por hashes perceptuais;",
            "- `duplicate_mask_consistency.csv`: Dice/IoU entre mascaras de imagens exatas;",
            "- `resolution_counts.csv`: distribuicao de resolucoes;",
            "- `summary.json`: resumo estruturado e ambiente;",
            "- `foreground_fraction_histogram.png`: distribuicao de area relativa;",
            "- `sample_montage.png`: amostras por quantis de area com overlay.",
            "",
            "## Proximo passo",
            "",
            "Revisar visualmente os candidatos perceptuais e aprovar as regras de agrupamento antes de gerar os manifestos de treino, validacao e teste.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")

    all_pairs: list[tuple[str, str, Path, Path]] = []
    unpaired: list[dict[str, str]] = []

    for spec in DATASETS:
        pairs, issues = collect_pairs(dataset_root, spec)
        all_pairs.extend((spec.name, stem, image, mask) for stem, image, mask in pairs)
        unpaired.extend(issues)
        print(f"{spec.name}: {len(pairs)} pares; {len(issues)} problemas de pareamento")

    inventory: list[dict[str, Any]] = []
    for index, (dataset, stem, image_path, mask_path) in enumerate(all_pairs, start=1):
        if index == 1 or index % 100 == 0 or index == len(all_pairs):
            print(f"Auditando {index}/{len(all_pairs)}...")
        inventory.append(audit_pair(dataset, stem, image_path, mask_path))

    exact_rows = exact_duplicate_groups(inventory)
    if args.skip_perceptual:
        perceptual: list[dict[str, Any]] = []
    else:
        print("Buscando candidatos perceptuais (comparacao par a par)...")
        perceptual = perceptual_pairs(
            inventory,
            phash_limit=args.phash_distance,
            dhash_limit=args.dhash_distance,
        )
    mask_comparisons = mask_consistency(exact_rows)

    summaries = {spec.name: dataset_summary(spec.name, inventory, unpaired) for spec in DATASETS}
    exact_group_count = len({row["group_id"] for row in exact_rows})
    exact_cross_group_count = len(
        {row["group_id"] for row in exact_rows if row["scope"] == "cross_dataset"}
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "script": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pillow_version": Image.__version__,
        "parameters": {
            "phash_distance": args.phash_distance,
            "dhash_distance": args.dhash_distance,
            "skip_perceptual": args.skip_perceptual,
            "mask_binarization_for_audit": "> 127",
            "component_connectivity": 8,
        },
        "datasets": summaries,
        "duplicates": {
            "exact_groups": exact_group_count,
            "exact_cross_dataset_groups": exact_cross_group_count,
            "perceptual_candidate_pairs": len(perceptual),
            "perceptual_cross_dataset_pairs": sum(
                pair["scope"] == "cross_dataset" for pair in perceptual
            ),
            "exact_image_mask_comparisons": len(mask_comparisons),
            "exact_image_pairs_with_different_binary_masks": sum(
                comparison["same_binary_mask"] is False for comparison in mask_comparisons
            ),
        },
    }

    write_csv(output_dir / "dataset_inventory.csv", inventory, INVENTORY_FIELDS)
    write_csv(
        output_dir / "unpaired_files.csv",
        unpaired,
        ("dataset", "stem", "issue", "image_paths", "mask_paths"),
    )
    write_csv(
        output_dir / "exact_duplicate_groups.csv",
        exact_rows,
        (
            "group_id",
            "group_size",
            "scope",
            "datasets",
            "dataset",
            "stem",
            "image_path",
            "mask_path",
            "image_pixel_sha256",
            "mask_binary_sha256",
        ),
    )
    write_csv(
        output_dir / "perceptual_duplicate_pairs.csv",
        perceptual,
        (
            "scope",
            "dataset_a",
            "stem_a",
            "image_path_a",
            "dataset_b",
            "stem_b",
            "image_path_b",
            "phash_distance",
            "dhash_distance",
            "requires_visual_review",
        ),
    )
    write_csv(
        output_dir / "duplicate_mask_consistency.csv",
        mask_comparisons,
        (
            "group_id",
            "dataset_a",
            "stem_a",
            "mask_path_a",
            "dataset_b",
            "stem_b",
            "mask_path_b",
            "same_shape",
            "same_binary_mask",
            "mask_dice",
            "mask_iou",
        ),
    )
    write_csv(
        output_dir / "resolution_counts.csv",
        resolution_rows(inventory),
        ("dataset", "width", "height", "count"),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    create_histogram(inventory, output_dir / "foreground_fraction_histogram.png")
    create_sample_montage(inventory, output_dir / "sample_montage.png")
    create_markdown_report(
        output_dir / "AUDITORIA_DATASETS.md",
        dataset_summaries=summaries,
        exact_rows=exact_rows,
        perceptual=perceptual,
        mask_comparisons=mask_comparisons,
        thresholds={
            "phash_distance": args.phash_distance,
            "dhash_distance": args.dhash_distance,
        },
    )

    print(json.dumps(summary["duplicates"], indent=2, ensure_ascii=False))
    print(f"Auditoria concluida: {output_dir / 'AUDITORIA_DATASETS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
