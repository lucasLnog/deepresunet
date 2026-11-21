#!/usr/bin/env python3
"""Audita visual e numericamente P0/P1/P2 sem treinar ou abrir partições protegidas."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/preprocessing_audit.yaml")
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_hwc_uint8(sample: dict[str, object]) -> np.ndarray:
    image = np.asarray(sample["image"], dtype=np.float32).transpose(1, 2, 0)
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float64) / 255.0
    return 0.2126 * values[..., 0] + 0.7152 * values[..., 1] + 0.0722 * values[..., 2]


def diagnostics(
    canonical_id: str,
    candidate: str,
    image: np.ndarray,
    reference: np.ndarray,
) -> dict[str, object]:
    values = image.astype(np.float64) / 255.0
    difference = np.abs(image.astype(np.int16) - reference.astype(np.int16)) / 255.0
    light = luminance(image)
    return {
        "canonical_id": canonical_id,
        "candidate": candidate,
        "red_mean": float(values[..., 0].mean()),
        "green_mean": float(values[..., 1].mean()),
        "blue_mean": float(values[..., 2].mean()),
        "luminance_mean": float(light.mean()),
        "luminance_std": float(light.std()),
        "near_black_fraction": float((values <= 1.0 / 255.0).mean()),
        "near_white_fraction": float((values >= 254.0 / 255.0).mean()),
        "mean_absolute_rgb_change_from_p0": float(difference.mean()),
        "p95_absolute_rgb_change_from_p0": float(np.quantile(difference, 0.95)),
    }


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt
    import yaml

    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root / "src"))

    from tcc_segmentation.data import (  # noqa: PLC0415
        ComposePair,
        DevelopmentDataset,
        ImageOnlyPairTransform,
        ResizePad,
        build_color_preprocessor,
    )

    run_id = datetime.now(timezone.utc).strftime("audit_P0_P1_P2_%Y%m%dT%H%M%SZ")
    output_dir = root / config["runtime"]["output_root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_used.yaml")

    datasets = {}
    for candidate, candidate_config in config["candidates"].items():
        color = build_color_preprocessor(
            candidate_config["method"],
            minkowski_p=float(candidate_config.get("minkowski_p", 6.0)),
            clip_limit=float(candidate_config.get("clip_limit", 2.0)),
            tile_grid_size=tuple(candidate_config.get("tile_grid_size", [8, 8])),
        )
        transform = ComposePair(
            [
                ImageOnlyPairTransform(color),
                ResizePad(
                    height=int(config["data"]["height"]),
                    width=int(config["data"]["width"]),
                ),
            ]
        )
        datasets[candidate] = DevelopmentDataset(
            root / config["data"]["validation_manifest"],
            data_root,
            expected_split="validation",
            transform=transform,
            mask_threshold=int(config["data"]["mask_threshold"]),
        )

    ids = [str(value) for value in config["visualization"]["fixed_validation_ids"]]
    index_by_id = {str(row["canonical_id"]): index for index, row in enumerate(datasets["P0"].rows)}
    missing = [canonical_id for canonical_id in ids if canonical_id not in index_by_id]
    if missing:
        raise ValueError(f"IDs fixos ausentes: {missing}")

    images: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for canonical_id in ids:
        index = index_by_id[canonical_id]
        images[canonical_id] = {
            candidate: to_hwc_uint8(dataset[index]) for candidate, dataset in datasets.items()
        }
        reference = images[canonical_id]["P0"]
        for candidate in ("P0", "P1", "P2"):
            rows.append(
                diagnostics(
                    canonical_id,
                    candidate,
                    images[canonical_id][candidate],
                    reference,
                )
            )

    write_csv(output_dir / "preprocessing_diagnostics.csv", rows)

    figure, axes = plt.subplots(len(ids), 5, figsize=(20, 3.25 * len(ids)))
    for row_index, canonical_id in enumerate(ids):
        p0, p1, p2 = (images[canonical_id][key] for key in ("P0", "P1", "P2"))
        diff1 = np.abs(p1.astype(np.int16) - p0.astype(np.int16)).mean(axis=2)
        diff2 = np.abs(p2.astype(np.int16) - p0.astype(np.int16)).mean(axis=2)
        tiles = (p0, p1, p2, diff1, diff2)
        titles = ("P0 — RGB", "P1 — Shades-of-Gray", "P2 — CLAHE L*", "|P1−P0|", "|P2−P0|")
        for column, (tile, title) in enumerate(zip(tiles, titles)):
            if column < 3:
                axes[row_index, column].imshow(tile)
            else:
                axes[row_index, column].imshow(tile, cmap="magma", vmin=0, vmax=64)
            axes[row_index, column].set_title(f"{canonical_id} | {title}")
            axes[row_index, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "preprocessing_panel.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for axis, canonical_id in zip(axes.ravel(), ids):
        for candidate in ("P0", "P1", "P2"):
            axis.hist(
                luminance(images[canonical_id][candidate]).ravel(),
                bins=64,
                range=(0.0, 1.0),
                density=True,
                histtype="step",
                linewidth=1.5,
                label=candidate,
            )
        axis.set_title(canonical_id)
        axis.set_xlabel("Luminância relativa")
        axis.set_ylabel("Densidade")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "luminance_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    summary: dict[str, object] = {
        "status": "completed",
        "run_id": run_id,
        "candidates": config["candidates"],
        "fixed_validation_ids": ids,
        "images_checked": len(ids),
        "aggregate_diagnostics": {},
        "selection_performed": False,
        "model_trained": False,
        "partitions_accessed": ["validation"],
        "protected_partitions_accessed": False,
    }
    for candidate in ("P0", "P1", "P2"):
        subset = [row for row in rows if row["candidate"] == candidate]
        summary["aggregate_diagnostics"][candidate] = {
            key: float(np.mean([float(row[key]) for row in subset]))
            for key in (
                "red_mean",
                "green_mean",
                "blue_mean",
                "luminance_mean",
                "luminance_std",
                "near_black_fraction",
                "near_white_fraction",
                "mean_absolute_rgb_change_from_p0",
                "p95_absolute_rgb_change_from_p0",
            )
        }
    write_json(output_dir / "preprocessing_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
