#!/usr/bin/env python3
"""Audita A1/A2 em casos fixos de treino, sem treinar ou abrir partições protegidas."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/augmentation_audit.yaml"))
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_pair(sample: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(sample["image"], dtype=np.float32).transpose(1, 2, 0)
    image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    mask = (np.asarray(sample["mask"])[0] >= 0.5).astype(np.uint8)
    return image, mask


def overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = 0.55 * result[selected] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def trace_names(trace: list[dict[str, object]]) -> list[str]:
    ignored = {"OneOf"}
    return [str(record["transform"]) for record in trace if record["transform"] not in ignored]


def diagnostics(
    canonical_id: str,
    pipeline: str,
    variant: int,
    seed: int,
    image: np.ndarray,
    mask: np.ndarray,
    reference_image: np.ndarray,
    reference_mask: np.ndarray,
    trace: list[dict[str, object]],
    retention_gate: float,
) -> dict[str, object]:
    original_pixels = int(reference_mask.sum())
    augmented_pixels = int(mask.sum())
    retention = float(augmented_pixels / original_pixels) if original_pixels else 1.0
    binary = set(np.unique(mask)).issubset({0, 1})
    aligned_shape = image.shape[:2] == mask.shape == reference_mask.shape
    black = np.all(image == 0, axis=2)
    names = trace_names(trace)
    reasons = []
    if not binary:
        reasons.append("mask_not_binary")
    if not aligned_shape:
        reasons.append("shape_mismatch")
    if retention < retention_gate:
        reasons.append("foreground_retention_below_gate")
    return {
        "canonical_id": canonical_id,
        "pipeline": pipeline,
        "variant": variant,
        "seed": seed,
        "applied_transforms": "|".join(names) if names else "none",
        "mask_binary": binary,
        "image_mask_shape_aligned": aligned_shape,
        "original_foreground_pixels": original_pixels,
        "augmented_foreground_pixels": augmented_pixels,
        "foreground_retention_ratio": retention,
        "foreground_fraction": float(mask.mean()),
        "black_border_fraction": float(black.mean()),
        "mean_absolute_rgb_change": float(
            np.abs(image.astype(np.int16) - reference_image.astype(np.int16)).mean() / 255.0
        ),
        "requires_review": bool(reasons),
        "review_reasons": "|".join(reasons),
    }


def validate_protocol(config: dict) -> None:
    if config["runtime"]["allowed_split"] != "train":
        raise ValueError("A auditoria da etapa de aumento de dados só pode usar treino")
    if config["runtime"]["model_training_allowed"]:
        raise ValueError("Treinamento deve permanecer desativado nesta auditoria")
    if config["pipelines"]["A1"]["affine"]["shear_degrees"] != [0.0, 0.0]:
        raise ValueError("Shear não foi autorizado")
    if not config["visualization"]["selected_before_augmentation_results"]:
        raise ValueError("Os casos precisam ter sido registrados antes da visualização")


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt
    import yaml

    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    sys.path.insert(0, str(root / "src"))

    from tcc_segmentation.data import (  # noqa: PLC0415
        ComposePair,
        DevelopmentDataset,
        ResizePad,
        build_augmentation_pipeline,
    )

    run_id = datetime.now(timezone.utc).strftime("audit_A1_A2_%Y%m%dT%H%M%SZ")
    output_dir = root / config["runtime"]["output_root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_used.yaml")

    dataset = DevelopmentDataset(
        root / config["data"]["train_manifest"],
        data_root,
        expected_split="train",
        transform=ComposePair(
            [ResizePad(height=int(config["data"]["height"]), width=int(config["data"]["width"]))]
        ),
        mask_threshold=int(config["data"]["mask_threshold"]),
    )
    ids = [str(value) for value in config["visualization"]["fixed_train_ids"]]
    index_by_id = {str(row["canonical_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [canonical_id for canonical_id in ids if canonical_id not in index_by_id]
    if missing:
        raise ValueError(f"IDs fixos ausentes do treino: {missing}")

    variants = [int(value) for value in config["visualization"]["variant_seeds"]]
    if len(variants) != int(config["visualization"]["variants_per_pipeline_per_image"]):
        raise ValueError("Quantidade de sementes não corresponde às variantes registradas")
    retention_gate = float(config["gate"]["minimum_foreground_retention_ratio_for_review"])
    rows: list[dict[str, object]] = []
    panels: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = {}

    for image_index, canonical_id in enumerate(ids):
        reference_image, reference_mask = to_pair(dataset[index_by_id[canonical_id]])
        panels[canonical_id] = [(reference_image, reference_mask, "A0 — original")]
        for pipeline_index, pipeline_name in enumerate(("A1", "A2"), start=1):
            for variant_index, base_seed in enumerate(variants, start=1):
                seed = base_seed + 100 * image_index + 10 * pipeline_index
                transform = build_augmentation_pipeline(
                    pipeline_name, config["pipelines"], seed=seed
                )
                image, mask, trace = transform.apply_with_trace(reference_image, reference_mask)
                names = trace_names(trace)
                title = f"{pipeline_name} v{variant_index}\n" + (
                    ", ".join(names) if names else "sem alteração sorteada"
                )
                panels[canonical_id].append((image, mask, title))
                rows.append(
                    diagnostics(
                        canonical_id,
                        pipeline_name,
                        variant_index,
                        seed,
                        image,
                        mask,
                        reference_image,
                        reference_mask,
                        trace,
                        retention_gate,
                    )
                )

    write_csv(output_dir / "augmentation_diagnostics.csv", rows)

    figure, axes = plt.subplots(len(ids), 7, figsize=(24, 3.0 * len(ids)))
    for row_index, canonical_id in enumerate(ids):
        for column, (image, mask, title) in enumerate(panels[canonical_id]):
            color = (255, 60, 40) if column == 0 else (30, 220, 70)
            axes[row_index, column].imshow(overlay(image, mask, color))
            axes[row_index, column].set_title(title, fontsize=8)
            if column == 0:
                axes[row_index, column].set_ylabel(canonical_id, fontsize=8)
            axes[row_index, column].axis("off")
    figure.suptitle(
        "Auditoria visual A1/A2 — somente treino; verde = máscara transformada",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_dir / "augmentation_panel.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(len(ids), 7, figsize=(24, 2.4 * len(ids)))
    for row_index, canonical_id in enumerate(ids):
        for column, (_, mask, title) in enumerate(panels[canonical_id]):
            axes[row_index, column].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[row_index, column].set_title(title.split("\n")[0], fontsize=8)
            if column == 0:
                axes[row_index, column].set_ylabel(canonical_id, fontsize=8)
            axes[row_index, column].axis("off")
    figure.suptitle(
        "Máscaras após transformação pareada — inspeção de forma e recorte", fontsize=14
    )
    figure.tight_layout()
    figure.savefig(output_dir / "augmentation_mask_panel.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    counts: Counter[str] = Counter()
    for row in rows:
        for name in str(row["applied_transforms"]).split("|"):
            if name != "none":
                counts[name] += 1
    summary = {
        "status": "completed",
        "run_id": run_id,
        "scientific_role": config["scientific_role"],
        "fixed_train_ids": ids,
        "images_checked": len(ids),
        "variants_checked_per_pipeline": len(variants),
        "pairs_checked": len(rows),
        "transform_application_counts": dict(sorted(counts.items())),
        "minimum_foreground_retention_ratio_observed": float(
            min(float(row["foreground_retention_ratio"]) for row in rows)
        ),
        "rows_requiring_review": int(sum(bool(row["requires_review"]) for row in rows)),
        "all_masks_binary": all(bool(row["mask_binary"]) for row in rows),
        "all_image_mask_shapes_aligned": all(bool(row["image_mask_shape_aligned"]) for row in rows),
        "selection_performed": False,
        "model_trained": False,
        "partitions_accessed": ["train"],
        "protected_partitions_accessed": False,
        "scientific_winner_requires_validation_training_metrics": True,
    }
    write_json(output_dir / "augmentation_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
