#!/usr/bin/env python3
"""Valida o pipeline de dados somente em treino ou validacao."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Raiz alternativa contendo Dataset/, por exemplo o cache do Colab.",
    )
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--with-torch-loader", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    sys.path.insert(0, str(root / "src"))
    from tcc_segmentation.data import (  # noqa: PLC0415
        DevelopmentDataset,
        ResizePad,
        create_development_dataloader,
    )

    config = yaml.safe_load((root / "configs" / "environment.yaml").read_text(encoding="utf-8"))
    manifest = root / "manifests" / "splits" / f"dfuc_{args.split}.csv"
    transform = ResizePad(
        height=int(config["data"]["target_height"]),
        width=int(config["data"]["target_width"]),
    )
    dataset = DevelopmentDataset(
        manifest,
        data_root,
        expected_split=args.split,
        transform=transform,
        mask_threshold=int(config["data"]["mask_threshold"]),
    )

    requested = min(args.samples, len(dataset))
    sample_summaries: list[dict[str, object]] = []
    for index in range(requested):
        sample = dataset[index]
        unique_mask = np.unique(sample["mask"])
        if not set(unique_mask.tolist()).issubset({0.0, 1.0}):
            raise AssertionError(f"Mascara nao binaria: {unique_mask}")
        sample_summaries.append(
            {
                "canonical_id": sample["canonical_id"],
                "image_shape": list(sample["image"].shape),
                "mask_shape": list(sample["mask"].shape),
                "mask_values": unique_mask.tolist(),
            }
        )

    torch_batch: dict[str, object] | None = None
    if args.with_torch_loader:
        loader = create_development_dataloader(
            dataset,
            batch_size=int(config["dataloader"]["batch_size_smoke"]),
            shuffle=args.split == "train",
            seed=int(config["dataloader"]["seed"]),
            num_workers=int(config["dataloader"]["num_workers_smoke"]),
        )
        batch = next(iter(loader))
        torch_batch = {
            "image_shape": list(batch["image"].shape),
            "mask_shape": list(batch["mask"].shape),
            "device": str(batch["image"].device),
        }

    result = {
        "status": "ok",
        "split": args.split,
        "manifest_records": len(dataset),
        "samples_checked": requested,
        "samples": sample_summaries,
        "torch_batch": torch_batch,
        "protected_partitions_accessed": False,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
