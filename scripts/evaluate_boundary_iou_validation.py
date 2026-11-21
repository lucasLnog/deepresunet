#!/usr/bin/env python3
"""Reavalia um checkpoint DR-IFT/DR-IFT-B com Boundary IoU na validação."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-dilation-ratio", type=float, default=0.02)
    return parser.parse_args()


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    import yaml

    root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not math.isclose(args.boundary_dilation_ratio, 0.02, abs_tol=1e-12):
        raise ValueError("A ablação de perímetro exige Boundary IoU com faixa de 2% da diagonal")
    if any(marker in str(run_dir).casefold() for marker in ("test", "external", "fuseg")):
        raise ValueError("Caminho de partição protegida bloqueado")

    sys.path.insert(0, str(root / "src"))
    from tcc_segmentation.data import DevelopmentDataset, ResizePad, create_development_dataloader
    from tcc_segmentation.sam_validation import boundary_iou
    from tcc_segmentation.baseline_training import make_model
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    if config["experiment_id"] not in {"DR-IFT", "DR-IFT-B"}:
        raise ValueError("Somente DR-IFT e DR-IFT-B são permitidas")
    manifest = root / config["data"]["validation_manifest"]
    if any(marker in str(manifest).casefold() for marker in ("test", "external", "fuseg")):
        raise ValueError("Manifest protegido bloqueado")
    normalization = config["data"]["normalization"]
    dataset = DevelopmentDataset(
        manifest,
        data_root,
        expected_split="validation",
        transform=ResizePad(int(config["data"]["height"]), int(config["data"]["width"])),
        mask_threshold=int(config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )
    if len(dataset) != 294:
        raise ValueError(f"Esperadas 294 imagens, obtidas {len(dataset)}")
    expected = read_csv(run_dir / "validation_per_image_selected.csv")
    expected_by_id = {row["canonical_id"]: row for row in expected}
    thresholds = {float(row["threshold"]) for row in expected}
    if len(thresholds) != 1:
        raise ValueError("CSV esperado não contém threshold único")
    threshold = thresholds.pop()

    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device)
    checkpoint = load_training_checkpoint(
        run_dir / "best_checkpoint.pt", model=model, map_location=device, restore_rng=False
    )
    loader = create_development_dataloader(
        dataset,
        batch_size=int(
            json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))[
                "selected_batch_size"
            ]
        ),
        shuffle=False,
        seed=int(config["seed"]),
        num_workers=0,
    )

    rows = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            probabilities = torch.sigmoid(model(images).float()).cpu().numpy()
            references = batch["mask"].numpy() >= 0.5
            predictions = probabilities >= threshold
            for index, canonical_id in enumerate(batch["canonical_id"]):
                prediction = predictions[index, 0]
                reference = references[index, 0]
                tp = int((prediction & reference).sum())
                fp = int((prediction & ~reference).sum())
                fn = int((~prediction & reference).sum())
                dice = 2.0 * tp / (2.0 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
                expected_row = expected_by_id[str(canonical_id)]
                if not math.isclose(
                    dice,
                    float(expected_row["dice"]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(f"{canonical_id}: Dice não reproduziu o CSV original")
                rows.append(
                    {
                        "canonical_id": str(canonical_id),
                        "experiment_id": config["experiment_id"],
                        "seed": int(config["seed"]),
                        "threshold": threshold,
                        "dice": dice,
                        "boundary_iou": boundary_iou(
                            prediction, reference, args.boundary_dilation_ratio
                        ),
                    }
                )
    if [row["canonical_id"] for row in rows] != [row["canonical_id"] for row in expected]:
        raise ValueError("Ordem ou identidade das imagens não reproduziu o CSV original")
    write_csv(output / "validation_boundary_per_image.csv", rows)
    boundary_values = [float(row["boundary_iou"]) for row in rows]
    summary = {
        "status": "completed",
        "partition": "validation",
        "run_id": run_dir.name,
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": threshold,
        "boundary_dilation_ratio": args.boundary_dilation_ratio,
        "boundary_iou_mean": float(np.mean(boundary_values)),
        "boundary_iou_median": float(np.median(boundary_values)),
        "boundary_iou_std": float(np.std(boundary_values, ddof=1)),
        "records": len(rows),
        "dice_reproduction_passed": True,
        "protected_partitions_accessed": False,
    }
    (output / "boundary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
