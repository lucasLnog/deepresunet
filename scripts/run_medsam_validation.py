#!/usr/bin/env python3
"""Compara DR-IFT, SAM 1 e MedSAM na validação congelada da avaliação do MedSAM na validação."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/medsam_validation.yaml")
    )
    parser.add_argument("--network-checkpoint", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--medsam-checkpoint", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    sys.path.insert(0, str(root / "src"))

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageFont
    from segment_anything import SamPredictor, sam_model_registry

    from tcc_segmentation.data import DevelopmentDataset, ResizePad
    from tcc_segmentation.data.manifest import resolve_dataset_path
    from tcc_segmentation.sam_refinement import (
        PROMPT_MODES,
        binary_metric_row,
        canvas_to_original,
        original_mask_to_canvas,
        refine_prediction_with_sam,
        sha256_file,
    )
    from tcc_segmentation.sam_validation import boundary_iou
    from tcc_segmentation.medsam_refinement import refine_prediction_with_medsam_box
    from tcc_segmentation.medsam_validation import (
        BASELINE_VARIANT,
        GENERIC_SAM_VARIANTS,
        MEDSAM_VARIANT,
        VARIANTS,
        aggregate_variant_rows,
        apply_advance_rule,
        paired_comparisons,
        subgroup_summary,
        validate_day13b_config,
    )
    from tcc_segmentation.baseline_training import make_model, write_csv, write_json
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_day13b_config(config)
    manifest = root / config["data"]["manifest"]
    fixed_manifest = root / config["data"]["fixed_visualization_manifest"]
    for path, expected in (
        (manifest, config["data"]["manifest_sha256"]),
        (fixed_manifest, config["data"]["fixed_visualization_manifest_sha256"]),
    ):
        if sha256_file(path).casefold() != expected.casefold():
            raise ValueError(f"hash de arquivo congelado divergente: {path}")

    run_id = (
        f"{config['runtime']['run_prefix']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output = root / config["runtime"]["output_root"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    network_run = root / config["network"]["run_directory"]
    network_checkpoint = (
        args.network_checkpoint or network_run / config["network"]["checkpoint_filename"]
    )
    sam_checkpoint = args.sam_checkpoint or root / config["sam"]["checkpoint_path"]
    medsam_checkpoint = args.medsam_checkpoint or root / config["medsam"]["checkpoint_path"]
    required = [
        network_checkpoint,
        sam_checkpoint,
        medsam_checkpoint,
        network_run / "config_used.yaml",
    ]
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(f"arquivo obrigatório ausente: {path}")
    if sha256_file(sam_checkpoint) != config["sam"]["checkpoint_sha256"]:
        raise ValueError("hash do checkpoint SAM genérico divergente")
    if sha256_file(medsam_checkpoint) != config["medsam"]["checkpoint_sha256"]:
        raise ValueError("hash do checkpoint MedSAM divergente")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    network_config = yaml.safe_load((network_run / "config_used.yaml").read_text(encoding="utf-8"))
    if network_config.get("experiment_id") != "DR-IFT" or int(network_config.get("seed", -1)) != 42:
        raise ValueError("run da rede não corresponde ao DR-IFT seed 42")
    model_config = dict(network_config)
    model_config["model"] = {**network_config["model"], "encoder_weights": None}
    network_model = make_model(model_config).to(device)
    payload = load_training_checkpoint(
        network_checkpoint, model=network_model, map_location=device, restore_rng=False
    )
    if payload.get("run_id") != config["network"]["run_id"] or int(payload.get("epoch", -1)) != 56:
        raise ValueError("identidade do checkpoint DR-IFT divergente")
    network_model.eval().requires_grad_(False)

    sam_model = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint)).to(device)
    sam_model.eval().requires_grad_(False)
    predictor = SamPredictor(sam_model)
    medsam_model = sam_model_registry["vit_b"](checkpoint=str(medsam_checkpoint)).to(device)
    medsam_model.eval().requires_grad_(False)

    normalization = network_config["data"]["normalization"]
    transform = ResizePad(config["network"]["input_height"], config["network"]["input_width"])
    dataset = DevelopmentDataset(
        manifest,
        data_root,
        "validation",
        transform=transform,
        mask_threshold=int(network_config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )
    if len(dataset) != 294:
        raise ValueError(f"esperadas 294 imagens; encontradas {len(dataset)}")
    ids = [str(row["canonical_id"]) for row in dataset.rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados na validação")
    with fixed_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        fixed_rows = list(csv.DictReader(handle))
    fixed_ids = [row["canonical_id"] for row in fixed_rows]
    if len(fixed_ids) != 12 or not set(fixed_ids).issubset(ids):
        raise ValueError("casos visuais congelados inválidos")

    threshold = float(config["network"]["threshold"])
    max_components = int(config["prompts"]["max_components_safety"])
    metric_rows, prompt_audit, fixed_visuals = [], [], {}
    started_all = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(len(dataset)):
        sample, manifest_row = dataset[index], dataset.rows[index]
        canonical_id = str(sample["canonical_id"])
        with Image.open(resolve_dataset_path(data_root, manifest_row["image_path"])) as source:
            original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        input_tensor = torch.as_tensor(sample["image"])[None].to(device)
        began = time.perf_counter()
        with torch.inference_mode():
            probability_canvas = (
                torch.sigmoid(network_model(input_tensor).float())[0, 0].cpu().numpy()
            )
        network_seconds = time.perf_counter() - began
        probability_original = canvas_to_original(
            probability_canvas,
            original_height=original_rgb.shape[0],
            original_width=original_rgb.shape[1],
            interpolation="bilinear",
        )
        reference = np.asarray(sample["mask"])[0] >= 0.5
        predictions = {BASELINE_VARIANT: probability_canvas >= threshold}
        post_seconds = {BASELINE_VARIANT: 0.0}
        source_components = None

        for mode, variant in zip(PROMPT_MODES, GENERIC_SAM_VARIANTS):
            began = time.perf_counter()
            original_prediction, audit = refine_prediction_with_sam(
                predictor=predictor,
                rgb_image=original_rgb,
                network_probabilities=probability_original,
                network_threshold=threshold,
                mode=mode,
                max_components_safety=max_components,
                sam_image_encoder_size=int(config["sam"]["image_encoder_size"]),
                mask_input_size=int(config["prompts"]["mask_input_size"][0]),
                outside_logit=float(config["prompts"]["outside_logit"]),
            )
            post_seconds[variant] = time.perf_counter() - began
            predictions[variant] = original_mask_to_canvas(
                original_prediction,
                target_height=reference.shape[0],
                target_width=reference.shape[1],
            )
            source_components = int(audit["component_count"])
            prompt_audit.append({"canonical_id": canonical_id, "variant": variant, **audit})

        began = time.perf_counter()
        medsam_original, medsam_audit = refine_prediction_with_medsam_box(
            model=medsam_model,
            rgb_image=original_rgb,
            network_probabilities=probability_original,
            network_threshold=threshold,
            max_components_safety=max_components,
            output_threshold=float(config["medsam"]["output_threshold"]),
        )
        post_seconds[MEDSAM_VARIANT] = time.perf_counter() - began
        predictions[MEDSAM_VARIANT] = original_mask_to_canvas(
            medsam_original, target_height=reference.shape[0], target_width=reference.shape[1]
        )
        prompt_audit.append(
            {"canonical_id": canonical_id, "variant": MEDSAM_VARIANT, **medsam_audit}
        )
        if source_components != int(medsam_audit["component_count"]):
            raise RuntimeError("SAM e MedSAM não receberam os mesmos componentes")

        common = {
            "canonical_id": canonical_id,
            "foreground_area_bin": int(manifest_row["foreground_area_bin"]),
            "foreground_fraction": float(manifest_row["foreground_fraction"]),
            "reference_components_8": int(manifest_row["connected_components_8"]),
            "reference_pixels": int(reference.sum()),
            "network_source_components": source_components,
            "network_seconds": network_seconds,
        }
        for variant in VARIANTS:
            prediction = predictions[variant]
            metric_rows.append(
                {
                    **common,
                    "variant": variant,
                    "postprocess_seconds": post_seconds[variant],
                    "prediction_pixels": int(prediction.sum()),
                    **binary_metric_row(prediction, reference),
                    "boundary_iou": boundary_iou(prediction, reference, 0.02),
                }
            )
        if canonical_id in fixed_ids:
            rgb_canvas, _ = transform(
                original_rgb, np.zeros(original_rgb.shape[:2], dtype=np.uint8)
            )
            fixed_visuals[canonical_id] = {
                "rgb": rgb_canvas,
                "reference": reference,
                "probability": probability_canvas,
                **predictions,
            }
        completed = index + 1
        if completed == 1 or completed % 10 == 0 or completed == len(dataset):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(dataset),
                        "canonical_id": canonical_id,
                        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
                    }
                ),
                flush=True,
            )

    if len(metric_rows) != 294 * 5 or len(prompt_audit) != 294 * 4:
        raise RuntimeError("quantidade inesperada de registros")
    summary = aggregate_variant_rows(metric_rows)
    comparisons, differences = paired_comparisons(
        metric_rows,
        bootstrap_samples=config["statistics"]["bootstrap_samples"],
        permutation_samples=config["statistics"]["permutation_samples"],
        seed=config["statistics"]["seed"],
    )
    decision = apply_advance_rule(comparisons, config)
    write_csv(output / "validation_per_image.csv", metric_rows)
    write_csv(output / "variant_summary.csv", summary)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "paired_differences_per_image.csv", differences)
    write_csv(output / "validation_by_area_bin.csv", subgroup_summary(metric_rows))
    write_json(output / "advance_decision.json", decision)
    write_json(
        output / "prompt_audit.json",
        {
            "record_count": len(prompt_audit),
            "records": prompt_audit,
            "all_prompts_from_network": all(
                not row["ground_truth_used_for_prompt"] for row in prompt_audit
            ),
            "protected_partitions_accessed": False,
        },
    )

    def font(size):
        for candidate in (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ):
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def overlay(rgb, mask, color):
        result = rgb.astype(np.float32).copy()
        result[mask] = 0.45 * result[mask] + 0.55 * np.asarray(color)
        return np.clip(result, 0, 255).astype(np.uint8)

    def errors(pred, ref):
        result = np.zeros((*pred.shape, 3), dtype=np.uint8)
        result[pred & ref] = (0, 190, 0)
        result[pred & ~ref] = (255, 165, 0)
        result[~pred & ref] = (230, 0, 0)
        return result

    def save_panel(path, columns):
        tile_w, tile_h, label_h = 320, 240, 22
        panel = Image.new(
            "RGB", (tile_w * len(columns), (tile_h + label_h) * len(fixed_ids)), "white"
        )
        draw = ImageDraw.Draw(panel)
        for row_index, cid in enumerate(fixed_ids):
            row = fixed_visuals[cid]
            for col_index, (title, key) in enumerate(columns):
                if key == "rgb":
                    tile = row[key]
                elif key == "probability":
                    tile = np.repeat(
                        (np.clip(row[key], 0, 1) * 255).astype(np.uint8)[..., None], 3, 2
                    )
                elif key == "reference":
                    tile = overlay(row["rgb"], row[key], (235, 40, 40))
                elif key.startswith("errors:"):
                    tile = errors(row[key.split(":", 1)[1]], row["reference"])
                else:
                    tile = overlay(row["rgb"], row[key], (20, 200, 50))
                x, y = col_index * tile_w, row_index * (tile_h + label_h)
                panel.paste(
                    Image.fromarray(tile).resize((tile_w, tile_h), Image.Resampling.BILINEAR),
                    (x, y + label_h),
                )
                draw.text((x + 3, y + 3), f"{cid} | {title}", fill="black", font=font(11))
        panel.save(path)

    save_panel(
        output / "fixed_validation_panel_all_models.png",
        [
            ("imagem", "rgb"),
            ("referência", "reference"),
            ("prob. DR-IFT", "probability"),
            ("DR-IFT", BASELINE_VARIANT),
            ("SAM box", "SAM_box"),
            ("SAM mask", "SAM_mask_input"),
            ("SAM box+mask", "SAM_box_plus_mask"),
            ("MedSAM box", MEDSAM_VARIANT),
            ("erros MedSAM", "errors:MedSAM_box"),
        ],
    )
    save_panel(
        output / "fixed_validation_panel_medsam.png",
        [
            ("imagem", "rgb"),
            ("referência", "reference"),
            ("prob. DR-IFT", "probability"),
            ("DR-IFT", BASELINE_VARIANT),
            ("SAM box+mask", "SAM_box_plus_mask"),
            ("MedSAM box", MEDSAM_VARIANT),
            ("erros MedSAM", "errors:MedSAM_box"),
        ],
    )

    summary_map = {row["variant"]: row for row in summary}
    effects = [
        row
        for row in comparisons
        if row["metric"] == "dice" and row["reference"] == BASELINE_VARIANT
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    labels = ["DR-IFT", "SAM box", "SAM mask", "SAM box+mask", "MedSAM"]
    values = [summary_map[v]["dice_mean"] for v in VARIANTS]
    axes[0].bar(labels, values, color=["#475569", "#2563eb", "#7c3aed", "#0891b2", "#dc2626"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Dice macro por imagem")
    axes[0].set_title("Validação congelada (n=294)")
    axes[0].tick_params(axis="x", rotation=15)
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.015, f"{value:.3f}", ha="center")
    means = [row["mean_difference"] for row in effects]
    axes[1].errorbar(
        means,
        [row["candidate"] for row in effects],
        xerr=[
            [m - row["ci95_low"] for m, row in zip(means, effects)],
            [row["ci95_high"] - m for m, row in zip(means, effects)],
        ],
        fmt="o",
        capsize=5,
        color="#0f172a",
        ecolor="#2563eb",
    )
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set_xlabel("Diferença de Dice vs DR-IFT")
    axes[1].set_title("IC 95% bootstrap pareado")
    fig.tight_layout()
    fig.savefig(output / "medsam_validation_summary.png", dpi=180)
    plt.close(fig)

    elapsed = time.perf_counter() - started_all
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
    }
    write_json(output / "environment_report.json", environment)
    report = {
        "status": "completed",
        "run_id": run_id,
        "scientific_role": config["scientific_role"],
        "partition": "validation",
        "records": 294,
        "metric_rows": len(metric_rows),
        "prompt_audit_records": len(prompt_audit),
        "network": {
            "run_id": payload["run_id"],
            "checkpoint_epoch": int(payload["epoch"]),
            "threshold": threshold,
            "checkpoint_sha256": sha256_file(network_checkpoint),
        },
        "sam": {
            "checkpoint_sha256": sha256_file(sam_checkpoint),
            "repository_commit": config["sam"]["repository_commit"],
            "weights_frozen": True,
        },
        "medsam": {
            "checkpoint_sha256": sha256_file(medsam_checkpoint),
            "preprocessing": config["medsam"]["preprocessing"],
            "weights_frozen": True,
        },
        "advance_decision": decision,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "all_prompts_from_network": True,
        "ground_truth_used_only_for_metrics_and_panels": True,
        "manifest_sha256": sha256_file(manifest),
        "fixed_visualization_manifest_sha256": sha256_file(fixed_manifest),
        "partitions_accessed": ["validation"],
        "protected_partitions_accessed": False,
    }
    write_json(output / "technical_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
