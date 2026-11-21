#!/usr/bin/env python3
"""Executa o smoke test do SAM 1 ViT-B somente em casos fixos da validação."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/sam_vit_b_integration.yaml"))
    parser.add_argument("--network-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    sys.path.insert(0, str(root / "src"))

    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageFont

    from tcc_segmentation.data import DevelopmentDataset, ResizePad
    from tcc_segmentation.data.manifest import resolve_dataset_path
    from tcc_segmentation.sam_refinement import (
        PROMPT_MODES,
        binary_metric_row,
        canvas_to_original,
        original_mask_to_canvas,
        refine_prediction_with_sam,
        sha256_file,
        validate_day12_config,
    )
    from tcc_segmentation.baseline_training import make_model, write_csv, write_json
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_day12_config(config)

    run_id = (
        f"{config['runtime']['run_prefix']}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir = root / config["runtime"]["output_root"] / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    network_run = root / config["network"]["run_directory"]
    network_checkpoint = (
        args.network_checkpoint.expanduser().resolve()
        if args.network_checkpoint
        else network_run / config["network"]["checkpoint_filename"]
    )
    sam_checkpoint = (
        args.sam_checkpoint.expanduser().resolve()
        if args.sam_checkpoint
        else root / config["sam"]["checkpoint_path"]
    )
    if not network_checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint DR-IFT ausente: {network_checkpoint}. "
            "Use exatamente o best_checkpoint.pt da seed 42."
        )
    if not sam_checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint SAM ausente: {sam_checkpoint}. Execute a célula de download."
        )
    stored_config_path = network_run / "config_used.yaml"
    if not stored_config_path.is_file():
        raise FileNotFoundError(f"Configuração da DR-IFT ausente: {stored_config_path}")
    network_config = yaml.safe_load(stored_config_path.read_text(encoding="utf-8"))
    if network_config.get("experiment_id") != "DR-IFT" or int(network_config.get("seed")) != 42:
        raise ValueError("diretório não corresponde ao DR-IFT da seed 42")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    model_config = dict(network_config)
    model_config["model"] = dict(network_config["model"])
    # O checkpoint contém todos os pesos; impedir um download ImageNet redundante.
    model_config["model"]["encoder_weights"] = None
    network_model = make_model(model_config).to(device)
    payload = load_training_checkpoint(
        network_checkpoint, model=network_model, map_location=device, restore_rng=False
    )
    if payload.get("run_id") != config["network"]["run_id"]:
        raise ValueError("run_id do checkpoint DR-IFT difere do protocolo")
    if int(payload.get("epoch", -1)) != int(config["network"]["checkpoint_epoch"]):
        raise ValueError("época do checkpoint DR-IFT difere do protocolo")
    network_model.eval()
    for parameter in network_model.parameters():
        parameter.requires_grad_(False)

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError(
            "segment-anything não instalado. Execute a célula de instalação do notebook."
        ) from error
    sam_model = sam_model_registry[config["sam"]["model_type"]](checkpoint=str(sam_checkpoint)).to(
        device
    )
    sam_model.eval()
    for parameter in sam_model.parameters():
        parameter.requires_grad_(False)
    predictor = SamPredictor(sam_model)

    normalization = network_config["data"]["normalization"]
    transform = ResizePad(
        int(config["network"]["input_height"]), int(config["network"]["input_width"])
    )
    validation_dataset = DevelopmentDataset(
        root / config["data"]["manifest"],
        data_root,
        "validation",
        transform=transform,
        mask_threshold=int(network_config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )
    index_by_id = {
        str(row["canonical_id"]): index for index, row in enumerate(validation_dataset.rows)
    }
    missing = [value for value in config["data"]["smoke_ids"] if value not in index_by_id]
    if missing:
        raise ValueError(f"IDs fixos ausentes da validação: {missing}")

    threshold = float(config["network"]["threshold"])
    metric_rows: list[dict] = []
    prompt_audit: list[dict] = []
    visual_rows: list[dict] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started_all = time.perf_counter()

    for canonical_id in config["data"]["smoke_ids"]:
        dataset_index = index_by_id[canonical_id]
        sample = validation_dataset[dataset_index]
        manifest_row = validation_dataset.rows[dataset_index]
        image_path = resolve_dataset_path(data_root, manifest_row["image_path"])
        with Image.open(image_path) as source:
            original_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        input_tensor = torch.as_tensor(sample["image"])[None, ...].to(device)
        network_started = time.perf_counter()
        with torch.inference_mode():
            probability_canvas = torch.sigmoid(network_model(input_tensor).float())[0, 0]
        network_seconds = time.perf_counter() - network_started
        probability_canvas_np = probability_canvas.cpu().numpy()
        probability_original = canvas_to_original(
            probability_canvas_np,
            original_height=original_rgb.shape[0],
            original_width=original_rgb.shape[1],
            interpolation="bilinear",
        )
        reference_canvas = np.asarray(sample["mask"])[0] >= 0.5
        network_binary = probability_canvas_np >= threshold
        baseline = {
            "canonical_id": canonical_id,
            "variant": "DR-IFT_without_SAM",
            "network_seconds": network_seconds,
            "sam_seconds": 0.0,
            "component_count": None,
            **binary_metric_row(network_binary, reference_canvas),
        }
        metric_rows.append(baseline)
        outputs = {}
        audits = {}
        for mode in PROMPT_MODES:
            mode_started = time.perf_counter()
            sam_original, audit = refine_prediction_with_sam(
                predictor=predictor,
                rgb_image=original_rgb,
                network_probabilities=probability_original,
                network_threshold=threshold,
                mode=mode,
                max_components_safety=int(config["prompts"]["max_components_safety"]),
                sam_image_encoder_size=int(config["sam"]["image_encoder_size"]),
                mask_input_size=int(config["prompts"]["mask_input_size"][0]),
                outside_logit=float(config["prompts"]["outside_logit"]),
            )
            sam_seconds = time.perf_counter() - mode_started
            sam_canvas = original_mask_to_canvas(
                sam_original,
                target_height=reference_canvas.shape[0],
                target_width=reference_canvas.shape[1],
            )
            outputs[mode] = sam_canvas
            audits[mode] = audit
            metric_rows.append(
                {
                    "canonical_id": canonical_id,
                    "variant": f"SAM_{mode}",
                    "network_seconds": network_seconds,
                    "sam_seconds": sam_seconds,
                    "component_count": audit["component_count"],
                    **binary_metric_row(sam_canvas, reference_canvas),
                }
            )
            prompt_audit.append({"canonical_id": canonical_id, **audit})
        rgb_canvas, _ = transform(original_rgb, np.zeros(original_rgb.shape[:2], dtype=np.uint8))
        visual_rows.append(
            {
                "canonical_id": canonical_id,
                "rgb": rgb_canvas,
                "reference": reference_canvas,
                "probability": probability_canvas_np,
                "network": network_binary,
                **outputs,
            }
        )

    elapsed = time.perf_counter() - started_all
    write_csv(output_dir / "smoke_metrics.csv", metric_rows)
    write_json(
        output_dir / "prompt_audit.json",
        {
            "records": prompt_audit,
            "all_prompts_from_network": all(
                not row["ground_truth_used_for_prompt"] for row in prompt_audit
            ),
            "protected_partitions_accessed": False,
        },
    )

    def font(size: int):
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def mask_overlay(rgb, mask, color):
        result = rgb.astype(np.float32).copy()
        result[mask] = 0.45 * result[mask] + 0.55 * np.asarray(color)
        return np.clip(result, 0, 255).astype(np.uint8)

    def error_map(prediction, reference):
        result = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        result[prediction & reference] = (0, 190, 0)
        result[prediction & ~reference] = (255, 165, 0)
        result[~prediction & reference] = (230, 0, 0)
        return result

    tile_w, tile_h, label_h = 512, 384, 28
    columns = [
        ("imagem", "rgb"),
        ("referência", "reference"),
        ("prob. DR-IFT", "probability"),
        ("DR-IFT", "network"),
        ("SAM box", "box"),
        ("SAM mask", "mask_input"),
        ("SAM box+mask", "box_plus_mask"),
        ("erros box+mask", "errors"),
    ]
    panel = Image.new(
        "RGB", (tile_w * len(columns), (tile_h + label_h) * len(visual_rows)), "white"
    )
    draw = ImageDraw.Draw(panel)
    for row_index, row in enumerate(visual_rows):
        for col_index, (title, key) in enumerate(columns):
            if key == "rgb":
                tile = row["rgb"]
            elif key == "probability":
                gray = np.clip(row[key] * 255, 0, 255).astype(np.uint8)
                tile = np.repeat(gray[..., None], 3, axis=2)
            elif key == "reference":
                tile = mask_overlay(row["rgb"], row[key], (235, 40, 40))
            elif key == "errors":
                tile = error_map(row["box_plus_mask"], row["reference"])
            else:
                tile = mask_overlay(row["rgb"], row[key], (20, 200, 50))
            x, y = col_index * tile_w, row_index * (tile_h + label_h)
            panel.paste(Image.fromarray(tile), (x, y + label_h))
            draw.text(
                (x + 5, y + 5), f"{row['canonical_id']} | {title}", fill="black", font=font(15)
            )
    panel_path = output_dir / "sam_smoke_panel.png"
    panel.save(panel_path)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "sam_repository_commit": config["sam"]["repository_commit"],
    }
    write_json(output_dir / "environment_report.json", environment)
    report = {
        "status": "completed",
        "run_id": run_id,
        "scientific_role": config["scientific_role"],
        "network": {
            "run_id": payload["run_id"],
            "checkpoint_epoch": int(payload["epoch"]),
            "threshold": threshold,
            "checkpoint_sha256": sha256_file(network_checkpoint),
        },
        "sam": {
            "family": "SAM 1",
            "model_type": "vit_b",
            "checkpoint_sha256": sha256_file(sam_checkpoint),
            "repository_commit": config["sam"]["repository_commit"],
            "weights_frozen": True,
        },
        "smoke_ids": list(config["data"]["smoke_ids"]),
        "variants": ["DR-IFT_without_SAM", *[f"SAM_{mode}" for mode in PROMPT_MODES]],
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "all_prompts_from_network": True,
        "ground_truth_used_only_for_metrics_and_panel": True,
        "smoke_is_for_selection": False,
        "implementation_revision": int(config["implementation_revision"]),
        "mask_input_alignment": config["prompts"]["mask_input_alignment"],
        "partitions_accessed": ["validation"],
        "protected_partitions_accessed": False,
        "artifacts": {
            "metrics": str(output_dir / "smoke_metrics.csv"),
            "prompt_audit": str(output_dir / "prompt_audit.json"),
            "panel": str(panel_path),
        },
    }
    write_json(output_dir / "technical_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
