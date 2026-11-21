#!/usr/bin/env python3
"""Avalia os três modos do SAM nas 294 imagens da validação congelada."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/sam_validation.yaml"))
    parser.add_argument("--network-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
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
    from tcc_segmentation.sam_validation import (
        BASELINE_VARIANT,
        METRICS,
        SAM_VARIANTS,
        aggregate_variant_rows,
        boundary_iou,
        paired_comparisons,
        select_sam_variant,
        subgroup_summary,
        validate_day13_config,
    )
    from tcc_segmentation.baseline_training import make_model, write_csv, write_json
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_day13_config(config)

    manifest_path = root / config["data"]["manifest"]
    fixed_manifest_path = root / config["data"]["fixed_visualization_manifest"]
    if sha256_file(manifest_path).casefold() != config["data"]["manifest_sha256"].casefold():
        raise ValueError("hash da validação difere do protocolo congelado")
    if (
        sha256_file(fixed_manifest_path).casefold()
        != config["data"]["fixed_visualization_manifest_sha256"].casefold()
    ):
        raise ValueError("hash dos casos visuais difere do protocolo congelado")

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
    for required in (
        network_checkpoint,
        sam_checkpoint,
        network_run / "config_used.yaml",
        manifest_path,
        fixed_manifest_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"arquivo obrigatório ausente: {required}")

    network_config = yaml.safe_load((network_run / "config_used.yaml").read_text(encoding="utf-8"))
    if network_config.get("experiment_id") != "DR-IFT" or int(network_config.get("seed", -1)) != 42:
        raise ValueError("diretório não corresponde ao DR-IFT representativo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    model_config = dict(network_config)
    model_config["model"] = dict(network_config["model"])
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
            "segment-anything não instalado. Execute a célula de instalação."
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
        int(config["network"]["input_height"]),
        int(config["network"]["input_width"]),
    )
    dataset = DevelopmentDataset(
        manifest_path,
        data_root,
        "validation",
        transform=transform,
        mask_threshold=int(network_config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"],
        normalization_std=normalization["std"],
    )
    expected_records = int(config["data"]["expected_records"])
    if len(dataset) != expected_records:
        raise ValueError(f"esperadas {expected_records} imagens; encontradas {len(dataset)}")
    ids = [str(row["canonical_id"]) for row in dataset.rows]
    if len(ids) != len(set(ids)):
        raise ValueError("IDs duplicados na validação")

    with fixed_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        fixed_rows = list(csv.DictReader(handle))
    if len(fixed_rows) != int(config["data"]["expected_fixed_cases"]):
        raise ValueError("quantidade de casos visuais diferente da congelada")
    fixed_ids = [row["canonical_id"] for row in fixed_rows]
    if not set(fixed_ids).issubset(ids):
        raise ValueError("caso visual não pertence à validação")

    threshold = float(config["network"]["threshold"])
    boundary_ratio = float(config["evaluation"]["boundary_dilation_ratio"])
    metric_rows: list[dict] = []
    prompt_audit: list[dict] = []
    fixed_visuals: dict[str, dict] = {}
    progress_every = int(config["runtime"]["progress_every_images"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started_all = time.perf_counter()

    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        manifest_row = dataset.rows[dataset_index]
        canonical_id = str(sample["canonical_id"])
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

        outputs: dict[str, np.ndarray] = {}
        audits: dict[str, dict] = {}
        mode_seconds: dict[str, float] = {}
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
            mode_seconds[mode] = time.perf_counter() - mode_started
            outputs[mode] = original_mask_to_canvas(
                sam_original,
                target_height=reference_canvas.shape[0],
                target_width=reference_canvas.shape[1],
            )
            audits[mode] = audit
            prompt_audit.append({"canonical_id": canonical_id, **audit})

        common = {
            "canonical_id": canonical_id,
            "foreground_area_bin": int(manifest_row["foreground_area_bin"]),
            "foreground_fraction": float(manifest_row["foreground_fraction"]),
            "reference_components_8": int(manifest_row["connected_components_8"]),
            "reference_pixels": int(reference_canvas.sum()),
            "network_source_components": int(audits["box"]["component_count"]),
            "network_seconds": network_seconds,
        }

        def metric_record(variant: str, prediction: np.ndarray, sam_seconds: float):
            return {
                **common,
                "variant": variant,
                "sam_seconds": sam_seconds,
                "prediction_pixels": int(prediction.sum()),
                **binary_metric_row(prediction, reference_canvas),
                "boundary_iou": boundary_iou(prediction, reference_canvas, boundary_ratio),
            }

        metric_rows.append(metric_record(BASELINE_VARIANT, network_binary, 0.0))
        for mode, variant in zip(PROMPT_MODES, SAM_VARIANTS):
            metric_rows.append(metric_record(variant, outputs[mode], mode_seconds[mode]))

        if canonical_id in fixed_ids:
            rgb_canvas, _ = transform(
                original_rgb, np.zeros(original_rgb.shape[:2], dtype=np.uint8)
            )
            fixed_visuals[canonical_id] = {
                "canonical_id": canonical_id,
                "rgb": rgb_canvas,
                "reference": reference_canvas,
                "probability": probability_canvas_np,
                "network": network_binary,
                **outputs,
            }

        completed = dataset_index + 1
        if completed == 1 or completed % progress_every == 0 or completed == len(dataset):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(dataset),
                        "canonical_id": canonical_id,
                        "elapsed_seconds": round(time.perf_counter() - started_all, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started_all
    if set(fixed_visuals) != set(fixed_ids):
        raise RuntimeError("painel fixo incompleto")
    if len(metric_rows) != expected_records * 4 or len(prompt_audit) != expected_records * 3:
        raise RuntimeError("quantidade inesperada de registros de saída")

    summary_rows = aggregate_variant_rows(metric_rows)
    comparison_rows, difference_rows = paired_comparisons(
        metric_rows,
        bootstrap_samples=int(config["statistics"]["bootstrap_samples"]),
        permutation_samples=int(config["statistics"]["permutation_samples"]),
        seed=int(config["statistics"]["seed"]),
    )
    selection = select_sam_variant(
        summary_rows,
        tolerance=float(config["evaluation"]["plateau_tolerance"]),
        tie_break_order=config["evaluation"]["tie_break_order"],
    )
    selected_variant = selection["selected_variant"]
    selected_comparison = f"{selected_variant}_minus_{BASELINE_VARIANT}"
    selected_effect = next(
        row
        for row in comparison_rows
        if row["comparison"] == selected_comparison and row["metric"] == "dice"
    )
    selection.update(
        {
            "partition": "validation",
            "selected_vs_baseline_dice": selected_effect,
            "inference_role": "development_selection_not_final_effect_estimate",
            "threshold_optimization_performed": False,
            "selected_mode_to_freeze_before_test": True,
            "protected_partitions_accessed": False,
        }
    )

    write_csv(output_dir / "validation_per_image.csv", metric_rows)
    write_csv(output_dir / "variant_summary.csv", summary_rows)
    write_csv(output_dir / "paired_comparisons.csv", comparison_rows)
    write_csv(output_dir / "paired_differences_per_image.csv", difference_rows)
    write_csv(output_dir / "validation_by_area_bin.csv", subgroup_summary(metric_rows))
    write_json(output_dir / "selection.json", selection)
    write_json(
        output_dir / "prompt_audit.json",
        {
            "records": prompt_audit,
            "record_count": len(prompt_audit),
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

    def save_panel(path: Path, columns):
        tile_w, tile_h, label_h = 384, 288, 24
        ordered = [fixed_visuals[item] for item in fixed_ids]
        panel = Image.new(
            "RGB", (tile_w * len(columns), (tile_h + label_h) * len(ordered)), "white"
        )
        draw = ImageDraw.Draw(panel)
        for row_index, row in enumerate(ordered):
            for col_index, (title, key) in enumerate(columns):
                if key == "rgb":
                    tile = row["rgb"]
                elif key == "probability":
                    gray = np.clip(row[key] * 255, 0, 255).astype(np.uint8)
                    tile = np.repeat(gray[..., None], 3, axis=2)
                elif key == "reference":
                    tile = mask_overlay(row["rgb"], row[key], (235, 40, 40))
                elif key == "selected_errors":
                    tile = error_map(row[selected_variant.removeprefix("SAM_")], row["reference"])
                else:
                    tile = mask_overlay(row["rgb"], row[key], (20, 200, 50))
                tile_image = Image.fromarray(tile).resize(
                    (tile_w, tile_h), resample=Image.Resampling.BILINEAR
                )
                x, y = col_index * tile_w, row_index * (tile_h + label_h)
                panel.paste(tile_image, (x, y + label_h))
                draw.text(
                    (x + 4, y + 4),
                    f"{row['canonical_id']} | {title}",
                    fill="black",
                    font=font(12),
                )
        panel.save(path)

    save_panel(
        output_dir / "fixed_validation_panel_all_modes.png",
        [
            ("imagem", "rgb"),
            ("referência", "reference"),
            ("prob. DR-IFT", "probability"),
            ("DR-IFT", "network"),
            ("SAM box", "box"),
            ("SAM mask", "mask_input"),
            ("SAM box+mask", "box_plus_mask"),
            ("erros selecionado", "selected_errors"),
        ],
    )
    selected_key = selected_variant.removeprefix("SAM_")
    save_panel(
        output_dir / "fixed_validation_panel_selected.png",
        [
            ("imagem", "rgb"),
            ("referência", "reference"),
            ("prob. DR-IFT", "probability"),
            ("DR-IFT", "network"),
            (selected_variant, selected_key),
            ("erros selecionado", "selected_errors"),
        ],
    )

    summary_by_variant = {row["variant"]: row for row in summary_rows}
    dice_effects = [row for row in comparison_rows if row["metric"] == "dice"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ["DR-IFT", "SAM box", "SAM mask", "SAM box+mask"]
    values = [
        summary_by_variant[variant]["dice_mean"] for variant in (BASELINE_VARIANT, *SAM_VARIANTS)
    ]
    colors = ["#475569", "#2563eb", "#7c3aed", "#0891b2"]
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel("Dice macro por imagem")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Validação congelada (n=294)")
    axes[0].tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.015, f"{value:.3f}", ha="center")
    means = [row["mean_difference"] for row in dice_effects]
    low = [mean - row["ci95_low"] for mean, row in zip(means, dice_effects)]
    high = [row["ci95_high"] - mean for mean, row in zip(means, dice_effects)]
    axes[1].errorbar(
        means,
        ["box", "mask_input", "box+mask"],
        xerr=[low, high],
        fmt="o",
        color="#0f172a",
        ecolor="#2563eb",
        capsize=5,
    )
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Diferença de Dice: SAM − DR-IFT")
    axes[1].set_title("IC 95% bootstrap pareado")
    figure.tight_layout()
    figure.savefig(output_dir / "sam_validation_summary.png", dpi=180)
    plt.close(figure)

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
        "partition": "validation",
        "records": expected_records,
        "metric_rows": len(metric_rows),
        "prompt_audit_records": len(prompt_audit),
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
        "selection": selection,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "all_prompts_from_network": True,
        "ground_truth_used_only_for_metrics_and_panels": True,
        "implementation_revision": int(config["implementation_revision"]),
        "mask_input_alignment": config["prompts"]["mask_input_alignment"],
        "manifest_sha256": sha256_file(manifest_path),
        "fixed_visualization_manifest_sha256": sha256_file(fixed_manifest_path),
        "partitions_accessed": ["validation"],
        "protected_partitions_accessed": False,
        "artifacts": {
            "per_image": str(output_dir / "validation_per_image.csv"),
            "variant_summary": str(output_dir / "variant_summary.csv"),
            "paired_comparisons": str(output_dir / "paired_comparisons.csv"),
            "paired_differences": str(output_dir / "paired_differences_per_image.csv"),
            "area_bins": str(output_dir / "validation_by_area_bin.csv"),
            "selection": str(output_dir / "selection.json"),
            "prompt_audit": str(output_dir / "prompt_audit.json"),
            "panel_all_modes": str(output_dir / "fixed_validation_panel_all_modes.png"),
            "panel_selected": str(output_dir / "fixed_validation_panel_selected.png"),
            "summary_figure": str(output_dir / "sam_validation_summary.png"),
        },
    }
    write_json(output_dir / "technical_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
