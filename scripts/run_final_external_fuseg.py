#!/usr/bin/env python3
"""Executa uma unica avaliacao externa bloqueada nas unidades canonicas FUSeg."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-lock", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root, data_root = args.project_root.resolve(), args.data_root.resolve()
    sys.path.insert(0, str(root / "src"))

    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageFont

    from tcc_segmentation.data import ResizePad
    from tcc_segmentation.data.manifest import resolve_dataset_path
    from tcc_segmentation.data.masks import load_binary_mask
    from tcc_segmentation.sam_refinement import (
        canvas_to_original,
        original_mask_to_canvas,
        refine_prediction_with_sam,
    )
    from tcc_segmentation.baseline_training import make_model, write_csv, write_json
    from tcc_segmentation.external_evaluation import (
        CONFIRMATION_TOKEN,
        aggregate_rows_with_ci,
        average_reference_metric_rows,
        external_sam_contrast,
        load_external_manifest,
        verify_external_preflight_lock,
    )
    from tcc_segmentation.external_protocol import sha256_file
    from tcc_segmentation.final_evaluation import binary_metric_row, deterministic_panel_ids
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    if args.confirm != CONFIRMATION_TOKEN:
        raise PermissionError("Confirmacao explicita da abertura do FUSeg ausente")
    protocol, _ = verify_external_preflight_lock(root, args.protocol, args.preflight_lock)
    manifest_path = root / protocol["data"]["manifest"]
    manifest_rows = load_external_manifest(
        manifest_path,
        expected_sha256=protocol["data"]["manifest_sha256"],
        expected_units=int(protocol["data"]["expected_canonical_units"]),
    )
    panel_ids = deterministic_panel_ids(
        [row["canonical_id"] for row in manifest_rows],
        int(protocol["visualization"]["panel_units"]),
    )

    run_id = f"FINAL_EXTERNAL_FUSEG_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = root / "results/final_external_fuseg" / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "execution_identity.json",
        {
            "run_id": run_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": sha256_file(args.protocol),
            "preflight_lock_sha256": sha256_file(args.preflight_lock),
            "manifest_sha256": sha256_file(manifest_path),
            "confirmation_token_received": True,
            "panel_selection_rule": protocol["visualization"]["selection_rule"],
            "panel_ids": panel_ids,
        },
    )

    primary, secondary = protocol["primary_model"], protocol["secondary_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(primary["seed"]))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    run_dir = root / primary["run_directory"]
    used_config = yaml.safe_load((root / primary["config"]).read_text(encoding="utf-8"))
    model_config = dict(used_config)
    model_config["model"] = dict(used_config["model"])
    model_config["model"]["encoder_weights"] = None
    model = make_model(model_config).to(device)
    payload = load_training_checkpoint(
        run_dir / primary["checkpoint"], model=model, map_location=device, restore_rng=False
    )
    if payload.get("run_id") != primary["run_id"]:
        raise RuntimeError("run_id interno do checkpoint DRIFT diverge do protocolo")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise RuntimeError("segment-anything nao instalado") from error
    sam_config = yaml.safe_load((root / secondary["config"]).read_text(encoding="utf-8"))
    sam_model = sam_model_registry["vit_b"](checkpoint=str(root / secondary["sam_checkpoint"])).to(
        device
    )
    sam_model.eval()
    for parameter in sam_model.parameters():
        parameter.requires_grad_(False)
    sam_predictor = SamPredictor(sam_model)

    target_height = int(protocol["data"]["image_height"])
    target_width = int(protocol["data"]["image_width"])
    transform = ResizePad(target_height, target_width)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    threshold = float(primary["threshold"])
    all_rows: list[dict] = []
    sam_audits: list[dict] = []
    visuals: dict[str, dict] = {}
    started = time.perf_counter()

    for row_index, row in enumerate(manifest_rows):
        image_path = resolve_dataset_path(data_root, row["image_path"])
        mask_paths = [
            resolve_dataset_path(data_root, value) for value in row["mask_paths"].split("|")
        ]
        with Image.open(image_path) as source:
            original = np.asarray(source.convert("RGB"), dtype=np.uint8)
        references = []
        rgb_canvas = None
        for mask_path in mask_paths:
            reference_original = load_binary_mask(mask_path, threshold=128)
            if reference_original.shape != original.shape[:2]:
                raise ValueError(f"Imagem e referencia incompativeis em {row['canonical_id']}")
            current_rgb, reference = transform(original, reference_original)
            if rgb_canvas is None:
                rgb_canvas = current_rgb
            elif not np.array_equal(rgb_canvas, current_rgb):
                raise RuntimeError("Transformacao RGB nao deterministica")
            references.append(reference.astype(bool))
        if rgb_canvas is None:
            raise ValueError(f"Unidade sem referencia: {row['canonical_id']}")

        image_array = rgb_canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        image_array = np.ascontiguousarray((image_array - mean) / std, dtype=np.float32)
        with torch.inference_mode():
            probability = (
                torch.sigmoid(model(torch.as_tensor(image_array)[None].to(device)).float())[0, 0]
                .cpu()
                .numpy()
            )
        drift_selected = probability >= threshold
        drift_fixed = probability >= 0.5

        probability_original = canvas_to_original(
            probability,
            original_height=original.shape[0],
            original_width=original.shape[1],
            interpolation="bilinear",
        )
        sam_original, audit = refine_prediction_with_sam(
            predictor=sam_predictor,
            rgb_image=original,
            network_probabilities=probability_original,
            network_threshold=threshold,
            mode="box_plus_mask",
            max_components_safety=int(sam_config["prompts"]["max_components_safety"]),
            sam_image_encoder_size=int(sam_config["sam"]["image_encoder_size"]),
            mask_input_size=int(sam_config["prompts"]["mask_input_size"][0]),
            outside_logit=float(sam_config["prompts"]["outside_logit"]),
        )
        if audit["ground_truth_used_for_prompt"]:
            raise RuntimeError("Referencia usada indevidamente no prompt SAM")
        sam_canvas = original_mask_to_canvas(
            sam_original, target_height=target_height, target_width=target_width
        )
        sam_audits.append(
            {
                "canonical_id": row["canonical_id"],
                "component_count": audit["component_count"],
                "empty_prediction_policy_applied": audit["empty_prediction_policy_applied"],
                "ground_truth_used_for_prompt": audit["ground_truth_used_for_prompt"],
            }
        )

        predictions = (
            ("DRIFT", "selected", threshold, drift_selected),
            ("DRIFT", "fixed_0_5", 0.5, drift_fixed),
            ("DRIFT_SAM_BPM", "selected", threshold, sam_canvas),
        )
        for arm_id, policy, applied_threshold, prediction in predictions:
            reference_rows = [binary_metric_row(prediction, reference) for reference in references]
            all_rows.append(
                {
                    "canonical_id": row["canonical_id"],
                    "arm_id": arm_id,
                    "threshold_policy": policy,
                    "threshold": applied_threshold,
                    "reference_count": len(references),
                    "foreground_fraction": float(row["foreground_fraction"]),
                    "reference_pixels_mean": float(
                        np.mean([reference.sum() for reference in references])
                    ),
                    "prediction_pixels": int(prediction.sum()),
                    **average_reference_metric_rows(
                        reference_rows, reference_count=len(references)
                    ),
                }
            )

        if row["canonical_id"] in panel_ids:
            visuals[row["canonical_id"]] = {
                "rgb": rgb_canvas,
                "reference_mean": np.mean(np.stack(references), axis=0),
                "probability": probability,
                "DRIFT": drift_selected,
                "DRIFT_SAM_BPM": sam_canvas,
            }
        completed = row_index + 1
        if completed == 1 or completed % 25 == 0 or completed == len(manifest_rows):
            print(json.dumps({"completed": completed, "total": len(manifest_rows)}), flush=True)

    expected_rows = len(manifest_rows) * 3
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Quantidade de linhas inesperada: {len(all_rows)} != {expected_rows}")
    write_csv(output / "external_per_unit.csv", all_rows)
    write_csv(output / "sam_prompt_audit.csv", sam_audits)
    stats = protocol["statistics"]
    summaries = aggregate_rows_with_ci(
        all_rows,
        bootstrap_samples=int(stats["bootstrap_samples"]),
        seed=int(stats["random_seed"]),
    )
    write_csv(output / "external_summary.csv", summaries)
    comparisons = external_sam_contrast(
        all_rows,
        bootstrap_samples=int(stats["bootstrap_samples"]),
        permutation_samples=int(stats["permutation_samples"]),
        seed=int(stats["random_seed"]) + 100_000,
    )
    write_csv(output / "external_sam_comparison.csv", comparisons)

    def font(size: int):
        for candidate in (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
        ):
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    def overlay(rgb, mask, color, alpha=0.55):
        result = rgb.astype(np.float32).copy()
        strength = np.asarray(mask, dtype=np.float32)[..., None] * alpha
        result = result * (1.0 - strength) + np.asarray(color, dtype=np.float32) * strength
        return np.clip(result, 0, 255).astype(np.uint8)

    columns = (
        ("imagem", "rgb"),
        ("referencia media", "reference_mean"),
        ("prob. DRIFT", "probability"),
        ("DRIFT", "DRIFT"),
        ("DRIFT+SAM", "DRIFT_SAM_BPM"),
    )
    tile_w, tile_h, label_h = 384, 288, 24
    panel = Image.new("RGB", (tile_w * len(columns), (tile_h + label_h) * len(panel_ids)), "white")
    draw = ImageDraw.Draw(panel)
    for row_number, canonical_id in enumerate(panel_ids):
        item = visuals.get(canonical_id)
        if item is None or set(item) != {
            "rgb",
            "reference_mean",
            "probability",
            "DRIFT",
            "DRIFT_SAM_BPM",
        }:
            raise RuntimeError(f"Painel externo incompleto para {canonical_id}")
        for column_number, (title, key) in enumerate(columns):
            if key == "rgb":
                tile = item[key]
            elif key == "probability":
                gray = np.clip(item[key] * 255, 0, 255).astype(np.uint8)
                tile = np.repeat(gray[..., None], 3, axis=2)
            elif key == "reference_mean":
                tile = overlay(item["rgb"], item[key], (235, 40, 40))
            else:
                tile = overlay(item["rgb"], item[key], (20, 200, 50))
            x, y = column_number * tile_w, row_number * (tile_h + label_h)
            panel.paste(
                Image.fromarray(tile).resize((tile_w, tile_h), Image.Resampling.BILINEAR),
                (x, y + label_h),
            )
            draw.text((x + 4, y + 4), f"{canonical_id} | {title}", fill="black", font=font(12))
    panel.save(output / "fixed_external_panel_performance_blind.png")

    report = {
        "status": "completed",
        "run_id": run_id,
        "partition": "external_FUSeg",
        "raw_records": int(protocol["data"]["raw_records"]),
        "excluded_records": int(protocol["data"]["excluded_records"]),
        "valid_reference_records": int(protocol["data"]["valid_reference_records"]),
        "canonical_units": len(manifest_rows),
        "duplicate_reference_policy": protocol["data"]["duplicate_reference_policy"],
        "threshold_optimization_performed": False,
        "checkpoint_selection_performed": False,
        "model_selection_performed": False,
        "fine_tuning_performed": False,
        "test_time_augmentation_performed": False,
        "all_predictions_completed_before_summary": True,
        "elapsed_seconds": time.perf_counter() - started,
        "protocol_sha256": sha256_file(args.protocol),
        "preflight_lock_sha256": sha256_file(args.preflight_lock),
        "manifest_sha256": sha256_file(manifest_path),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "panel_selection_rule": "performance-blind deterministic lexical spacing",
        "panel_ids": panel_ids,
        "artifacts": {
            "per_unit": "external_per_unit.csv",
            "summary": "external_summary.csv",
            "sam_comparison": "external_sam_comparison.csv",
            "sam_prompt_audit": "sam_prompt_audit.csv",
            "panel": "fixed_external_panel_performance_blind.png",
        },
    }
    write_json(output / "final_external_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaidas finais: {output}")

    del model, sam_predictor, sam_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
