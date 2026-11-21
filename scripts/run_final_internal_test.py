#!/usr/bin/env python3
"""Executa uma única avaliação bloqueada nas 294 unidades do teste interno."""

from __future__ import annotations

import argparse
import csv
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
    root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    sys.path.insert(0, str(root / "src"))

    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageFont

    from tcc_segmentation.data import ComposePair, ImageOnlyPairTransform, ResizePad, shades_of_gray
    from tcc_segmentation.data.manifest import resolve_dataset_path
    from tcc_segmentation.data.masks import load_canonical_mask
    from tcc_segmentation.sam_refinement import (
        canvas_to_original,
        original_mask_to_canvas,
        refine_prediction_with_sam,
    )
    from tcc_segmentation.final_evaluation import (
        CONFIRMATION_TOKEN,
        aggregate_rows,
        apply_primary_holm,
        binary_metric_row,
        deterministic_panel_ids,
        load_final_test_manifest,
        paired_contrast,
        verify_preflight_lock,
    )
    from tcc_segmentation.final_protocol import sha256_file
    from tcc_segmentation.baseline_training import make_model, write_csv, write_json
    from tcc_segmentation.training import load_training_checkpoint, seed_everything

    if args.confirm != CONFIRMATION_TOKEN:
        raise PermissionError("Confirmação explícita da abertura do teste ausente")
    protocol, lock = verify_preflight_lock(root, args.protocol, args.preflight_lock)
    manifest_path = root / protocol["data"]["manifest"]
    manifest_rows = load_final_test_manifest(
        manifest_path,
        expected_sha256=protocol["data"]["manifest_sha256"],
        expected_records=int(protocol["data"]["expected_records"]),
    )
    canonical_ids = [row["canonical_id"] for row in manifest_rows]
    panel_ids = deterministic_panel_ids(canonical_ids, 12)

    run_id = f"FINAL_INTERNAL_TEST_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output = root / "results/final_evaluation" / run_id
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
            "panel_selection_rule": "12 IDs equally spaced after lexical sorting; performance-blind",
            "panel_ids": panel_ids,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(protocol["representative_run_policy"]["seed"]))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def load_sample(row, *, p1: bool, imagenet: bool):
        image_path = resolve_dataset_path(data_root, row["image_path"])
        mask_paths = [
            resolve_dataset_path(data_root, value) for value in row["mask_paths"].split("|")
        ]
        with Image.open(image_path) as source:
            original = np.asarray(source.convert("RGB"), dtype=np.uint8)
        mask = load_canonical_mask(mask_paths, policy=row["mask_policy"], threshold=128)
        transform = ResizePad(
            int(protocol["data"]["image_height"]), int(protocol["data"]["image_width"])
        )
        if p1:
            transform = ComposePair(
                [ImageOnlyPairTransform(lambda value: shades_of_gray(value, 6.0)), transform]
            )
        image, mask = transform(original, mask)
        tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        if imagenet:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
            tensor = (tensor - mean) / std
        return original, image, mask.astype(bool), np.ascontiguousarray(tensor)

    arm_by_id = {arm["id"]: arm for arm in protocol["arms"]}
    network_arms = [arm for arm in protocol["arms"] if arm["id"] != "DRIFT_SAM_BPM"]
    all_rows: list[dict] = []
    visuals: dict[str, dict] = {item: {} for item in panel_ids}
    sam_predictor = None
    sam_config = yaml.safe_load(
        (root / arm_by_id["DRIFT_SAM_BPM"]["config"]).read_text(encoding="utf-8")
    )
    started = time.perf_counter()

    for arm_index, arm in enumerate(network_arms):
        run_dir = root / arm["run_directory"]
        used_config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
        model_config = dict(used_config)
        model_config["model"] = dict(used_config["model"])
        model_config["model"]["encoder_weights"] = None
        model = make_model(model_config).to(device)
        checkpoint_path = run_dir / arm["checkpoint"]
        payload = load_training_checkpoint(
            checkpoint_path, model=model, map_location=device, restore_rng=False
        )
        if payload.get("run_id") != arm["run_id"]:
            raise RuntimeError(f"run_id interno do checkpoint diverge para {arm['id']}")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if arm["id"] == "DRIFT":
            try:
                from segment_anything import SamPredictor, sam_model_registry
            except ImportError as error:
                raise RuntimeError("segment-anything não instalado") from error
            sam_arm = arm_by_id["DRIFT_SAM_BPM"]
            sam_model = sam_model_registry["vit_b"](
                checkpoint=str(root / sam_arm["sam_checkpoint"])
            ).to(device)
            sam_model.eval()
            for parameter in sam_model.parameters():
                parameter.requires_grad_(False)
            sam_predictor = SamPredictor(sam_model)

        for row_index, manifest_row in enumerate(manifest_rows):
            p1 = arm["preprocessing"].startswith("P1")
            imagenet = arm["normalization"] == "imagenet"
            original, rgb_canvas, reference, image_array = load_sample(
                manifest_row, p1=p1, imagenet=imagenet
            )
            input_tensor = torch.as_tensor(image_array)[None].to(device)
            with torch.inference_mode():
                probability = torch.sigmoid(model(input_tensor).float())[0, 0].cpu().numpy()
            policies = (("selected", float(arm["threshold"])), ("fixed_0_5", 0.5))
            for policy, threshold in policies:
                prediction = probability >= threshold
                all_rows.append(
                    {
                        "canonical_id": manifest_row["canonical_id"],
                        "foreground_area_bin": int(manifest_row["foreground_area_bin"]),
                        "foreground_fraction": float(manifest_row["foreground_fraction"]),
                        "arm_id": arm["id"],
                        "threshold_policy": policy,
                        "threshold": threshold,
                        "reference_pixels": int(reference.sum()),
                        "prediction_pixels": int(prediction.sum()),
                        **binary_metric_row(prediction, reference),
                    }
                )

            if arm["id"] == "DRIFT":
                if sam_predictor is None:
                    raise RuntimeError("SAM não inicializado")
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
                    network_threshold=float(arm["threshold"]),
                    mode="box_plus_mask",
                    max_components_safety=int(sam_config["prompts"]["max_components_safety"]),
                    sam_image_encoder_size=int(sam_config["sam"]["image_encoder_size"]),
                    mask_input_size=int(sam_config["prompts"]["mask_input_size"][0]),
                    outside_logit=float(sam_config["prompts"]["outside_logit"]),
                )
                if audit["ground_truth_used_for_prompt"]:
                    raise RuntimeError("Referência usada indevidamente no prompt SAM")
                sam_canvas = original_mask_to_canvas(
                    sam_original, target_height=reference.shape[0], target_width=reference.shape[1]
                )
                all_rows.append(
                    {
                        "canonical_id": manifest_row["canonical_id"],
                        "foreground_area_bin": int(manifest_row["foreground_area_bin"]),
                        "foreground_fraction": float(manifest_row["foreground_fraction"]),
                        "arm_id": "DRIFT_SAM_BPM",
                        "threshold_policy": "selected",
                        "threshold": float(arm["threshold"]),
                        "reference_pixels": int(reference.sum()),
                        "prediction_pixels": int(sam_canvas.sum()),
                        **binary_metric_row(sam_canvas, reference),
                    }
                )
                if manifest_row["canonical_id"] in visuals:
                    visuals[manifest_row["canonical_id"]].update(
                        {
                            "rgb": rgb_canvas,
                            "reference": reference,
                            "probability": probability,
                            "DRIFT": probability >= float(arm["threshold"]),
                            "DRIFT_SAM_BPM": sam_canvas,
                        }
                    )
            elif arm["id"] == "DRIFTB" and manifest_row["canonical_id"] in visuals:
                visuals[manifest_row["canonical_id"]]["DRIFTB"] = probability >= float(
                    arm["threshold"]
                )

            completed = row_index + 1
            if completed == 1 or completed % 25 == 0 or completed == len(manifest_rows):
                print(
                    json.dumps(
                        {
                            "arm": arm["id"],
                            "arm_number": arm_index + 1,
                            "arms_total": len(network_arms),
                            "completed": completed,
                            "total": len(manifest_rows),
                        }
                    ),
                    flush=True,
                )
        del model
        if arm["id"] == "DRIFT":
            del sam_predictor, sam_model
            sam_predictor = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    expected_rows = len(network_arms) * len(manifest_rows) * 2 + len(manifest_rows)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Quantidade de linhas inesperada: {len(all_rows)} != {expected_rows}")
    write_csv(output / "test_per_image_all_arms.csv", all_rows)
    summaries = aggregate_rows(all_rows)
    write_csv(output / "test_summary_all_arms.csv", summaries)

    stats = protocol["statistics"]
    comparison_rows: list[dict] = []
    families = (
        ("primary", protocol["primary_contrasts"]),
        ("secondary", protocol["secondary_planned_contrasts"]),
        ("exploratory", protocol["exploratory_contrasts"]),
    )
    contrast_index = 0
    for family, contrasts in families:
        for contrast in contrasts:
            comparison_rows.extend(
                paired_contrast(
                    all_rows,
                    contrast_id=contrast["id"],
                    candidate=contrast["candidate"],
                    control=contrast["control"],
                    threshold_policy="selected",
                    family=family,
                    bootstrap_samples=int(stats["bootstrap_samples"]),
                    permutation_samples=int(stats["permutation_samples"]),
                    seed=int(stats["random_seed"]) + contrast_index * 100,
                )
            )
            contrast_index += 1
    apply_primary_holm(comparison_rows)

    for contrast_index, contrast in enumerate(protocol["primary_contrasts"]):
        if "SAM" in contrast["id"]:
            continue
        comparison_rows.extend(
            paired_contrast(
                all_rows,
                contrast_id=contrast["id"],
                candidate=contrast["candidate"],
                control=contrast["control"],
                threshold_policy="fixed_0_5",
                family="sensitivity",
                bootstrap_samples=int(stats["bootstrap_samples"]),
                permutation_samples=int(stats["permutation_samples"]),
                seed=int(stats["random_seed"]) + 50_000 + contrast_index * 100,
            )
        )
    write_csv(output / "paired_comparisons.csv", comparison_rows)

    def font(size: int):
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

    columns = (
        ("imagem", "rgb"),
        ("referência", "reference"),
        ("prob. DRIFT", "probability"),
        ("DRIFT", "DRIFT"),
        ("DRIFT+SAM", "DRIFT_SAM_BPM"),
        ("DRIFTB", "DRIFTB"),
    )
    tile_w, tile_h, label_h = 384, 288, 24
    panel = Image.new("RGB", (tile_w * len(columns), (tile_h + label_h) * len(panel_ids)), "white")
    draw = ImageDraw.Draw(panel)
    for row_number, canonical_id in enumerate(panel_ids):
        item = visuals[canonical_id]
        if set(item) != {"rgb", "reference", "probability", "DRIFT", "DRIFT_SAM_BPM", "DRIFTB"}:
            raise RuntimeError(f"Painel incompleto para {canonical_id}")
        for column_number, (title, key) in enumerate(columns):
            if key == "rgb":
                tile = item[key]
            elif key == "probability":
                gray = np.clip(item[key] * 255, 0, 255).astype(np.uint8)
                tile = np.repeat(gray[..., None], 3, axis=2)
            elif key == "reference":
                tile = overlay(item["rgb"], item[key], (235, 40, 40))
            else:
                tile = overlay(item["rgb"], item[key], (20, 200, 50))
            x, y = column_number * tile_w, row_number * (tile_h + label_h)
            panel.paste(
                Image.fromarray(tile).resize((tile_w, tile_h), Image.Resampling.BILINEAR),
                (x, y + label_h),
            )
            draw.text((x + 4, y + 4), f"{canonical_id} | {title}", fill="black", font=font(12))
    panel.save(output / "fixed_test_panel_performance_blind.png")

    report = {
        "status": "completed",
        "run_id": run_id,
        "partition": "internal_test",
        "records": len(manifest_rows),
        "network_arms": len(network_arms),
        "derived_sam_arm": True,
        "threshold_optimization_performed": False,
        "checkpoint_selection_performed": False,
        "model_selection_performed": False,
        "all_arms_completed_before_summary": True,
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
            "per_image": "test_per_image_all_arms.csv",
            "summary": "test_summary_all_arms.csv",
            "comparisons": "paired_comparisons.csv",
            "panel": "fixed_test_panel_performance_blind.png",
        },
    }
    write_json(output / "final_test_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaídas finais: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
