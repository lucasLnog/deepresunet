#!/usr/bin/env python3
"""Treina um baseline usando exclusivamente os manifestos de treino e validação."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_baseline_config(config: dict[str, Any]) -> None:
    if config.get("experiment_type") != "development_baseline":
        raise ValueError("experiment_type deve ser development_baseline")
    if config.get("scientific_result") is not True:
        raise ValueError("Baseline de desenvolvimento deve marcar scientific_result=true")
    data = config["data"]
    expected = {
        "train_manifest": "dfuc_train.csv",
        "validation_manifest": "dfuc_validation.csv",
    }
    for field, filename in expected.items():
        if Path(str(data[field])).name.casefold() != filename:
            raise ValueError(f"{field} deve apontar exclusivamente para {filename}")
    if bool(data.get("augmentation")):
        raise ValueError("Os baselines iniciais devem permanecer sem augmentation")
    if data.get("preprocessing") != "resize_pad_scale_0_1":
        raise ValueError("Pré-processamento do baseline inicial foi alterado")
    if any(bool(value) for value in config["protection"].values()):
        raise ValueError("Teste e avaliação externa devem permanecer bloqueados")
    training = config["training"]
    if int(training["max_epochs"]) < 1:
        raise ValueError("max_epochs deve ser positivo")
    if int(training["early_stopping_patience"]) < 1:
        raise ValueError("early_stopping_patience deve ser positivo")
    if float(training["threshold_diagnostic"]) != 0.5:
        raise ValueError("Checkpoint dos baselines deve usar threshold diagnóstico 0,5")
    scheduler = config["scheduler"]
    if scheduler.get("monitor") != "validation_dice_mean" or scheduler.get("mode") != "max":
        raise ValueError("Scheduler deve monitorar validation_dice_mean em modo max")


def history_row(epoch: int, elapsed: float, train_result, validation_result, lr: float):
    return {
        "epoch": epoch,
        "elapsed_seconds": round(elapsed, 4),
        "learning_rate": lr,
        "train_loss_total": train_result["loss"]["total"],
        "train_loss_bce": train_result["loss"]["bce"],
        "train_loss_dice": train_result["loss"]["dice_loss"],
        "train_dice_mean": train_result["metrics"]["dice"]["mean"],
        "validation_loss_total": validation_result["loss"]["total"],
        "validation_loss_bce": validation_result["loss"]["bce"],
        "validation_loss_dice": validation_result["loss"]["dice_loss"],
        "validation_dice_mean": validation_result["metrics"]["dice"]["mean"],
        "validation_dice_median": validation_result["metrics"]["dice"]["median"],
        "validation_iou_mean": validation_result["metrics"]["iou"]["mean"],
        "validation_precision_mean": validation_result["metrics"]["precision"]["mean"],
        "validation_recall_mean": validation_result["metrics"]["recall"]["mean"],
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else project_root
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_baseline_config(config)
    sys.path.insert(0, str(project_root / "src"))

    import torch

    from tcc_segmentation.data import DevelopmentDataset, ResizePad, create_development_dataloader
    from tcc_segmentation.models import build_segmentation_model, count_parameters
    from tcc_segmentation.training import (
        BCEDiceLoss,
        evaluate_epoch,
        load_training_checkpoint,
        save_training_checkpoint,
        seed_everything,
        train_epoch,
    )
    from tcc_segmentation.training.reporting import append_history, write_json, write_per_image_csv

    training_config = config["training"]
    seed = int(training_config["seed"])
    seed_everything(seed, bool(training_config["deterministic_algorithms"]))
    if bool(training_config["require_cuda"]) and not torch.cuda.is_available():
        raise RuntimeError("CUDA obrigatória para o baseline completo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(training_config["mixed_precision"] and device.type == "cuda")

    data_config = config["data"]
    train_manifest = project_root / str(data_config["train_manifest"])
    validation_manifest = project_root / str(data_config["validation_manifest"])
    transform = ResizePad(height=int(data_config["height"]), width=int(data_config["width"]))
    train_dataset = DevelopmentDataset(
        train_manifest,
        data_root,
        expected_split="train",
        transform=transform,
        mask_threshold=int(data_config["mask_threshold"]),
    )
    validation_dataset = DevelopmentDataset(
        validation_manifest,
        data_root,
        expected_split="validation",
        transform=transform,
        mask_threshold=int(data_config["mask_threshold"]),
    )
    train_loader = create_development_dataloader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(data_config["num_workers"]),
    )
    validation_loader = create_development_dataloader(
        validation_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(data_config["num_workers"]),
    )

    model = build_segmentation_model(config["model"]).to(device)
    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        smooth=float(config["loss"]["smooth"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler_config["mode"]),
        factor=float(scheduler_config["factor"]),
        patience=int(scheduler_config["patience"]),
        threshold=float(scheduler_config["threshold"]),
        threshold_mode=str(scheduler_config["threshold_mode"]),
        min_lr=float(scheduler_config["min_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manifest_hashes = {
        "train": sha256_file(train_manifest),
        "validation": sha256_file(validation_manifest),
    }

    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else project_root / args.resume
        output_dir = resume_path.parent.parent
        payload = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
            restore_rng=True,
        )
        if payload["manifest_hashes"] != manifest_hashes:
            raise RuntimeError("Hashes dos manifestos diferem do checkpoint")
        if payload["config"] != config:
            raise RuntimeError("Configuração atual difere do checkpoint")
        run_id = str(payload["run_id"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        early_state = payload.get("extra", {}).get("early_stopping", {})
        early_best = float(early_state.get("best", best_score))
        epochs_without_material_improvement = int(early_state.get("epochs_without_improvement", 0))
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{config['experiment_id']}_seed{seed}_{timestamp}"
        output_dir = project_root / str(config["output"]["root"]) / run_id
        start_epoch = 1
        best_score = float("-inf")
        early_best = float("-inf")
        epochs_without_material_improvement = 0

    checkpoints_dir = output_dir / "checkpoints"
    per_image_dir = output_dir / "per_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": config["experiment_type"],
        "experiment_id": config["experiment_id"],
        "scientific_result": True,
        "development_only": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp,
        "parameters": count_parameters(model),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "manifest_hashes": manifest_hashes,
        "config_sha256": sha256_file(config_path),
        "protected_partitions_accessed": False,
    }
    write_json(output_dir / "run_metadata.json", metadata)

    max_epochs = int(training_config["max_epochs"])
    early_patience = int(training_config["early_stopping_patience"])
    early_min_delta = float(training_config["early_stopping_min_delta"])
    threshold = float(training_config["threshold_diagnostic"])
    last_epoch = start_epoch - 1
    termination_reason = "max_epochs"
    started = time.perf_counter()

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_started = time.perf_counter()
        train_result = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=amp,
            threshold=threshold,
            gradient_clip_norm=training_config.get("gradient_clip_norm"),
        )
        validation_result = evaluate_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            threshold=threshold,
        )
        score = float(validation_result["metrics"]["dice"]["mean"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        row = history_row(
            epoch,
            time.perf_counter() - epoch_started,
            train_result,
            validation_result,
            current_lr,
        )
        append_history(output_dir / "history.csv", row)
        improved = score > best_score
        if improved:
            best_score = score
        if score > early_best + early_min_delta:
            early_best = score
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1
        scheduler.step(score)
        last_epoch = epoch
        extra = {
            "validation_metrics": validation_result["metrics"],
            "validation_loss": validation_result["loss"],
            "gradient_diagnostics": train_result["gradient_diagnostics"],
            "early_stopping": {
                "best": early_best,
                "epochs_without_improvement": epochs_without_material_improvement,
            },
            "protected_partitions_accessed": False,
        }
        if epoch % int(config["output"]["save_last_every_epochs"]) == 0:
            save_training_checkpoint(
                checkpoints_dir / "last.pt",
                kind="last",
                epoch=epoch,
                best_score=best_score,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                run_id=run_id,
                manifest_hashes=manifest_hashes,
                extra=extra,
            )
        if improved:
            save_training_checkpoint(
                checkpoints_dir / "best.pt",
                kind="best",
                epoch=epoch,
                best_score=best_score,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                run_id=run_id,
                manifest_hashes=manifest_hashes,
                extra=extra,
            )
            write_per_image_csv(
                per_image_dir / "best_validation.csv", validation_result["per_image"]
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": row["train_loss_total"],
                    "validation_loss": row["validation_loss_total"],
                    "validation_dice": score,
                    "best_validation_dice": best_score,
                    "learning_rate": current_lr,
                    "epochs_without_material_improvement": epochs_without_material_improvement,
                    "amp_skipped_optimizer_steps": train_result["gradient_diagnostics"].get(
                        "amp_skipped_optimizer_steps", 0
                    ),
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epochs_without_material_improvement >= early_patience:
            termination_reason = "early_stopping"
            break

    best_payload = load_training_checkpoint(
        checkpoints_dir / "best.pt",
        model=model,
        map_location=device,
        restore_rng=False,
    )
    best_validation = evaluate_epoch(
        model=model,
        loader=validation_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        threshold=threshold,
    )
    write_per_image_csv(
        per_image_dir / "best_validation_reloaded.csv", best_validation["per_image"]
    )
    summary = {
        "run_id": run_id,
        "status": "completed",
        "scientific_result": True,
        "development_only": True,
        "experiment_id": config["experiment_id"],
        "epochs_completed": last_epoch,
        "termination_reason": termination_reason,
        "best_checkpoint_epoch": int(best_payload["epoch"]),
        "best_checkpoint_reloaded": True,
        "selection_metric": "validation_dice_mean_at_threshold_0.5",
        "best_validation_metrics_at_0.5": best_validation["metrics"],
        "best_validation_loss": best_validation["loss"],
        "threshold_selection_pending": True,
        "total_elapsed_seconds": time.perf_counter() - started,
        "protected_partitions_accessed": False,
    }
    if device.type == "cuda":
        summary["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
