#!/usr/bin/env python3
"""Executa overfit de sanidade em poucas imagens, exclusivamente do treino."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--config", type=Path, default=Path("configs/overfit_smoke.yaml"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stabilize-from",
        type=Path,
        default=None,
        help="Inicia uma nova execução a partir de um best.pt, sem sobrescrever a origem.",
    )
    parser.add_argument(
        "--stabilization-epochs",
        type=int,
        default=30,
        help="Número máximo de épocas da nova execução de estabilização.",
    )
    parser.add_argument(
        "--stabilization-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate usado após carregar o checkpoint de origem.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_type") != "overfit_smoke":
        raise ValueError("Este comando aceita somente experiment_type=overfit_smoke")
    if config.get("scientific_result") is not False:
        raise ValueError("O smoke overfit deve ser marcado scientific_result=false")
    data = config["data"]
    if data.get("split") != "train":
        raise ValueError("O overfit aceita exclusivamente split=train")
    manifest_name = Path(str(data["manifest"])).name.casefold()
    if manifest_name != "dfuc_train.csv":
        raise ValueError("O overfit aceita exclusivamente dfuc_train.csv")
    if bool(data.get("augmentation")):
        raise ValueError("Augmentation deve permanecer desativada no overfit")
    protection = config["protection"]
    if any(bool(protection[key]) for key in ("allow_validation", "allow_test", "allow_external")):
        raise ValueError("Particoes nao destinadas ao overfit devem permanecer bloqueadas")
    samples = int(data["samples"])
    if not 2 <= samples <= 8:
        raise ValueError("O overfit deve usar entre 2 e 8 amostras")


def gate_passed(max_target_streak: int, target_patience: int) -> bool:
    """Aprova somente quando a meta foi sustentada pelo número exigido de épocas."""
    return max_target_streak >= target_patience


def streaks_from_history(history_path: Path, target_dice: float) -> tuple[int, int]:
    """Recupera sequência atual e máxima para uma retomada fiel da execução."""
    if not history_path.exists():
        return 0, 0
    current_streak = 0
    max_streak = 0
    with history_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            score = float(row["memorization_dice_mean"])
            current_streak = current_streak + 1 if score >= target_dice else 0
            max_streak = max(max_streak, current_streak)
    return current_streak, max_streak


def history_row(
    epoch: int, elapsed: float, train_result, memorization_result, learning_rate: float
):
    return {
        "epoch": epoch,
        "elapsed_seconds": round(elapsed, 4),
        "learning_rate": learning_rate,
        "train_loss_total": train_result["loss"]["total"],
        "train_loss_bce": train_result["loss"]["bce"],
        "train_loss_dice": train_result["loss"]["dice_loss"],
        "memorization_loss_total": memorization_result["loss"]["total"],
        "memorization_dice_mean": memorization_result["metrics"]["dice"]["mean"],
        "memorization_dice_min": memorization_result["metrics"]["dice"]["min"],
        "memorization_iou_mean": memorization_result["metrics"]["iou"]["mean"],
        "memorization_precision_mean": memorization_result["metrics"]["precision"]["mean"],
        "memorization_recall_mean": memorization_result["metrics"]["recall"]["mean"],
    }


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.stabilize_from is not None:
        raise ValueError("Use apenas --resume ou --stabilize-from, nunca ambos")
    if args.stabilization_epochs < 1:
        raise ValueError("--stabilization-epochs deve ser positivo")
    if args.stabilization_learning_rate <= 0:
        raise ValueError("--stabilization-learning-rate deve ser positivo")
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else project_root
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    sys.path.insert(0, str(project_root / "src"))

    import torch

    from tcc_segmentation.data import (
        DevelopmentDataset,
        ResizePad,
        create_development_dataloader,
    )
    from tcc_segmentation.models import build_segmentation_model, count_parameters
    from tcc_segmentation.training import (
        BCEDiceLoss,
        evaluate_epoch,
        load_training_checkpoint,
        save_training_checkpoint,
        seed_everything,
        train_epoch,
    )
    from tcc_segmentation.training.reporting import (
        append_history,
        write_json,
        write_per_image_csv,
    )
    from tcc_segmentation.training.visualization import save_overfit_panel

    training_config = config["training"]
    seed = int(training_config["seed"])
    seed_everything(seed, bool(training_config["deterministic_algorithms"]))
    if bool(training_config["require_cuda"]) and not torch.cuda.is_available():
        raise RuntimeError("CUDA obrigatoria para este smoke test, mas nao foi encontrada")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(training_config["mixed_precision"] and device.type == "cuda")

    data_config = config["data"]
    manifest_path = project_root / str(data_config["manifest"])
    transform = ResizePad(height=int(data_config["height"]), width=int(data_config["width"]))
    full_train_dataset = DevelopmentDataset(
        manifest_path,
        data_root,
        expected_split="train",
        transform=transform,
        mask_threshold=int(data_config["mask_threshold"]),
    )
    selected_indices = list(range(int(data_config["samples"])))
    selected_dataset = torch.utils.data.Subset(full_train_dataset, selected_indices)
    selected_ids = [full_train_dataset.rows[index]["canonical_id"] for index in selected_indices]
    train_loader = create_development_dataloader(
        selected_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(data_config["num_workers"]),
    )
    memorization_loader = create_development_dataloader(
        selected_dataset,
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
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manifest_hashes = {"train": sha256_file(manifest_path)}

    source_checkpoint = None
    source_run_id = None
    stabilization_mode = args.stabilize_from is not None
    effective_stabilization_epochs = int(args.stabilization_epochs)
    effective_stabilization_learning_rate = float(args.stabilization_learning_rate)
    if args.resume is not None:
        resume_path = args.resume
        if not resume_path.is_absolute():
            resume_path = project_root / resume_path
        output_dir = resume_path.parent.parent
        payload = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            restore_rng=True,
        )
        if payload["manifest_hashes"] != manifest_hashes:
            raise RuntimeError("Hash do manifesto difere do checkpoint")
        run_id = str(payload["run_id"])
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        runtime = payload.get("extra", {}).get("runtime", {})
        stabilization_mode = bool(runtime.get("stabilization", False))
        if stabilization_mode:
            source_checkpoint = Path(runtime["source_checkpoint"])
            source_run_id = runtime["source_run_id"]
            effective_stabilization_epochs = int(runtime["max_epochs"])
            effective_stabilization_learning_rate = float(runtime["learning_rate"])
    elif stabilization_mode:
        source_checkpoint = args.stabilize_from
        if not source_checkpoint.is_absolute():
            source_checkpoint = project_root / source_checkpoint
        payload = load_training_checkpoint(
            source_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
            restore_rng=False,
        )
        if payload["manifest_hashes"] != manifest_hashes:
            raise RuntimeError("Hash do manifesto difere do checkpoint de estabilizacao")
        source_run_id = str(payload["run_id"])
        for group in optimizer.param_groups:
            group["lr"] = effective_stabilization_learning_rate
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"overfit_stabilization_seed{seed}_{timestamp}"
        output_dir = project_root / str(config["output"]["root"]) / run_id
        start_epoch = 1
        best_score = float("-inf")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"overfit_smoke_seed{seed}_{timestamp}"
        output_dir = project_root / str(config["output"]["root"]) / run_id
        start_epoch = 1
        best_score = float("-inf")

    checkpoints_dir = output_dir / "checkpoints"
    per_image_dir = output_dir / "per_image"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_snapshot.json", config)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "overfit_smoke",
        "scientific_result": False,
        "project_root": str(project_root),
        "data_root": str(data_root),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "amp": amp,
        "selected_indices": selected_indices,
        "selected_canonical_ids": selected_ids,
        "parameters": count_parameters(model),
        "manifest_hashes": manifest_hashes,
        "config_sha256": sha256_file(config_path),
        "protected_partitions_accessed": False,
        "stabilization": {
            "enabled": stabilization_mode,
            "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
            "source_run_id": source_run_id,
            "learning_rate": (
                effective_stabilization_learning_rate if stabilization_mode else None
            ),
            "max_epochs": effective_stabilization_epochs if stabilization_mode else None,
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    threshold = float(training_config["threshold_diagnostic"])
    initial_result = evaluate_epoch(
        model=model,
        loader=memorization_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        threshold=threshold,
    )
    if start_epoch == 1:
        write_json(
            output_dir / "initial_metrics.json",
            {"loss": initial_result["loss"], "metrics": initial_result["metrics"]},
        )
        write_per_image_csv(per_image_dir / "epoch_0000.csv", initial_result["per_image"])

    max_epochs = (
        effective_stabilization_epochs if stabilization_mode else int(training_config["max_epochs"])
    )
    target_dice = float(training_config["target_dice"])
    target_patience = int(training_config["target_patience_epochs"])
    target_streak, max_target_streak = (
        streaks_from_history(output_dir / "history.csv", target_dice)
        if args.resume is not None
        else (0, 0)
    )
    started = time.perf_counter()
    last_epoch = start_epoch - 1

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
        memorization_result = evaluate_epoch(
            model=model,
            loader=memorization_loader,
            criterion=criterion,
            device=device,
            amp=amp,
            threshold=threshold,
        )
        score = float(memorization_result["metrics"]["dice"]["mean"])
        best_min_delta = float(config["output"]["best_min_delta"])
        target_reached_now = score >= target_dice
        improved = score > best_score + best_min_delta or (
            target_reached_now and score > best_score
        )
        if improved:
            best_score = score
        target_streak = target_streak + 1 if score >= target_dice else 0
        max_target_streak = max(max_target_streak, target_streak)
        last_epoch = epoch
        row = history_row(
            epoch,
            time.perf_counter() - epoch_started,
            train_result,
            memorization_result,
            optimizer.param_groups[0]["lr"],
        )
        append_history(output_dir / "history.csv", row)
        extra = {
            "memorization_metrics": memorization_result["metrics"],
            "loss": memorization_result["loss"],
            "gradient_diagnostics": train_result["gradient_diagnostics"],
            "selected_canonical_ids": selected_ids,
            "scientific_result": False,
            "runtime": {
                "stabilization": stabilization_mode,
                "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
                "source_run_id": source_run_id,
                "learning_rate": (
                    effective_stabilization_learning_rate
                    if stabilization_mode
                    else float(optimizer.param_groups[0]["lr"])
                ),
                "max_epochs": max_epochs,
            },
        }
        if epoch % int(config["output"]["save_every_epochs"]) == 0:
            save_training_checkpoint(
                checkpoints_dir / "last.pt",
                kind="last",
                epoch=epoch,
                best_score=best_score,
                model=model,
                optimizer=optimizer,
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
                scaler=scaler,
                config=config,
                run_id=run_id,
                manifest_hashes=manifest_hashes,
                extra=extra,
            )
        per_image_frequency = int(config["output"]["write_per_image_every_epochs"])
        if epoch % per_image_frequency == 0 or improved or target_streak >= target_patience:
            write_per_image_csv(
                per_image_dir / f"epoch_{epoch:04d}.csv",
                memorization_result["per_image"],
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "loss": row["memorization_loss_total"],
                    "dice_mean": score,
                    "dice_min": row["memorization_dice_min"],
                    "best_dice": best_score,
                    "target_streak": target_streak,
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if target_streak >= target_patience:
            break

    best_payload = load_training_checkpoint(
        checkpoints_dir / "best.pt",
        model=model,
        map_location=device,
        restore_rng=False,
    )
    final_result = evaluate_epoch(
        model=model,
        loader=memorization_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        threshold=threshold,
    )
    write_per_image_csv(per_image_dir / "best_reloaded.csv", final_result["per_image"])
    panel_path = output_dir / "overfit_panel.png"
    save_overfit_panel(
        model=model,
        dataset=selected_dataset,
        device=device,
        output_path=panel_path,
        threshold=threshold,
    )
    final_dice = float(final_result["metrics"]["dice"]["mean"])
    passed = gate_passed(max_target_streak, target_patience)
    summary = {
        "run_id": run_id,
        "status": "passed" if passed else "needs_review",
        "scientific_result": False,
        "epochs_completed": last_epoch,
        "best_checkpoint_epoch": int(best_payload["epoch"]),
        "best_checkpoint_reloaded": True,
        "best_score_recorded": float(best_payload["best_score"]),
        "reloaded_metrics": final_result["metrics"],
        "reloaded_loss": final_result["loss"],
        "target_dice": target_dice,
        "target_patience_epochs": target_patience,
        "final_target_streak": target_streak,
        "max_target_streak": max_target_streak,
        "target_achieved": passed,
        "best_checkpoint_above_target": final_dice >= target_dice,
        "stabilization": metadata["stabilization"],
        "total_elapsed_seconds": time.perf_counter() - started,
        "panel": str(panel_path.relative_to(project_root).as_posix()),
        "protected_partitions_accessed": False,
    }
    if device.type == "cuda":
        summary["cuda_peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    write_json(output_dir / "final_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
