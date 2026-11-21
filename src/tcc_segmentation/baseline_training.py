"""Pipeline reprodutível do D0, limitado a treino e validação."""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .data import create_development_dataloader
from .models import build_segmentation_model, count_parameters
from .training import (
    BCEDiceLoss,
    BCEDicePerimeterLoss,
    evaluate_epoch,
    load_training_checkpoint,
    save_training_checkpoint,
    seed_everything,
    train_epoch,
)


COUNT_NAMES = ("tp", "fp", "fn", "tn")
METRIC_NAMES = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
)


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return destination


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("Não é possível escrever CSV vazio")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_criterion(config: Mapping[str, Any]):
    loss = config["loss"]
    name = str(loss["name"])
    common = {
        "bce_weight": float(loss["bce_weight"]),
        "dice_weight": float(loss["dice_weight"]),
        "smooth": float(loss["smooth"]),
    }
    if name == "bce_with_logits_plus_soft_dice":
        return BCEDiceLoss(**common)
    if name == "bce_dice_plus_soft_perimeter":
        return BCEDicePerimeterLoss(
            **common,
            region_weight=float(loss["region_weight"]),
            perimeter_weight=float(loss["perimeter_weight"]),
            perimeter_epsilon=float(loss["perimeter_epsilon"]),
        )
    raise ValueError(f"Loss não reconhecida: {name}")


def make_model(config: Mapping[str, Any]):
    return build_segmentation_model(config["model"])


def make_optimizer(model, config: Mapping[str, Any]):
    training = config["training"]
    if str(training["optimizer"]).casefold() != "adam":
        raise ValueError("O protocolo controlado permite somente Adam nesta versão")
    return torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))


def make_scheduler(optimizer, config: Mapping[str, Any]):
    scheduler = config["training"]["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(scheduler["mode"]),
        factor=float(scheduler["factor"]),
        patience=int(scheduler["patience"]),
        min_lr=float(scheduler["min_learning_rate"]),
    )


def fixed_subset(dataset, canonical_ids: Sequence[str]):
    index_by_id = {str(row["canonical_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [value for value in canonical_ids if value not in index_by_id]
    if missing:
        raise ValueError(f"IDs fixos ausentes da validação: {missing}")
    return torch.utils.data.Subset(dataset, [index_by_id[value] for value in canonical_ids])


def preflight_batch_size(
    *,
    train_dataset,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    """Testa batch primário e fallback; o modelo usado aqui é descartado."""
    training = config["training"]
    candidates = [
        int(training["batch_size"]),
        int(training["oom_fallback_batch_size"]),
    ]
    candidates = list(dict.fromkeys(candidates))
    attempts: list[dict[str, Any]] = []
    for batch_size in candidates:
        seed_everything(int(config["seed"]))
        loader = create_development_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=int(config["seed"]),
            num_workers=0,
        )
        model = make_model(config).to(device)
        criterion = make_criterion(config)
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            batch = next(iter(loader))
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            model.train()
            logits = model(images)
            components = criterion.components(logits, masks)
            components["total"].backward()
            finite = bool(
                torch.isfinite(logits).all()
                and torch.isfinite(components["total"])
                and all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                )
            )
            if not finite:
                raise FloatingPointError("Preflight produziu valor não finito")
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
            attempts.append(
                {
                    "batch_size": batch_size,
                    "status": "passed",
                    "image_shape": list(images.shape),
                    "logit_shape": list(logits.shape),
                    "loss": float(components["total"].detach().cpu()),
                    "peak_memory_bytes": peak_memory,
                }
            )
            del model, loader, batch, images, masks, logits, components
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return batch_size, {
                "selected_batch_size": batch_size,
                "attempts": attempts,
                "protected_partitions_accessed": False,
            }
        except torch.cuda.OutOfMemoryError as error:
            attempts.append(
                {"batch_size": batch_size, "status": "cuda_out_of_memory", "error": str(error)}
            )
            del model, loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as error:
            if "out of memory" not in str(error).casefold():
                raise
            attempts.append(
                {"batch_size": batch_size, "status": "cuda_out_of_memory", "error": str(error)}
            )
            del model, loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError(f"Nenhum batch do preflight coube na memória: {attempts}")


def _history_row(
    epoch: int,
    train_result: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    learning_rate: float,
    elapsed_seconds: float,
    best_validation_dice: float,
    epochs_without_material_improvement: int,
) -> dict[str, Any]:
    row = {
        "epoch": epoch,
        "train_loss": float(train_result["loss"]["total"]),
        "train_bce": float(train_result["loss"]["bce"]),
        "train_dice_loss": float(train_result["loss"]["dice_loss"]),
        "validation_loss": float(validation_result["loss"]["total"]),
        "validation_bce": float(validation_result["loss"]["bce"]),
        "validation_dice_loss": float(validation_result["loss"]["dice_loss"]),
        "validation_dice_at_0_5": float(validation_result["metrics"]["dice"]["mean"]),
        "validation_iou_at_0_5": float(validation_result["metrics"]["iou"]["mean"]),
        "learning_rate": learning_rate,
        "best_validation_dice_at_0_5": best_validation_dice,
        "epochs_without_material_improvement": epochs_without_material_improvement,
        "elapsed_seconds": elapsed_seconds,
    }
    for name, value in train_result["loss"].items():
        key = f"train_{name}"
        if key not in row:
            row[key] = float(value)
    for name, value in validation_result["loss"].items():
        key = f"validation_{name}"
        if key not in row:
            row[key] = float(value)
    return row


def train_d0(
    *,
    train_dataset,
    validation_dataset,
    config: Mapping[str, Any],
    output_dir: str | Path,
    run_id: str,
    device: torch.device,
    manifest_hashes: Mapping[str, str],
    resume: bool = False,
    preflight_result: tuple[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Treina um braço controlado e salva best/last sem partições protegidas."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completed_summary_path = output / "training_summary.json"
    if resume and completed_summary_path.is_file():
        return json.loads(completed_summary_path.read_text(encoding="utf-8"))
    history_path = output / "history.csv"
    last_path = output / "last_checkpoint.pt"
    best_path = output / "best_checkpoint.pt"
    training = config["training"]
    threshold = float(training["checkpoint_threshold"])

    if resume and last_path.is_file():
        history = read_csv(history_path)
        selected_batch_size = int(history[0]["batch_size"])
        preflight_report = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    else:
        if history_path.exists() or last_path.exists() or best_path.exists():
            raise FileExistsError("Diretório de execução já contém treinamento; use resume=True")
        if preflight_result is None:
            selected_batch_size, preflight_report = preflight_batch_size(
                train_dataset=train_dataset, config=config, device=device
            )
        else:
            selected_batch_size = int(preflight_result[0])
            preflight_report = dict(preflight_result[1])
        write_json(output / "preflight.json", preflight_report)
        history = []

    seed_everything(int(config["seed"]))
    model = make_model(config).to(device)
    criterion = make_criterion(config)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    start_epoch = 1
    best_score = -math.inf
    material_reference = -math.inf
    epochs_without_material_improvement = 0

    if resume and last_path.is_file():
        payload = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            restore_rng=True,
        )
        start_epoch = int(payload["epoch"]) + 1
        best_score = float(payload["best_score"])
        extra = payload.get("extra", {})
        material_reference = float(extra.get("material_reference", best_score))
        epochs_without_material_improvement = int(
            extra.get("epochs_without_material_improvement", 0)
        )

    validation_loader = create_development_dataloader(
        validation_dataset,
        batch_size=selected_batch_size,
        shuffle=False,
        seed=int(config["seed"]),
        num_workers=int(training["num_workers"]),
    )
    early = training["early_stopping"]
    stopped_early = False

    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        epoch_start = time.perf_counter()
        if hasattr(train_dataset.transform, "set_epoch"):
            train_dataset.transform.set_epoch(epoch)
        train_loader = create_development_dataloader(
            train_dataset,
            batch_size=selected_batch_size,
            shuffle=True,
            seed=int(config["seed"]) + epoch,
            num_workers=int(training["num_workers"]),
        )
        train_result = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            amp=False,
            threshold=threshold,
        )
        validation_result = evaluate_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp=False,
            threshold=threshold,
        )
        validation_dice = float(validation_result["metrics"]["dice"]["mean"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(validation_dice)

        if validation_dice > best_score:
            best_score = validation_dice
            save_training_checkpoint(
                best_path,
                kind="best",
                epoch=epoch,
                best_score=best_score,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                run_id=run_id,
                manifest_hashes=manifest_hashes,
                extra={"selected_batch_size": selected_batch_size},
            )

        min_delta = float(early["min_delta"])
        if material_reference == -math.inf or validation_dice >= material_reference + min_delta:
            material_reference = validation_dice
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1

        row = _history_row(
            epoch,
            train_result,
            validation_result,
            current_lr,
            time.perf_counter() - epoch_start,
            best_score,
            epochs_without_material_improvement,
        )
        row["batch_size"] = selected_batch_size
        if hasattr(train_dataset.transform, "get_epoch_stats"):
            for name, value in train_dataset.transform.get_epoch_stats().items():
                row[f"augmentation_{name}"] = value
        history.append(row)
        write_csv(history_path, history)
        save_training_checkpoint(
            last_path,
            kind="last",
            epoch=epoch,
            best_score=best_score,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            run_id=run_id,
            manifest_hashes=manifest_hashes,
            extra={
                "selected_batch_size": selected_batch_size,
                "material_reference": material_reference,
                "epochs_without_material_improvement": epochs_without_material_improvement,
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epochs_without_material_improvement >= int(early["patience"]):
            stopped_early = True
            break

    best_payload = load_training_checkpoint(
        best_path, model=model, map_location=device, restore_rng=False
    )
    result = {
        "run_id": run_id,
        "experiment_id": str(config.get("experiment_id", "D0")),
        "scientific_role": str(config.get("scientific_role", "central_baseline")),
        "epochs_executed": int(history[-1]["epoch"]),
        "best_epoch": int(best_payload["epoch"]),
        "best_validation_dice_at_0_5": float(best_payload["best_score"]),
        "stopped_early": stopped_early,
        "selected_batch_size": selected_batch_size,
        "model_parameters": count_parameters(model),
        "device": str(device),
        "protected_partitions_accessed": False,
    }
    write_json(completed_summary_path, result)
    return result


def thresholds_from_config(config: Mapping[str, Any]) -> np.ndarray:
    selection = config["threshold_selection"]
    start = int(round(float(selection["start"]) * 100))
    stop = int(round(float(selection["stop"]) * 100))
    step = int(round(float(selection["step"]) * 100))
    return np.asarray([value / 100.0 for value in range(start, stop + 1, step)])


def _metrics_from_counts(tp, fp, fn, tn) -> dict[str, np.ndarray]:
    tp = np.asarray(tp, dtype=np.float64)
    fp = np.asarray(fp, dtype=np.float64)
    fn = np.asarray(fn, dtype=np.float64)
    tn = np.asarray(tn, dtype=np.float64)

    def divide(num, den, fallback=1.0):
        return np.divide(num, den, out=np.full_like(num, fallback), where=den > 0)

    dice = divide(2 * tp, 2 * tp + fp + fn)
    iou = divide(tp, tp + fp + fn)
    precision_den = tp + fp
    precision = np.divide(
        tp,
        precision_den,
        out=np.where(tp + fn > 0, np.zeros_like(tp), np.ones_like(tp)),
        where=precision_den > 0,
    )
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    accuracy = divide(tp + tn, tp + tn + fp + fn)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


@torch.inference_mode()
def sweep_validation_thresholds(
    *,
    model,
    loader,
    device: torch.device,
    thresholds: np.ndarray,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    chunk_size: int = 7,
) -> dict[str, Any]:
    """Calcula contagens por imagem em uma grade fixa sem abrir o teste."""
    model.eval()
    ids: list[str] = []
    count_chunks: list[np.ndarray] = []
    threshold_tensor = torch.as_tensor(thresholds, device=device, dtype=torch.float32)
    for batch in loader:
        images = batch["image"].to(device)
        references = batch["mask"].to(device) >= 0.5
        probabilities = torch.sigmoid(model(images).float())
        batch_counts: list[torch.Tensor] = []
        for start in range(0, len(thresholds), chunk_size):
            current = threshold_tensor[start : start + chunk_size]
            predictions = probabilities.unsqueeze(0) >= current[:, None, None, None, None]
            refs = references.unsqueeze(0)
            dimensions = (2, 3, 4)
            tp = (predictions & refs).sum(dim=dimensions)
            fp = (predictions & ~refs).sum(dim=dimensions)
            fn = (~predictions & refs).sum(dim=dimensions)
            tn = (~predictions & ~refs).sum(dim=dimensions)
            batch_counts.append(torch.stack((tp, fp, fn, tn), dim=-1).cpu())
        count_chunks.append(torch.cat(batch_counts, dim=0).numpy())
        ids.extend(str(value) for value in batch["canonical_id"])
    counts = np.concatenate(count_chunks, axis=1)
    curve: list[dict[str, Any]] = []
    all_metrics: list[dict[str, np.ndarray]] = []
    for index, threshold in enumerate(thresholds):
        metric = _metrics_from_counts(*(counts[index, :, column] for column in range(4)))
        all_metrics.append(metric)
        row: dict[str, Any] = {"threshold": float(threshold)}
        for name, values in metric.items():
            row[name] = float(values.mean())
        curve.append(row)
    best_dice = max(row["dice"] for row in curve)
    candidates = [
        index
        for index, row in enumerate(curve)
        if math.isclose(row["dice"], best_dice, abs_tol=1e-12)
    ]
    selected_index = min(
        candidates, key=lambda index: (abs(thresholds[index] - 0.5), thresholds[index])
    )
    selected_threshold = float(thresholds[selected_index])

    def per_image_rows(index: int) -> list[dict[str, Any]]:
        metric = all_metrics[index]
        rows: list[dict[str, Any]] = []
        for image_index, canonical_id in enumerate(ids):
            row: dict[str, Any] = {
                "canonical_id": canonical_id,
                "threshold": float(thresholds[index]),
            }
            for count_index, name in enumerate(COUNT_NAMES):
                row[name] = int(counts[index, image_index, count_index])
            for name in METRIC_NAMES:
                row[name] = float(metric[name][image_index])
            metadata = metadata_by_id[canonical_id]
            row["foreground_fraction"] = float(metadata["foreground_fraction"])
            row["foreground_area_bin"] = int(metadata["foreground_area_bin"])
            row["connected_components_8"] = int(metadata["connected_components_8"])
            rows.append(row)
        return rows

    index_at_0_5 = int(np.argmin(np.abs(thresholds - 0.5)))
    return {
        "curve": curve,
        "selected_threshold": selected_threshold,
        "selected_index": selected_index,
        "per_image_selected": per_image_rows(selected_index),
        "per_image_at_0_5": per_image_rows(index_at_0_5),
        "protected_partitions_accessed": False,
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"images": len(rows), "macro": {}, "micro": {}}
    for name in METRIC_NAMES:
        values = np.asarray([float(row[name]) for row in rows])
        result["macro"][name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    totals = {name: sum(int(row[name]) for row in rows) for name in COUNT_NAMES}
    micro = _metrics_from_counts(
        np.asarray([totals["tp"]]),
        np.asarray([totals["fp"]]),
        np.asarray([totals["fn"]]),
        np.asarray([totals["tn"]]),
    )
    result["micro"] = {name: float(values[0]) for name, values in micro.items()}
    result["counts"] = totals
    return result


def subgroup_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(int(row["foreground_area_bin"]), []).append(row)
    output: list[dict[str, Any]] = []
    for area_bin, group in sorted(groups.items()):
        output.append(
            {
                "foreground_area_bin": area_bin,
                "images": len(group),
                "dice_mean": float(np.mean([float(row["dice"]) for row in group])),
                "iou_mean": float(np.mean([float(row["iou"]) for row in group])),
                "precision_mean": float(np.mean([float(row["precision"]) for row in group])),
                "recall_mean": float(np.mean([float(row["recall"]) for row in group])),
            }
        )
    return output


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    selector = mask.astype(bool)
    output[selector] = 0.55 * output[selector] + 0.45 * np.asarray(color)
    return np.clip(output, 0, 255).astype(np.uint8)


def _image_for_display(dataset, image_chw: np.ndarray) -> np.ndarray:
    """Reconstrói RGB sem alterar o tensor realmente fornecido ao modelo."""
    source = dataset
    while hasattr(source, "dataset"):
        source = source.dataset
    mean = getattr(source, "normalization_mean", None)
    std = getattr(source, "normalization_std", None)
    display = image_chw
    if mean is not None and std is not None:
        display = display * std + mean
    return np.uint8(np.clip(np.transpose(display, (1, 2, 0)) * 255, 0, 255))


@torch.inference_mode()
def save_fixed_validation_panel(
    *, model, dataset, device: torch.device, threshold: float, output_path: str | Path
) -> Path:
    model.eval()
    labels = ("imagem", "referência", "probabilidade", "predição", "erros")
    rendered: list[Image.Image] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        image_tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        probability = torch.sigmoid(model(image_tensor).float())[0, 0].cpu().numpy()
        reference = sample["mask"][0] >= 0.5
        prediction = probability >= threshold
        image = _image_for_display(dataset, sample["image"])
        probability_rgb = np.repeat(
            np.uint8(np.clip(probability * 255, 0, 255))[..., None], 3, axis=2
        )
        error = np.zeros_like(image)
        error[prediction & reference] = (0, 180, 0)
        error[prediction & ~reference] = (255, 165, 0)
        error[~prediction & reference] = (255, 0, 0)
        tiles = (
            image,
            _overlay(image, reference, (255, 0, 0)),
            probability_rgb,
            _overlay(image, prediction, (0, 220, 0)),
            error,
        )
        height, width = image.shape[:2]
        row = Image.new("RGB", (width * len(tiles), height + 28), "white")
        draw = ImageDraw.Draw(row)
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            left = column * width
            row.paste(Image.fromarray(tile), (left, 28))
            draw.text((left + 4, 6), f"{sample['canonical_id']} | {label}", fill="black")
        rendered.append(row)
    canvas = Image.new("RGB", (rendered[0].width, sum(row.height for row in rendered)), "white")
    top = 0
    for row in rendered:
        canvas.paste(row, (0, top))
        top += row.height
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination
