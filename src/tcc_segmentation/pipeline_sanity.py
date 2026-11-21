"""Ferramentas transparentes para o gate técnico da verificação do pipeline.

Este módulo acessa somente datasets já fornecidos pelo chamador. Ele não conhece
manifestos de teste ou bases externas e seus resultados não são métricas científicas.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from .training.engine import evaluate_epoch, train_epoch


def indices_for_canonical_ids(dataset, canonical_ids: Sequence[str]) -> list[int]:
    """Resolve IDs pré-fixados sem depender da ordem física do manifesto."""
    requested = [str(value) for value in canonical_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("canonical_ids contém repetição")
    index_by_id = {str(row["canonical_id"]): index for index, row in enumerate(dataset.rows)}
    missing = [value for value in requested if value not in index_by_id]
    if missing:
        raise ValueError(f"IDs ausentes no dataset permitido: {missing}")
    return [index_by_id[value] for value in requested]


def deterministic_prefix_indices(length: int, count: int) -> list[int]:
    """Seleciona um prefixo determinístico para smoke test, nunca por desempenho."""
    if length < 1 or count < 1:
        raise ValueError("length e count devem ser positivos")
    return list(range(min(length, count)))


def batch_diagnostics(batch: dict[str, Any]) -> dict[str, Any]:
    """Resume shapes, faixas e valores das máscaras de um lote."""
    images = batch["image"]
    masks = batch["mask"]
    return {
        "canonical_ids": [str(value) for value in batch["canonical_id"]],
        "image_shape": list(images.shape),
        "mask_shape": list(masks.shape),
        "image_dtype": str(images.dtype),
        "mask_dtype": str(masks.dtype),
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "mask_unique_values": [
            float(value) for value in torch.unique(masks).detach().cpu().tolist()
        ],
        "all_finite": bool(torch.isfinite(images).all() and torch.isfinite(masks).all()),
    }


@torch.inference_mode()
def collect_predictions(model, loader, device: torch.device) -> list[dict[str, Any]]:
    """Coleta entradas, referências, logits e probabilidades para inspeção visual."""
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images).float()
        probabilities = torch.sigmoid(logits)
        for index, canonical_id in enumerate(batch["canonical_id"]):
            records.append(
                {
                    "canonical_id": str(canonical_id),
                    "image": images[index].detach().cpu().numpy(),
                    "mask": masks[index, 0].detach().cpu().numpy(),
                    "logits": logits[index, 0].detach().cpu().numpy(),
                    "probability": probabilities[index, 0].detach().cpu().numpy(),
                }
            )
    return records


def run_overfit_sanity(
    *,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    threshold: float,
    min_loss_reduction_fraction: float,
    min_macro_dice: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Treina apenas o subconjunto fixo e retorna histórico e gate diagnóstico."""
    if epochs < 1:
        raise ValueError("epochs deve ser positivo")
    history: list[dict[str, Any]] = []
    initial_loss: float | None = None
    accepted = False
    for epoch in range(1, epochs + 1):
        result = train_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            amp=False,
            threshold=threshold,
        )
        total_loss = float(result["loss"]["total"])
        macro_dice = float(result["metrics"]["dice"]["mean"])
        if initial_loss is None:
            initial_loss = total_loss
        loss_reduction = 1.0 - total_loss / max(initial_loss, 1e-12)
        row = {
            "epoch": epoch,
            "loss": total_loss,
            "bce": float(result["loss"]["bce"]),
            "dice_loss": float(result["loss"]["dice_loss"]),
            "macro_dice_at_0_5": macro_dice,
            "loss_reduction_fraction": loss_reduction,
            "gradient_l2_norm": float(result["gradient_diagnostics"]["gradient_l2_norm"]),
        }
        history.append(row)
        accepted = bool(
            loss_reduction >= min_loss_reduction_fraction and macro_dice >= min_macro_dice
        )
        if accepted:
            break
    return history, accepted


def run_smoke_sanity(
    *,
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    """Executa uma época técnica e não atribui significado científico às métricas."""
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
    return {
        "scientific_result": False,
        "train": train_result,
        "validation": validation_result,
    }


def save_history_csv(history: Sequence[dict[str, Any]], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("Histórico vazio")
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return destination


def prediction_max_abs_difference(
    first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]]
) -> float:
    if [row["canonical_id"] for row in first] != [row["canonical_id"] for row in second]:
        raise ValueError("Predições correspondem a IDs diferentes")
    return max(float(np.max(np.abs(a["logits"] - b["logits"]))) for a, b in zip(first, second))


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    selector = mask.astype(bool)
    output[selector] = 0.55 * output[selector] + 0.45 * np.asarray(color)
    return np.clip(output, 0, 255).astype(np.uint8)


def save_before_after_panel(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
    path: str | Path,
    threshold: float = 0.5,
) -> Path:
    """Salva imagem, referência e probabilidades antes/depois para os mesmos IDs."""
    if [row["canonical_id"] for row in before] != [row["canonical_id"] for row in after]:
        raise ValueError("Antes e depois contêm IDs diferentes")
    labels = (
        "imagem",
        "referencia",
        "prob. antes",
        "pred. antes",
        "prob. depois",
        "pred. depois",
    )
    rows: list[Image.Image] = []
    for initial, final in zip(before, after):
        image = np.transpose(initial["image"], (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        reference = initial["mask"] >= 0.5
        before_probability = initial["probability"]
        after_probability = final["probability"]
        grayscale_before = np.repeat(
            np.uint8(np.clip(before_probability * 255.0, 0, 255))[..., None], 3, axis=2
        )
        grayscale_after = np.repeat(
            np.uint8(np.clip(after_probability * 255.0, 0, 255))[..., None], 3, axis=2
        )
        tiles = (
            image,
            _overlay(image, reference, (255, 0, 0)),
            grayscale_before,
            _overlay(image, before_probability >= threshold, (0, 180, 255)),
            grayscale_after,
            _overlay(image, after_probability >= threshold, (0, 220, 0)),
        )
        height, width = image.shape[:2]
        row = Image.new("RGB", (width * len(tiles), height + 28), "white")
        draw = ImageDraw.Draw(row)
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            left = column * width
            row.paste(Image.fromarray(tile), (left, 28))
            draw.text((left + 4, 6), label, fill="black")
        draw.text((4, height + 13), initial["canonical_id"], fill="black")
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def json_safe(value: Any) -> Any:
    """Converte estruturas com tensores em tipos serializáveis."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(value: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination
