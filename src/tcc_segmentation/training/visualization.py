"""Paineis de depuracao do overfit sem depender de matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image.astype(np.float32).copy()
    selector = mask.astype(bool)
    output[selector] = 0.55 * output[selector] + 0.45 * np.asarray(color, dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


@torch.inference_mode()
def save_overfit_panel(
    *, model, dataset, device: torch.device, output_path: str | Path, threshold: float
) -> None:
    model.eval()
    rows: list[Image.Image] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        probability = torch.sigmoid(model(tensor).float())[0, 0].cpu().numpy()
        reference = sample["mask"][0] >= 0.5
        prediction = probability >= threshold
        image = np.transpose(sample["image"], (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        probability_rgb = np.repeat(
            np.clip(probability * 255.0, 0, 255).astype(np.uint8)[..., None], 3, axis=2
        )
        tiles = [
            image,
            _overlay(image, reference, (255, 0, 0)),
            probability_rgb,
            _overlay(image, prediction, (0, 255, 0)),
        ]
        row = Image.new("RGB", (image.shape[1] * 4, image.shape[0] + 24), "white")
        draw = ImageDraw.Draw(row)
        labels = ("imagem", "referencia", "probabilidade", "predicao")
        for column, (tile, label) in enumerate(zip(tiles, labels)):
            left = column * image.shape[1]
            row.paste(Image.fromarray(tile), (left, 24))
            draw.text((left + 4, 4), f"{sample['canonical_id']} | {label}", fill="black")
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    top = 0
    for row in rows:
        canvas.paste(row, (0, top))
        top += row.height
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
