"""Construção e validação dos braços D0-A1 e D0-A2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from .data import ResizePad, build_augmentation_pipeline


ALLOWED_EXPERIMENTS = {"D0-A1": "A1", "D0-A2": "A2"}


def validate_augmentation_training_config(
    config: Mapping[str, Any],
    *,
    d0_config: Mapping[str, Any] | None = None,
    audit_config: Mapping[str, Any] | None = None,
) -> None:
    experiment_id = str(config.get("experiment_id"))
    if experiment_id not in ALLOWED_EXPERIMENTS:
        raise ValueError(f"Braço de augmentation não autorizado: {experiment_id}")
    data = config["data"]
    expected = ALLOWED_EXPERIMENTS[experiment_id]
    if str(data.get("augmentation")) != expected:
        raise ValueError(f"{experiment_id} exige augmentation {expected}")
    if str(data.get("color_preprocessing")) != "none":
        raise ValueError("A triagem A1/A2 exige RGB P0 sem pré-processamento")
    if bool(config["model"].get("pretrained")):
        raise ValueError("A triagem A1/A2 não permite pesos pré-treinados")
    if config["model"].get("encoder_weights") is not None:
        raise ValueError("A triagem A1/A2 exige encoder_weights nulo")
    if int(config["training"].get("num_workers", -1)) != 0:
        raise ValueError("A retomada reproduzível desta triagem exige num_workers=0")
    safety = config["augmentation_safety"]
    if str(safety.get("rejection_rule")) != "empty_positive_mask_only":
        raise ValueError("Somente máscara positiva tornada vazia pode ser rejeitada")
    if int(safety.get("max_resamples", -1)) != 3:
        raise ValueError("O protocolo congelado exige no máximo três novos sorteios")
    if str(safety.get("fallback")) != "identity_after_resize":
        raise ValueError("Fallback deve ser identidade após o resize")
    if float(safety.get("retention_review_threshold", -1)) != 0.5:
        raise ValueError("O alerta de auditoria deve permanecer em 0,5")
    if bool(safety.get("retention_threshold_filters_training", True)):
        raise ValueError("O alerta de 50% não pode filtrar o treinamento")
    protected = {str(value).casefold() for value in data["protected_partitions"]}
    if not {"test", "external", "fuseg"}.issubset(protected):
        raise ValueError("Partições protegidas incompletas")

    if d0_config is not None:
        for key in ("model", "loss", "training", "threshold_selection", "confirmation"):
            if config[key] != d0_config[key]:
                raise ValueError(f"{key} difere do D0")
        for key in (
            "train_manifest",
            "validation_manifest",
            "height",
            "width",
            "mask_threshold",
            "image_scale",
            "color_preprocessing",
            "protected_partitions",
        ):
            if data[key] != d0_config["data"][key]:
                raise ValueError(f"data.{key} difere do D0")
        if (
            config["visualization"]["fixed_validation_ids"]
            != d0_config["visualization"]["fixed_validation_ids"]
        ):
            raise ValueError("Casos fixos de validação diferem do D0")
        if int(config["seed"]) != int(d0_config["seed"]):
            raise ValueError("Seed de triagem difere do D0")

    if audit_config is not None and config["augmentation_pipelines"] != audit_config["pipelines"]:
        raise ValueError("Políticas A1/A2 diferem da auditoria visual aprovada")


@dataclass
class EpochSeededSafeAugmentation:
    """Resize determinístico seguido de augmentation com proteção de máscara vazia."""

    name: str
    height: int
    width: int
    pipelines: Mapping[str, Any]
    augmentation_seed_base: int
    max_resamples: int = 3
    _epoch: int = 0
    _pipeline: object | None = None
    _stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._resize = ResizePad(self.height, self.width)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
        seed = int(self.augmentation_seed_base) + self._epoch
        self._pipeline = build_augmentation_pipeline(self.name, self.pipelines, seed=seed)
        self._stats = {
            "epoch_seed": seed,
            "samples_seen": 0,
            "empty_mask_rejections": 0,
            "resample_attempts": 0,
            "identity_fallbacks": 0,
        }

    def get_epoch_stats(self) -> dict[str, int]:
        return dict(self._stats)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        resized_image, resized_mask = self._resize(image, mask)
        self._stats["samples_seen"] += 1
        originally_positive = bool(np.any(resized_mask > 0))
        for attempt in range(self.max_resamples + 1):
            transformed_image, transformed_mask = self._pipeline(resized_image, resized_mask)
            became_empty = originally_positive and not bool(np.any(transformed_mask > 0))
            if not became_empty:
                return transformed_image, transformed_mask
            self._stats["empty_mask_rejections"] += 1
            if attempt < self.max_resamples:
                self._stats["resample_attempts"] += 1
        self._stats["identity_fallbacks"] += 1
        return resized_image.copy(), resized_mask.copy()


def make_augmentation_pair_transform(config: Mapping[str, Any]) -> EpochSeededSafeAugmentation:
    validate_augmentation_training_config(config)
    data = config["data"]
    safety = config["augmentation_safety"]
    return EpochSeededSafeAugmentation(
        name=str(data["augmentation"]),
        height=int(data["height"]),
        width=int(data["width"]),
        pipelines=config["augmentation_pipelines"],
        augmentation_seed_base=int(config["augmentation_randomness"]["seed_base"]),
        max_resamples=int(safety["max_resamples"]),
    )


def save_first_training_batch_panel(dataset, output_path, *, samples: int = 4):
    """Salva exatamente o que a rede receberá no primeiro lote demonstrativo."""
    tiles: list[Image.Image] = []
    for index in range(min(int(samples), len(dataset))):
        sample = dataset[index]
        image = np.uint8(np.clip(np.transpose(sample["image"], (1, 2, 0)) * 255.0, 0, 255))
        mask = sample["mask"][0] >= 0.5
        overlay = image.astype(np.float32).copy()
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.asarray((30, 220, 70))
        overlay = np.uint8(np.clip(overlay, 0, 255))
        mask_rgb = np.repeat(np.uint8(mask)[..., None] * 255, 3, axis=2)
        height, width = mask.shape
        row = Image.new("RGB", (width * 3, height + 28), "white")
        draw = ImageDraw.Draw(row)
        for column, (array, label) in enumerate(
            ((image, "imagem transformada"), (mask_rgb, "máscara"), (overlay, "sobreposição"))
        ):
            left = column * width
            row.paste(Image.fromarray(array), (left, 28))
            draw.text((left + 4, 6), f"{sample['canonical_id']} | {label}", fill="black")
        tiles.append(row)
    canvas = Image.new("RGB", (tiles[0].width, sum(tile.height for tile in tiles)), "white")
    top = 0
    for tile in tiles:
        canvas.paste(tile, (0, top))
        top += tile.height
    from pathlib import Path

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination
