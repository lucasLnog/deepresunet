"""Contrato congelado para o contraste DR-RT versus DR-IT."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from PIL import Image, ImageDraw

from .augmentation_experiment import EpochSeededSafeAugmentation

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ARMS = {"DR-RT": None, "DR-IT": "imagenet"}


def validate_transfer_config(
    config: Mapping[str, Any], *, a1_config=None, audit_config=None
) -> None:
    arm = str(config.get("experiment_id"))
    if arm not in ARMS:
        raise ValueError(f"braço de transferência não autorizado: {arm}")
    model, data = config["model"], config["data"]
    if model.get("architecture") != "deep_res_unet_resnet34":
        raise ValueError("o contraste exige DeepResUNetResNet34")
    if model.get("encoder_weights") != ARMS[arm]:
        raise ValueError("encoder_weights não corresponde ao braço")
    if bool(model.get("freeze_encoder")):
        raise ValueError("DR-RT e DR-IT exigem encoder treinável")
    if data.get("augmentation") != "A1" or data.get("color_preprocessing") != "none":
        raise ValueError("ambos os braços exigem P0 + A1")
    norm = data.get("normalization", {})
    if tuple(norm.get("mean", ())) != IMAGENET_MEAN or tuple(norm.get("std", ())) != IMAGENET_STD:
        raise ValueError("ambos os braços exigem normalização ImageNet idêntica")
    if int(config["training"].get("num_workers", -1)) != 0:
        raise ValueError("retomada reproduzível exige num_workers=0")
    protected = {str(x).casefold() for x in data["protected_partitions"]}
    if not {"test", "external", "fuseg"}.issubset(protected):
        raise ValueError("partições protegidas incompletas")
    if a1_config is not None:
        for key in ("loss", "training", "threshold_selection", "confirmation"):
            if config[key] != a1_config[key]:
                raise ValueError(f"{key} difere do A1")
        for key in (
            "train_manifest",
            "validation_manifest",
            "height",
            "width",
            "mask_threshold",
            "color_preprocessing",
            "augmentation",
            "protected_partitions",
        ):
            if data[key] != a1_config["data"][key]:
                raise ValueError(f"data.{key} difere do A1")
        if (
            config["visualization"]["fixed_validation_ids"]
            != a1_config["visualization"]["fixed_validation_ids"]
        ):
            raise ValueError("casos visuais diferem do A1")
    if audit_config is not None and config["augmentation_pipelines"] != audit_config["pipelines"]:
        raise ValueError("A1 difere da auditoria aprovada")


def validate_transfer_pair(random_config, imagenet_config) -> None:
    validate_transfer_config(random_config)
    validate_transfer_config(imagenet_config)
    ignored = {"experiment_id", "scientific_role", "runtime", "model"}
    for key in set(random_config) - ignored:
        if random_config[key] != imagenet_config[key]:
            raise ValueError(f"braços diferem indevidamente em {key}")
    mr, mi = dict(random_config["model"]), dict(imagenet_config["model"])
    mr.pop("encoder_weights")
    mi.pop("encoder_weights")
    if mr != mi:
        raise ValueError("modelos diferem além da inicialização do encoder")


def make_transfer_train_transform(config):
    validate_transfer_config(config)
    return EpochSeededSafeAugmentation(
        name="A1",
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        pipelines=config["augmentation_pipelines"],
        augmentation_seed_base=int(config["augmentation_randomness"]["seed_base"]),
        max_resamples=int(config["augmentation_safety"]["max_resamples"]),
    )


def save_normalized_batch_panel(dataset, output_path, *, samples=4):
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    rows = []
    for i in range(min(int(samples), len(dataset))):
        sample = dataset[i]
        rgb = np.uint8(np.clip(np.transpose(sample["image"] * std + mean, (1, 2, 0)) * 255, 0, 255))
        mask = sample["mask"][0] >= 0.5
        overlay = rgb.astype(np.float32)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([30, 220, 70])
        arrays = (
            rgb,
            np.repeat(mask[..., None], 3, axis=2).astype(np.uint8) * 255,
            np.uint8(overlay),
        )
        h, w = mask.shape
        row = Image.new("RGB", (3 * w, h + 28), "white")
        draw = ImageDraw.Draw(row)
        for j, (arr, label) in enumerate(
            zip(arrays, ("imagem após desfazer normalização", "máscara", "sobreposição"))
        ):
            row.paste(Image.fromarray(arr), (j * w, 28))
            draw.text((j * w + 4, 6), f"{sample['canonical_id']} | {label}", fill="black")
        rows.append(row)
    canvas = Image.new("RGB", (rows[0].width, sum(x.height for x in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path
