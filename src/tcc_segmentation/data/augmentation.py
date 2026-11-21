"""Pipelines A0/A1/A2 congelados para a ablação de data augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _pair(values: object, name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} deve conter exatamente dois valores")
    low, high = float(values[0]), float(values[1])
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"Faixa inválida em {name}: {values}")
    return low, high


def _probability(value: object, name: str) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} deve estar entre zero e um")
    return probability


def _validate_pair(image: np.ndarray, mask: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB uint8")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError("A máscara deve ser 2D e alinhada à imagem")


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _applied_trace(items: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not isinstance(items, (list, tuple)):
        return records
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            records.append(
                {"transform": str(item[0]).split(".")[-1], "params": _json_safe(item[1])}
            )
    return records


@dataclass
class AugmentationPairTransform:
    """Adapta Compose ao contrato `(image, mask) -> (image, mask)`."""

    name: str
    pipeline: object | None

    def set_seed(self, seed: int) -> None:
        if self.pipeline is not None:
            self.pipeline.set_random_seed(int(seed))

    def apply_with_trace(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
        _validate_pair(image, mask)
        if self.pipeline is None:
            return image.copy(), mask.copy(), []
        result = self.pipeline(image=image, mask=mask)
        transformed_image = np.asarray(result["image"], dtype=np.uint8)
        transformed_mask = np.asarray(result["mask"])
        _validate_pair(transformed_image, transformed_mask)
        return transformed_image, transformed_mask, _applied_trace(result.get("applied_transforms"))

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed_image, transformed_mask, _ = self.apply_with_trace(image, mask)
        return transformed_image, transformed_mask


def _geometric_transforms(config: Mapping[str, Any], A: object, cv2: object) -> list[object]:
    affine = config["affine"]
    scale = _pair(affine["scale"], "scale")
    rotation = _pair(affine["rotation_degrees"], "rotation_degrees")
    shear = _pair(affine["shear_degrees"], "shear_degrees")
    translation = affine["translation_fraction"]
    translation_x = _pair(translation["x"], "translation_fraction.x")
    translation_y = _pair(translation["y"], "translation_fraction.y")
    if shear != (0.0, 0.0):
        raise ValueError("Shear permanece proibido nesta versão do protocolo")
    return [
        A.HorizontalFlip(
            p=_probability(config["horizontal_flip_probability"], "horizontal_flip_probability")
        ),
        A.VerticalFlip(
            p=_probability(config["vertical_flip_probability"], "vertical_flip_probability")
        ),
        A.Affine(
            scale=scale,
            translate_percent={"x": translation_x, "y": translation_y},
            rotate=rotation,
            shear=shear,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            keep_ratio=bool(affine["keep_aspect_ratio"]),
            balanced_scale=bool(affine["balanced_scale"]),
            border_mode=cv2.BORDER_CONSTANT,
            fill=tuple(int(value) for value in affine["image_fill_rgb"]),
            fill_mask=int(affine["mask_fill"]),
            p=_probability(affine["probability"], "affine.probability"),
        ),
    ]


def build_augmentation_pipeline(
    name: str,
    pipelines_config: Mapping[str, Any],
    *,
    seed: int,
) -> AugmentationPairTransform:
    """Constrói A0, A1 ou A2 sem permitir combinações não registradas."""

    normalized = str(name).strip().upper()
    if normalized == "A0":
        return AugmentationPairTransform(name="A0", pipeline=None)
    if normalized not in {"A1", "A2"}:
        raise ValueError(f"Pipeline de augmentation desconhecido: {name}")

    try:
        import albumentations as A
        import cv2
    except ImportError as error:  # pragma: no cover - dependências fixadas no Colab
        raise RuntimeError("Albumentations e OpenCV são necessários para A1/A2") from error

    geometric_config = pipelines_config["A1"]
    transforms = _geometric_transforms(geometric_config, A, cv2)
    if normalized == "A2":
        a2 = pipelines_config["A2"]
        choices = a2["photometric_choices"]
        brightness = choices["brightness_contrast"]
        gamma = choices["gamma"]
        hsv = choices["hue_saturation_value"]
        transforms.append(
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=_pair(brightness["brightness_limit"], "brightness_limit"),
                        contrast_limit=_pair(brightness["contrast_limit"], "contrast_limit"),
                        p=float(brightness["weight"]),
                    ),
                    A.RandomGamma(
                        gamma_limit=tuple(int(value) for value in gamma["gamma_limit"]),
                        p=float(gamma["weight"]),
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=tuple(int(value) for value in hsv["hue_shift_limit"]),
                        sat_shift_limit=tuple(
                            int(value) for value in hsv["saturation_shift_limit"]
                        ),
                        val_shift_limit=tuple(int(value) for value in hsv["value_shift_limit"]),
                        p=float(hsv["weight"]),
                    ),
                ],
                p=_probability(
                    a2["photometric_one_of_probability"],
                    "photometric_one_of_probability",
                ),
            )
        )
    pipeline = A.Compose(
        transforms,
        p=1.0,
        strict=True,
        mask_interpolation=cv2.INTER_NEAREST,
        seed=int(seed),
        save_applied_params=True,
    )
    return AugmentationPairTransform(name=normalized, pipeline=pipeline)
