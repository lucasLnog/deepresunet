"""Validação e transforms da combinação planejada P1+A1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .data import ImageOnlyPairTransform, ResizePad, build_color_preprocessor
from .augmentation_experiment import EpochSeededSafeAugmentation


EXPERIMENT_ID = "P1-A1"


def validate_p1_a1_config(
    config: Mapping[str, Any],
    *,
    d0_config: Mapping[str, Any] | None = None,
    p1_config: Mapping[str, Any] | None = None,
    a1_config: Mapping[str, Any] | None = None,
    audit_config: Mapping[str, Any] | None = None,
) -> None:
    """Garante que somente P1 e A1 foram combinados sobre o D0."""

    if str(config.get("experiment_id")) != EXPERIMENT_ID:
        raise ValueError(f"Combinação não autorizada: {config.get('experiment_id')}")
    data = config["data"]
    if str(data.get("color_preprocessing")) != "shades_of_gray":
        raise ValueError("P1-A1 exige Shades-of-Gray")
    parameters = data.get("color_preprocessing_parameters", {})
    if float(parameters.get("minkowski_p", -1)) != 6.0:
        raise ValueError("P1-A1 exige Shades-of-Gray com p=6")
    if str(data.get("augmentation")) != "A1":
        raise ValueError("P1-A1 exige augmentation geométrico A1")
    if bool(config["model"].get("pretrained")):
        raise ValueError("P1-A1 deve começar do zero")
    if config["model"].get("encoder_weights") is not None:
        raise ValueError("P1-A1 exige encoder_weights nulo")
    if int(config["training"].get("num_workers", -1)) != 0:
        raise ValueError("A retomada reproduzível exige num_workers=0")

    safety = config["augmentation_safety"]
    if str(safety.get("rejection_rule")) != "empty_positive_mask_only":
        raise ValueError("Somente máscara positiva tornada vazia pode ser rejeitada")
    if int(safety.get("max_resamples", -1)) != 3:
        raise ValueError("O protocolo exige no máximo três novos sorteios")
    if str(safety.get("fallback")) != "identity_after_resize":
        raise ValueError("Fallback deve ser identidade após resize")
    if bool(safety.get("retention_threshold_filters_training", True)):
        raise ValueError("O alerta de retenção não pode filtrar o treino")

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
            "protected_partitions",
        ):
            if data[key] != d0_config["data"][key]:
                raise ValueError(f"data.{key} difere do D0")
        if int(config["seed"]) != int(d0_config["seed"]):
            raise ValueError("Seed difere do D0")
        if (
            config["visualization"]["fixed_validation_ids"]
            != d0_config["visualization"]["fixed_validation_ids"]
        ):
            raise ValueError("Casos fixos de validação diferem do D0")

    if p1_config is not None:
        for key in ("color_preprocessing", "color_preprocessing_parameters"):
            if data.get(key) != p1_config["data"].get(key):
                raise ValueError(f"P1 combinado difere do P1 selecionado em data.{key}")
    if a1_config is not None:
        if data["augmentation"] != a1_config["data"]["augmentation"]:
            raise ValueError("A1 combinado difere do A1 selecionado")
        for key in ("augmentation_randomness", "augmentation_safety", "augmentation_pipelines"):
            if config[key] != a1_config[key]:
                raise ValueError(f"{key} difere do A1 selecionado")
    if audit_config is not None and config["augmentation_pipelines"] != audit_config["pipelines"]:
        raise ValueError("Política A1 difere da auditoria visual aprovada")

    plan = config.get("comparison_plan", {})
    if plan.get("partition") != "validation":
        raise ValueError("A comparação P1+A1 deve usar somente validação")
    if plan.get("primary_metric") != "macro_dice_per_image":
        raise ValueError("A métrica primária deve permanecer Dice macro por imagem")
    if plan.get("primary_contrast") != "P1A1_minus_A1":
        raise ValueError("O contraste primário deve ser P1+A1 menos A1")
    if bool(plan.get("test_access_allowed", True)):
        raise ValueError("O teste deve permanecer bloqueado")


def _make_p1_image_transform(config: Mapping[str, Any]) -> ImageOnlyPairTransform:
    parameters = config["data"]["color_preprocessing_parameters"]
    color = build_color_preprocessor("shades_of_gray", minkowski_p=float(parameters["minkowski_p"]))
    return ImageOnlyPairTransform(color)


@dataclass
class P1A1TrainTransform:
    """Aplica P1 à imagem original e depois resize+A1 ao par imagem/máscara."""

    color: ImageOnlyPairTransform
    augmentation: EpochSeededSafeAugmentation

    def set_epoch(self, epoch: int) -> None:
        self.augmentation.set_epoch(epoch)

    def get_epoch_stats(self) -> dict[str, int]:
        return self.augmentation.get_epoch_stats()

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self.color(image, mask)
        return self.augmentation(image, mask)


def make_p1_a1_train_transform(config: Mapping[str, Any]) -> P1A1TrainTransform:
    validate_p1_a1_config(config)
    data = config["data"]
    safety = config["augmentation_safety"]
    augmentation = EpochSeededSafeAugmentation(
        name="A1",
        height=int(data["height"]),
        width=int(data["width"]),
        pipelines=config["augmentation_pipelines"],
        augmentation_seed_base=int(config["augmentation_randomness"]["seed_base"]),
        max_resamples=int(safety["max_resamples"]),
    )
    return P1A1TrainTransform(_make_p1_image_transform(config), augmentation)


@dataclass
class P1ValidationTransform:
    """Aplica P1 e resize, sem qualquer sorteio de augmentation."""

    color: ImageOnlyPairTransform
    resize: ResizePad

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image, mask = self.color(image, mask)
        return self.resize(image, mask)


def make_p1_validation_transform(config: Mapping[str, Any]) -> P1ValidationTransform:
    validate_p1_a1_config(config)
    data = config["data"]
    return P1ValidationTransform(
        _make_p1_image_transform(config),
        ResizePad(height=int(data["height"]), width=int(data["width"])),
    )
