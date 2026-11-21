"""Funcoes objetivo calculadas diretamente sobre logits."""

from __future__ import annotations

import torch
from torch import nn


def _validate_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.shape != targets.shape:
        raise ValueError(
            f"Logits e targets devem ter o mesmo shape: {logits.shape} != {targets.shape}"
        )
    if logits.ndim < 3:
        raise ValueError(f"Tensor de segmentacao esperado; ndim={logits.ndim}")


class SoftDiceLoss(nn.Module):
    """Um menos Dice probabilistico, calculado por imagem e depois promediado."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth deve ser positivo")
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_shapes(logits, targets)
        probabilities = torch.sigmoid(logits.float())
        targets_float = targets.float()
        dimensions = tuple(range(1, logits.ndim))
        intersection = (probabilities * targets_float).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets_float.sum(dim=dimensions)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combinacao pre-registrada de BCEWithLogits e Soft Dice."""

    def __init__(
        self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Pesos da loss nao podem ser negativos")
        if bce_weight + dice_weight <= 0:
            raise ValueError("Ao menos um peso da loss deve ser positivo")
        total = float(bce_weight + dice_weight)
        self.bce_weight = float(bce_weight) / total
        self.dice_weight = float(dice_weight) / total
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        _validate_shapes(logits, targets)
        bce = self.bce(logits.float(), targets.float())
        dice = self.dice(logits, targets)
        total = self.bce_weight * bce + self.dice_weight * dice
        return {"total": total, "bce": bce, "dice_loss": dice}

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.components(logits, targets)["total"]


class SoftPerimeterLoss(nn.Module):
    """Diferença relativa entre perímetros suaves, calculada por imagem.

    O contorno suave é a diferença entre a probabilidade e sua erosão 3x3.
    A normalização pelo perímetro da referência evita que lesões grandes dominem
    a média. Esta é uma implementação inspirada na perda de perímetro de
    El Jurdi et al. (2021), combinada a uma loss regional; não é apresentada
    como reprodução literal do código dos autores.
    """

    def __init__(self, epsilon: float = 1.0) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon deve ser positivo")
        self.epsilon = float(epsilon)

    @staticmethod
    def _soft_perimeter(values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("SoftPerimeterLoss espera tensores NCHW")
        eroded = -torch.nn.functional.max_pool2d(-values, kernel_size=3, stride=1, padding=1)
        contour = torch.clamp(values - eroded, min=0.0)
        return contour.sum(dim=(1, 2, 3))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _validate_shapes(logits, targets)
        probabilities = torch.sigmoid(logits.float())
        targets_float = targets.float()
        predicted = self._soft_perimeter(probabilities)
        reference = self._soft_perimeter(targets_float)
        relative_error = torch.abs(predicted - reference) / (reference + self.epsilon)
        return relative_error.mean()


class BCEDicePerimeterLoss(nn.Module):
    """Loss regional BCE+Dice com uma penalização suave de perímetro."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        region_weight: float = 0.9,
        perimeter_weight: float = 0.1,
        smooth: float = 1.0,
        perimeter_epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        if region_weight < 0 or perimeter_weight < 0:
            raise ValueError("Pesos regional/perímetro não podem ser negativos")
        if region_weight + perimeter_weight <= 0:
            raise ValueError("Ao menos um peso regional/perímetro deve ser positivo")
        total = float(region_weight + perimeter_weight)
        self.region_weight = float(region_weight) / total
        self.perimeter_weight = float(perimeter_weight) / total
        self.region = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth)
        self.perimeter = SoftPerimeterLoss(epsilon=perimeter_epsilon)

    def components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        region_components = self.region.components(logits, targets)
        perimeter = self.perimeter(logits, targets)
        total = self.region_weight * region_components["total"] + self.perimeter_weight * perimeter
        return {
            "total": total,
            "bce": region_components["bce"],
            "dice_loss": region_components["dice_loss"],
            "perimeter_loss": perimeter,
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.components(logits, targets)["total"]
