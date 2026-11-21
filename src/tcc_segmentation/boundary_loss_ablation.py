"""Contrato da ablação exploratória DR-IFT-B da ablação de perímetro."""

from __future__ import annotations

from typing import Any, Mapping


EXPECTED_LOSS = {
    "name": "bce_dice_plus_soft_perimeter",
    "bce_weight": 0.5,
    "dice_weight": 0.5,
    "region_weight": 0.9,
    "perimeter_weight": 0.1,
    "smooth": 1.0,
    "perimeter_epsilon": 1.0,
}


def validate_boundary_ablation_config(
    config: Mapping[str, Any], *, baseline_config: Mapping[str, Any] | None = None
) -> None:
    """Impede que a DR-IFT-B altere algo além da função objetivo."""
    if config.get("experiment_id") != "DR-IFT-B":
        raise ValueError("A ablação de borda deve usar experiment_id DR-IFT-B")
    if config.get("loss") != EXPECTED_LOSS:
        raise ValueError("A loss da DR-IFT-B difere do contrato congelado")
    if int(config.get("seed", -1)) not in {42, 1337, 2026}:
        raise ValueError("Seed fora do conjunto confirmatório congelado")
    if config.get("data", {}).get("protected_partitions") != [
        "test",
        "external",
        "FUSeg",
    ]:
        raise ValueError("Partições protegidas devem permanecer bloqueadas")
    if config.get("fine_tuning", {}).get("strategy") != "progressive":
        raise ValueError("DR-IFT-B deve preservar o fine-tuning progressivo")
    if config.get("boundary_ablation", {}).get("only_changed_factor") != "loss":
        raise ValueError("O único fator alterado deve ser a loss")
    if baseline_config is not None:
        for key in ("threshold_selection", "confirmation"):
            if config[key] != baseline_config[key]:
                raise ValueError(f"{key} difere do protocolo de origem")
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
            if config["data"][key] != baseline_config["data"][key]:
                raise ValueError(f"data.{key} difere do protocolo de origem")


def decision_from_seed_deltas(deltas: list[float]) -> dict[str, Any]:
    """Aplica a regra congelada sem inferência assintótica sobre apenas 3 seeds."""
    if len(deltas) != 3:
        raise ValueError("A decisão exige exatamente as três seeds congeladas")
    mean_delta = sum(float(value) for value in deltas) / 3.0
    wins = sum(float(value) > 0.0 for value in deltas)
    advance = bool(mean_delta >= 0.005 and wins >= 2)
    return {
        "mean_dice_delta": mean_delta,
        "seeds_won": wins,
        "minimum_mean_delta": 0.005,
        "minimum_seeds_won": 2,
        "advance": advance,
    }
