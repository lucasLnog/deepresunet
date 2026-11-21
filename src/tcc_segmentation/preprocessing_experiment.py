"""Validação e construção dos braços de pré-processamento da comparação de pré-processamento."""

from __future__ import annotations

from typing import Any, Mapping

from .data import (
    ComposePair,
    ImageOnlyPairTransform,
    ResizePad,
    build_color_preprocessor,
)


ALLOWED_EXPERIMENTS = {
    "D-P1": "shades_of_gray",
    "D-P2": "clahe_l_star",
}


def validate_preprocessing_training_config(config: Mapping[str, Any]) -> None:
    experiment_id = str(config.get("experiment_id"))
    if experiment_id not in ALLOWED_EXPERIMENTS:
        raise ValueError(f"Braço de pré-processamento não autorizado: {experiment_id}")
    data = config["data"]
    observed_method = str(data.get("color_preprocessing"))
    expected_method = ALLOWED_EXPERIMENTS[experiment_id]
    if observed_method != expected_method:
        raise ValueError(f"{experiment_id} exige {expected_method}; observado={observed_method}")
    if str(data.get("augmentation")) != "none":
        raise ValueError("A triagem P1/P2 não permite augmentation")
    if bool(config["model"].get("pretrained")):
        raise ValueError("A triagem P1/P2 não permite pesos pré-treinados")
    if config["model"].get("encoder_weights") is not None:
        raise ValueError("A triagem P1/P2 exige encoder_weights nulo")
    protected = {str(value).casefold() for value in data["protected_partitions"]}
    if not {"test", "external", "fuseg"}.issubset(protected):
        raise ValueError("Partições protegidas incompletas")


def make_preprocessing_pair_transform(config: Mapping[str, Any]) -> ComposePair:
    """Monta cor antes do resize, sem alterar a máscara."""

    validate_preprocessing_training_config(config)
    data = config["data"]
    parameters = data.get("color_preprocessing_parameters", {})
    color = build_color_preprocessor(
        str(data["color_preprocessing"]),
        minkowski_p=float(parameters.get("minkowski_p", 6.0)),
        clip_limit=float(parameters.get("clip_limit", 2.0)),
        tile_grid_size=tuple(parameters.get("tile_grid_size", [8, 8])),
    )
    return ComposePair(
        [
            ImageOnlyPairTransform(color),
            ResizePad(height=int(data["height"]), width=int(data["width"])),
        ]
    )
