"""Fabrica de modelos sem ativacao final: a saida sempre representa logits."""

from __future__ import annotations

from typing import Any, Mapping


def build_segmentation_model(config: Mapping[str, Any]):
    architecture = str(config.get("architecture", "")).casefold()
    activation = config.get("activation")
    if activation not in (None, "none"):
        raise ValueError("O modelo deve retornar logits; activation precisa ser null")

    if architecture == "unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            raise RuntimeError(
                "segmentation-models-pytorch nao instalado; execute no ambiente Colab."
            ) from error
        return smp.Unet(
            encoder_name=str(config["encoder"]),
            encoder_weights=config.get("encoder_weights"),
            in_channels=int(config.get("in_channels", 3)),
            classes=int(config.get("classes", 1)),
            activation=None,
        )

    if architecture in {"deep_res_unet", "deepresunet"}:
        if config.get("encoder_weights") not in (None, "none"):
            raise ValueError("A DeepResUNet historica nao possui encoder pre-treinado")
        from .deep_res_unet import DeepResUNet

        return DeepResUNet(
            in_channels=int(config.get("in_channels", 3)),
            classes=int(config.get("classes", 1)),
            base_channels=int(config.get("base_channels", 32)),
        )

    if architecture in {
        "deep_res_unet_resnet34",
        "deepresunet_resnet34",
        "deepresunet-r34",
    }:
        from .deep_res_unet import DeepResUNetResNet34

        return DeepResUNetResNet34(
            in_channels=int(config.get("in_channels", 3)),
            classes=int(config.get("classes", 1)),
            encoder_weights=config.get("encoder_weights"),
            freeze_encoder=bool(config.get("freeze_encoder", False)),
        )

    raise ValueError(f"Arquitetura nao suportada nesta fase: {architecture}")


def count_parameters(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}
