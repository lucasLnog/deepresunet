from pathlib import Path

import pytest
import yaml


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.tcc_segmentation.models import (  # noqa: E402
    DeepResUNetResNet34,
    build_segmentation_model,
    count_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def model_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / "models" / name).read_text(encoding="utf-8"))


def test_random_frozen_variant_preserves_shape_and_decoder_channels() -> None:
    model = build_segmentation_model(model_config("deep_res_unet_resnet34_random_frozen.yaml"))
    model.eval()
    with torch.inference_mode():
        output = model(torch.rand((1, 3, 64, 80)))
    assert output.shape == (1, 1, 64, 80)
    assert model.stem_conv.stride == (1, 1)
    assert model.up4.in_channels == 512 and model.up4.out_channels == 256
    assert model.up1.in_channels == 64 and model.up1.out_channels == 32
    assert model.final_projection.out_channels == 1


def test_frozen_encoder_stays_in_eval_and_decoder_receives_gradients() -> None:
    model = DeepResUNetResNet34(encoder_weights=None, freeze_encoder=True)
    model.train()
    assert model.encoder_is_frozen()
    assert not model.stem_bn.training
    assert not any(
        module.training
        for module in model.layer4.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    output = model(torch.rand((1, 3, 32, 32)))
    output.mean().backward()
    assert all(parameter.grad is None for parameter in model.encoder_parameters())
    decoder_gradients = [
        parameter.grad for parameter in model.decoder_parameters() if parameter.requires_grad
    ]
    assert decoder_gradients
    assert any(gradient is not None for gradient in decoder_gradients)


def test_progressive_unfreezing_releases_only_requested_stage() -> None:
    model = DeepResUNetResNet34(encoder_weights=None, freeze_encoder=True)
    model.set_trainable_encoder_stages(("layer4",))
    model.train()
    assert any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert not any(parameter.requires_grad for parameter in model.layer3.parameters())
    assert any(
        module.training
        for module in model.layer4.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    assert not model.stem_bn.training


def test_decoder_initialization_is_reproducible_across_encoder_controls() -> None:
    torch.manual_seed(42)
    first = DeepResUNetResNet34(encoder_weights=None, freeze_encoder=True)
    torch.manual_seed(42)
    second = DeepResUNetResNet34(encoder_weights=None, freeze_encoder=True)
    assert first.decoder_state_dict().keys() == second.decoder_state_dict().keys()
    assert all(
        torch.equal(first.decoder_state_dict()[name], second.decoder_state_dict()[name])
        for name in first.decoder_state_dict()
    )


def test_invalid_encoder_stage_is_rejected() -> None:
    model = DeepResUNetResNet34(encoder_weights=None, freeze_encoder=True)
    with pytest.raises(ValueError, match="invalidos"):
        model.set_trainable_encoder_stages(("layer5",))


def test_model_configuration_declares_transfer_controls() -> None:
    random_config = model_config("deep_res_unet_resnet34_random_frozen.yaml")
    imagenet_config = model_config("deep_res_unet_resnet34_imagenet_frozen.yaml")
    assert random_config["architecture"] == imagenet_config["architecture"]
    assert random_config["freeze_encoder"] is True
    assert imagenet_config["freeze_encoder"] is True
    assert random_config["encoder_weights"] is None
    assert imagenet_config["encoder_weights"] == "imagenet"
    assert (
        count_parameters(build_segmentation_model(random_config))["trainable"]
        < count_parameters(build_segmentation_model({**random_config, "freeze_encoder": False}))[
            "trainable"
        ]
    )
