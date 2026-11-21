from pathlib import Path

import pytest
import yaml


torch = pytest.importorskip("torch")

from src.tcc_segmentation.models import (  # noqa: E402
    DeepResUNet,
    build_segmentation_model,
    count_parameters,
)


ROOT = Path(__file__).resolve().parents[1]


def historical_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/models/deep_res_unet_historical.yaml").read_text(encoding="utf-8")
    )


def test_historical_parameter_count_is_preserved() -> None:
    model = build_segmentation_model(historical_config())
    assert count_parameters(model) == {"total": 8_212_029, "trainable": 8_212_029}


def test_deep_res_unet_returns_unbounded_logits_with_original_shape() -> None:
    model = build_segmentation_model(historical_config())
    model.eval()
    with torch.no_grad():
        model.final_projection.weight.zero_()
        model.final_projection.bias.fill_(2.0)
    with torch.inference_mode():
        output = model(torch.rand((1, 3, 64, 80)))
    assert output.shape == (1, 1, 64, 80)
    assert torch.allclose(output, torch.full_like(output, 2.0))


def test_deep_res_unet_rejects_incompatible_spatial_size() -> None:
    model = DeepResUNet()
    with pytest.raises(ValueError, match="divisiveis por 16"):
        model(torch.rand((1, 3, 65, 80)))


def test_historical_model_cannot_claim_pretrained_encoder() -> None:
    config = historical_config()
    config["encoder_weights"] = "imagenet"
    with pytest.raises(ValueError, match="nao possui encoder pre-treinado"):
        build_segmentation_model(config)
