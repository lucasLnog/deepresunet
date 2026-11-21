import copy
from pathlib import Path

import pytest
import yaml

from tcc_segmentation.fine_tuning import (
    stage_for_epoch,
    validate_fine_tuning_config,
    validate_fine_tuning_pair,
)

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    base = yaml.safe_load((ROOT / "configs/augmentation_d0_a1.yaml").read_text(encoding="utf-8"))
    arm = yaml.safe_load((ROOT / f"configs/{name}").read_text(encoding="utf-8"))
    arm.pop("base_config")

    def merge(a, b):
        out = copy.deepcopy(a)
        for k, v in b.items():
            out[k] = (
                merge(out[k], v)
                if k in out and isinstance(out[k], dict) and isinstance(v, dict)
                else v
            )
        return out

    return merge(base, arm)


def test_pair_is_controlled():
    fixed, progressive = load("fine_tuning_frozen_encoder.yaml"), load("fine_tuning_progressive.yaml")
    validate_fine_tuning_pair(fixed, progressive)
    assert fixed["model"] == progressive["model"]


def test_progressive_schedule_boundaries():
    cfg = load("fine_tuning_progressive.yaml")
    assert stage_for_epoch(cfg, 1) == ("decoder_warmup", ())
    assert stage_for_epoch(cfg, 15) == ("decoder_warmup", ())
    assert stage_for_epoch(cfg, 16) == ("layer4_released", ("layer4",))
    assert stage_for_epoch(cfg, 30) == ("layer4_released", ("layer4",))
    assert stage_for_epoch(cfg, 31) == ("layer3_layer4_released", ("layer3", "layer4"))


def test_fixed_never_releases_encoder():
    cfg = load("fine_tuning_frozen_encoder.yaml")
    for epoch in (1, 16, 31, 100):
        assert stage_for_epoch(cfg, epoch) == ("encoder_fixed", ())


def test_early_stopping_cannot_precede_third_phase():
    cfg = load("fine_tuning_progressive.yaml")
    cfg["fine_tuning"]["minimum_epochs_before_stopping"] = 30
    with pytest.raises(ValueError, match="31"):
        validate_fine_tuning_config(cfg)


def test_model_stage_trainability_when_torchvision_available():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tcc_segmentation.models import build_segmentation_model

    cfg = load("fine_tuning_progressive.yaml")
    model_config = {**cfg["model"], "encoder_weights": None}
    model = build_segmentation_model(model_config)
    model.set_trainable_encoder_stages(("layer4",))
    model.train()
    assert any(p.requires_grad for p in model.layer4.parameters())
    assert not any(p.requires_grad for p in model.layer3.parameters())
    assert not model.stem_bn.training


def test_runner_uses_normalized_panel_for_both_fine_tuning_arms():
    source = (ROOT / "scripts/run_baseline_d0.py").read_text(encoding="utf-8")
    assert "if (is_transfer_arm or is_fine_tuning_arm)" in source
