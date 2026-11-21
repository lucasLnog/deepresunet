from pathlib import Path

import numpy as np
import pytest
import yaml

from tcc_segmentation.sam_refinement import (
    PROMPT_MODES,
    component_box,
    component_mask_input,
    connected_components_8,
    original_mask_to_canvas,
    refine_prediction_with_sam,
    shifted_network_logits,
    validate_day12_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FakePredictor:
    def __init__(self):
        self.image = None
        self.calls = []

    def set_image(self, image):
        self.image = image

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        output = np.zeros(self.image.shape[:2], dtype=bool)
        if "box" in kwargs:
            x0, y0, x1, y1 = np.asarray(kwargs["box"], dtype=int)
            output[y0:y1, x0:x1] = True
        else:
            output[2:4, 2:4] = True
        return output[None, ...], np.asarray([0.75]), np.zeros((1, 256, 256))


def load_config():
    return yaml.safe_load((ROOT / "configs/sam_vit_b_integration.yaml").read_text(encoding="utf-8"))


def test_frozen_day12_contract_is_valid():
    config = load_config()
    validate_day12_config(config)
    assert config["network"]["threshold"] == 0.18
    assert config["sam"]["model_type"] == "vit_b"
    assert config["implementation_revision"] == 2
    assert tuple(config["prompts"]["modes"]) == PROMPT_MODES
    assert config["evaluation"]["smoke_is_for_selection"] is False


def test_protected_manifest_is_rejected():
    config = load_config()
    config["data"]["manifest"] = "manifests/splits/dfuc_test.csv"
    with pytest.raises(ValueError, match="protegida"):
        validate_day12_config(config)


def test_components_and_boxes_preserve_multiple_regions():
    mask = np.zeros((8, 9), dtype=bool)
    mask[1:3, 2:5] = True
    mask[6, 8] = True
    components = connected_components_8(mask)
    assert len(components) == 2
    boxes = [component_box(component).tolist() for component in components]
    assert boxes == [[2.0, 1.0, 5.0, 3.0], [8.0, 6.0, 9.0, 7.0]]


def test_threshold_is_zero_in_shifted_logit_space():
    shifted = shifted_network_logits(np.asarray([[0.18]], dtype=np.float32), 0.18)
    assert float(shifted[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_mask_input_preserves_aspect_and_sam_bottom_right_padding():
    # Paisagem 4x8 vira região válida 128x256. O restante é padding inferior,
    # exatamente como o preprocessamento oficial do SAM 1.
    logits = np.ones((4, 8), dtype=np.float32)
    component = np.ones((4, 8), dtype=bool)
    prompt = component_mask_input(logits, component)
    assert prompt.shape == (1, 256, 256)
    assert np.allclose(prompt[0, :128], 1.0)
    assert np.allclose(prompt[0, 128:], -10.0)


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_empty_network_prediction_stays_empty_without_calling_sam(mode):
    predictor = FakePredictor()
    output, audit = refine_prediction_with_sam(
        predictor=predictor,
        rgb_image=np.zeros((8, 9, 3), dtype=np.uint8),
        network_probabilities=np.zeros((8, 9), dtype=np.float32),
        network_threshold=0.18,
        mode=mode,
        max_components_safety=64,
    )
    assert not output.any()
    assert not predictor.calls
    assert audit["ground_truth_used_for_prompt"] is False
    assert audit["empty_prediction_policy_applied"] is True


@pytest.mark.parametrize("mode", PROMPT_MODES)
def test_each_component_generates_one_prompt_and_union(mode):
    probability = np.zeros((8, 9), dtype=np.float32)
    probability[1:3, 2:5] = 0.9
    probability[6, 8] = 0.9
    predictor = FakePredictor()
    output, audit = refine_prediction_with_sam(
        predictor=predictor,
        rgb_image=np.zeros((8, 9, 3), dtype=np.uint8),
        network_probabilities=probability,
        network_threshold=0.18,
        mode=mode,
        max_components_safety=64,
    )
    assert output.any()
    assert len(predictor.calls) == 2
    assert audit["component_count"] == 2
    assert audit["prompt_source"] == "DR-IFT prediction only"
    for call in predictor.calls:
        assert ("box" in call) == (mode in {"box", "box_plus_mask"})
        assert ("mask_input" in call) == (mode in {"mask_input", "box_plus_mask"})
        if "mask_input" in call:
            assert call["mask_input"].shape == (1, 256, 256)


def test_original_mask_returns_to_centered_canvas():
    original = np.ones((4, 8), dtype=bool)
    canvas = original_mask_to_canvas(original, target_height=8, target_width=8)
    assert canvas.shape == (8, 8)
    assert canvas[:2].sum() == 0
    assert canvas[2:6].all()
    assert canvas[6:].sum() == 0
