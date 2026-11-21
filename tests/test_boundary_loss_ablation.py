from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tcc_segmentation.boundary_loss_ablation import (
    decision_from_seed_deltas,
    validate_boundary_ablation_config,
)


ROOT = Path(__file__).resolve().parents[1]


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        result[key] = (
            merge(result[key], value)
            if key in result and isinstance(result[key], dict) and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def load_recursive(name: str):
    raw = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    base_name = raw.pop("base_config", None)
    if base_name is None:
        return raw
    return merge(load_recursive(base_name), raw)


@pytest.mark.parametrize(
    "name,seed",
    [
        ("configs/boundary_loss_ablation_seed42.yaml", 42),
        ("configs/boundary_loss_ablation_seed1337.yaml", 1337),
        ("configs/boundary_loss_ablation_seed2026.yaml", 2026),
    ],
)
def test_boundary_configs_are_frozen_and_valid(name, seed):
    config = load_recursive(name)
    baseline = load_recursive("configs/augmentation_d0_a1.yaml")
    validate_boundary_ablation_config(config, baseline_config=baseline)
    assert config["seed"] == seed
    assert config["loss"]["perimeter_weight"] == 0.1
    assert config["data"]["height"] == 384
    assert config["data"]["width"] == 512


def test_only_loss_differs_from_matched_drift_seed42():
    candidate = load_recursive("configs/boundary_loss_ablation_seed42.yaml")
    baseline = load_recursive("configs/fine_tuning_progressive.yaml")
    ignored = {
        "protocol_amendment",
        "stage",
        "experiment_id",
        "scientific_role",
        "loss",
        "boundary_ablation",
        "runtime",
    }
    for key in set(baseline) - ignored:
        assert candidate[key] == baseline[key], key


def test_decision_rule_requires_effect_size_and_two_seed_wins():
    assert decision_from_seed_deltas([0.006, 0.005, 0.004])["advance"]
    assert not decision_from_seed_deltas([0.004, 0.004, 0.004])["advance"]
    assert not decision_from_seed_deltas([0.020, -0.002, -0.002])["advance"]


def test_boundary_loss_is_finite_and_differentiable_when_torch_available():
    torch = pytest.importorskip("torch")
    from tcc_segmentation.training import BCEDicePerimeterLoss

    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 4:12, 5:11] = 1.0
    criterion = BCEDicePerimeterLoss()
    components = criterion.components(logits, targets)
    assert set(components) == {"total", "bce", "dice_loss", "perimeter_loss"}
    assert all(torch.isfinite(value) for value in components.values())
    components["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_boundary_weight_is_not_searchable():
    config = load_recursive("configs/boundary_loss_ablation_seed42.yaml")
    config["loss"]["perimeter_weight"] = 0.2
    with pytest.raises(ValueError, match="loss"):
        validate_boundary_ablation_config(config)
