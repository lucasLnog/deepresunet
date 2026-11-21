from __future__ import annotations

import copy
from pathlib import Path

import yaml

from tcc_segmentation.fine_tuning import (
    stage_for_epoch,
    validate_fine_tuning_config,
    validate_fine_tuning_pair,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    (1337, "DR-IF"): "fine_tuning_frozen_encoder_seed1337.yaml",
    (1337, "DR-IFT"): "fine_tuning_progressive_seed1337.yaml",
    (2026, "DR-IF"): "fine_tuning_frozen_encoder_seed2026.yaml",
    (2026, "DR-IFT"): "fine_tuning_progressive_seed2026.yaml",
}


def merge(left, right):
    output = copy.deepcopy(left)
    for key, value in right.items():
        output[key] = (
            merge(output[key], value)
            if key in output and isinstance(output[key], dict) and isinstance(value, dict)
            else value
        )
    return output


def load(name):
    raw = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / raw.pop("base_config")).read_text(encoding="utf-8"))
    return merge(base, raw)


def test_each_seed_is_a_controlled_pair():
    for seed in (1337, 2026):
        fixed = load(CONFIGS[(seed, "DR-IF")])
        progressive = load(CONFIGS[(seed, "DR-IFT")])
        validate_fine_tuning_pair(fixed, progressive)
        assert fixed["seed"] == progressive["seed"] == seed
        assert fixed["augmentation_randomness"] == progressive["augmentation_randomness"]
        assert fixed["model"] == progressive["model"]


def test_seeds_change_all_declared_random_streams_and_outputs():
    fixed_1337 = load(CONFIGS[(1337, "DR-IF")])
    fixed_2026 = load(CONFIGS[(2026, "DR-IF")])
    assert fixed_1337["seed"] != fixed_2026["seed"]
    assert (
        fixed_1337["augmentation_randomness"]["seed_base"]
        != fixed_2026["augmentation_randomness"]["seed_base"]
    )
    assert fixed_1337["runtime"]["output_root"] != fixed_2026["runtime"]["output_root"]


def test_architecture_and_progressive_schedule_remain_frozen():
    for (seed, arm), name in CONFIGS.items():
        config = load(name)
        validate_fine_tuning_config(config)
        assert config["model"]["classes"] == 1
        assert config["data"]["augmentation"] == "A1"
        assert config["data"]["color_preprocessing"] == "none"
        if arm == "DR-IFT":
            assert stage_for_epoch(config, 16) == ("layer4_released", ("layer4",))
            assert stage_for_epoch(config, 31) == (
                "layer3_layer4_released",
                ("layer3", "layer4"),
            )


def test_runner_adapts_canonical_peer_to_replication_seed():
    source = (ROOT / "scripts" / "run_baseline_d0.py").read_text(encoding="utf-8")
    assert 'peer_config["seed"] = config["seed"]' in source
    assert 'peer_config["augmentation_randomness"] = config["augmentation_randomness"]' in source
