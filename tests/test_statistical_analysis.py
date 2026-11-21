from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import pytest

from tcc_segmentation.statistical_analysis import (
    discover_result_directory,
    error_category,
    independent_macro_gap_ci,
    micro_metrics,
    relative_area_quintiles,
    summarize_rows,
)


def row(identifier: str, tp: int, fp: int, fn: int, tn: int, fraction: float):
    denominator = 2 * tp + fp + fn
    dice = 2 * tp / denominator if denominator else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    precision = tp / (tp + fp) if tp + fp else (0.0 if tp + fn else 1.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    return {
        "canonical_id": identifier,
        "foreground_fraction": fraction,
        "prediction_pixels": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "boundary_iou": iou,
    }


def test_micro_dice_uses_pooled_counts_not_mean_per_image() -> None:
    rows = [row("a", 1, 0, 0, 9, 0.1), row("b", 1, 9, 0, 0, 0.9)]
    result = micro_metrics(rows)
    assert result["dice"] == pytest.approx(4 / 13)
    assert result["dice"] != pytest.approx(sum(item["dice"] for item in rows) / 2)


def test_summary_bootstrap_is_deterministic() -> None:
    rows = [row(str(index), index + 1, 1, 1, 20, (index + 1) / 100) for index in range(8)]
    first = summarize_rows(rows, samples=200, seed=42)
    second = summarize_rows(rows, samples=200, seed=42)
    assert first == second
    assert first["dice_macro_ci95_low"] <= first["dice_macro"] <= first["dice_macro_ci95_high"]
    assert first["dice_micro_ci95_low"] <= first["dice_micro"] <= first["dice_micro_ci95_high"]


def test_relative_quintiles_depend_only_on_area_and_identifier() -> None:
    rows = [row(str(index), 1, 0, 0, 9, float(index)) for index in range(10)]
    bins = relative_area_quintiles(rows)
    assert [bins[str(index)] for index in range(10)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_error_taxonomy_distinguishes_limiting_metric_and_empty_prediction() -> None:
    assert error_category(row("empty", 0, 0, 4, 6, 0.4)) == "predicao_vazia"
    assert error_category(row("miss", 0, 2, 4, 4, 0.4)) == "sem_sobreposicao"
    assert error_category(row("under", 2, 0, 8, 90, 0.1)) == "baixo_dice_recall_limitado"
    assert error_category(row("over", 2, 8, 0, 90, 0.02)) == "baixo_dice_precisao_limitada"


def test_independent_gap_ci_is_deterministic_and_contains_constant_gap() -> None:
    internal = [row(str(index), 3, 1, 1, 10, 0.1) for index in range(6)]
    external = [row(str(index), 2, 1, 1, 10, 0.1) for index in range(8)]
    first = independent_macro_gap_ci(internal, external, samples=100, seed=7)
    second = independent_macro_gap_ci(internal, external, samples=100, seed=7)
    expected = external[0]["dice"] - internal[0]["dice"]
    assert first == second
    assert first[0] == pytest.approx(expected)
    assert first[1] == pytest.approx(expected)


def test_result_discovery_does_not_depend_on_parent_directory_name(tmp_path) -> None:
    older = tmp_path / "final_internal_test" / "RUN_OLDER"
    newer = tmp_path / "final_evaluation" / "RUN_NEWER"
    for directory in (older, newer):
        directory.mkdir(parents=True)
        for filename in ("per_image.csv", "summary.csv", "comparisons.csv"):
            (directory / filename).write_text("ok\n", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    result = discover_result_directory(
        [tmp_path / "final_evaluation", tmp_path],
        marker="per_image.csv",
        required_files=("per_image.csv", "summary.csv", "comparisons.csv"),
    )
    assert result == newer.resolve()


def test_figure_generator_requires_a_scalable_font() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/run_statistical_analysis.py"
    specification = importlib.util.spec_from_file_location("day16_runner", script)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    selected = module.font(32, bold=True)
    box = selected.getbbox("Análise")
    assert box[3] - box[1] >= 20
