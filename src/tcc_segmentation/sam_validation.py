"""Configuração e análise estatística do SAM na validação.

Este módulo não carrega imagens nem modelos. Ele mantém a regra de seleção e as
comparações pareadas separadas da inferência, o que permite testá-las localmente.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

BASELINE_VARIANT = "DR-IFT_without_SAM"
SAM_VARIANTS = ("SAM_box", "SAM_mask_input", "SAM_box_plus_mask")
VARIANTS = (BASELINE_VARIANT, *SAM_VARIANTS)
SAM_REPOSITORY_COMMIT = "6fdee8f2727f4506cfbbe553e23b895e27956588"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"
METRICS = (
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "boundary_iou",
)
PRIMARY_METRIC = "dice"


def validate_day13_config(config: Mapping[str, Any]) -> None:
    if config.get("stage") != "day13_sam_validation_selection":
        raise ValueError("Etapa inválida para a avaliação do SAM")
    if int(config.get("implementation_revision", -1)) != 2:
        raise ValueError("A avaliação do SAM na validação exige a revisão geométrica 2 do SAM")

    network = config.get("network", {})
    if network.get("experiment_id") != "DR-IFT" or int(network.get("seed", -1)) != 42:
        raise ValueError("A avaliação do SAM na validação exige o DR-IFT representativo da seed 42")
    if int(network.get("checkpoint_epoch", -1)) != 56:
        raise ValueError("checkpoint DR-IFT deve permanecer na época 56")
    if not math.isclose(float(network.get("threshold", -1)), 0.18, abs_tol=1e-12):
        raise ValueError("threshold DR-IFT deve permanecer congelado em 0,18")

    sam = config.get("sam", {})
    if sam.get("family") != "SAM 1" or sam.get("model_type") != "vit_b":
        raise ValueError("A avaliação do SAM na validação exige SAM 1 ViT-B")
    if sam.get("checkpoint_filename") != SAM_CHECKPOINT_FILENAME:
        raise ValueError("checkpoint SAM ViT-B não reconhecido")
    if sam.get("repository_commit") != SAM_REPOSITORY_COMMIT:
        raise ValueError("commit do SAM difere da versão congelada")
    if not bool(sam.get("weights_frozen")) or bool(sam.get("fine_tuning")):
        raise ValueError("pesos do SAM devem permanecer congelados")
    if int(sam.get("image_encoder_size", -1)) != 1024:
        raise ValueError("lado longo do encoder SAM deve ser 1024")

    prompts = config.get("prompts", {})
    if tuple(prompts.get("modes", ())) != ("box", "mask_input", "box_plus_mask"):
        raise ValueError("os três modos do smoke devem ser preservados")
    if prompts.get("source") != "DR-IFT prediction only" or bool(
        prompts.get("ground_truth_allowed")
    ):
        raise ValueError("prompts devem vir somente da DR-IFT")
    if not math.isclose(float(prompts.get("threshold_for_components", -1)), 0.18, abs_tol=1e-12):
        raise ValueError("threshold dos componentes deve acompanhar a DR-IFT")
    if prompts.get("multiple_components") != "one_prompt_per_component_union_outputs":
        raise ValueError("política de múltiplos componentes inválida")
    if prompts.get("empty_prediction") != "empty_output_without_sam_call":
        raise ValueError("política de predição vazia inválida")
    if int(prompts.get("max_components_safety", 0)) != 64:
        raise ValueError("limite de segurança de componentes foi alterado")
    if prompts.get("mask_input_alignment") != ("sam_resize_longest_side_then_bottom_right_padding"):
        raise ValueError("alinhamento do mask_input não corresponde à revisão 2")

    data = config.get("data", {})
    if data.get("split") != "validation" or int(data.get("expected_records", -1)) != 294:
        raise ValueError(
            "A avaliação do SAM na validação exige as 294 imagens da validação congelada"
        )
    manifest = str(data.get("manifest", "")).casefold()
    if any(marker in manifest for marker in ("test", "external", "fuseg")):
        raise ValueError("partição protegida bloqueada")
    fixed_manifest = str(data.get("fixed_visualization_manifest", "")).casefold()
    if any(marker in fixed_manifest for marker in ("test", "external", "fuseg")):
        raise ValueError("manifest visual protegido bloqueado")
    if int(data.get("expected_fixed_cases", -1)) != 12:
        raise ValueError("A avaliação do SAM na validação exige os 12 casos visuais congelados")

    evaluation = config.get("evaluation", {})
    if evaluation.get("baseline") != BASELINE_VARIANT:
        raise ValueError("baseline do SAM inválido")
    if tuple(evaluation.get("variants", ())) != SAM_VARIANTS:
        raise ValueError("candidatos SAM diferentes dos aprovados na integração técnica do SAM")
    if evaluation.get("primary_metric") != "macro_dice_per_image":
        raise ValueError("métrica primária deve ser Dice macro por imagem")
    if bool(evaluation.get("threshold_optimization_allowed")):
        raise ValueError("o SAM retorna máscara binária; não há novo threshold")
    if evaluation.get("selection_rule") != "highest_validation_macro_dice":
        raise ValueError("regra de seleção do modo SAM inválida")
    if tuple(evaluation.get("tie_break_order", ())) != SAM_VARIANTS:
        raise ValueError("ordem de desempate do SAM inválida")
    if not math.isclose(float(evaluation.get("plateau_tolerance", -1)), 0.0001):
        raise ValueError("tolerância de empate deve permanecer em 0,0001")
    if not math.isclose(float(evaluation.get("boundary_dilation_ratio", -1)), 0.02):
        raise ValueError("Boundary IoU deve usar faixa de 2% da diagonal")
    if not bool(evaluation.get("reference_used_only_after_prediction")):
        raise ValueError("a referência só pode entrar depois da predição")
    if not bool(evaluation.get("final_effect_requires_locked_test")):
        raise ValueError("efeito final deve permanecer reservado ao teste")

    statistics_config = config.get("statistics", {})
    if statistics_config.get("statistical_unit") != "canonical_image":
        raise ValueError("unidade estatística deve ser a imagem canônica")
    if int(statistics_config.get("bootstrap_samples", -1)) != 10_000:
        raise ValueError("quantidade de reamostragens bootstrap foi alterada")
    if int(statistics_config.get("permutation_samples", -1)) != 50_000:
        raise ValueError("quantidade de permutações foi alterada")
    if statistics_config.get("multiplicity") != ("Holm_across_three_primary_Dice_contrasts"):
        raise ValueError("correção de multiplicidade inválida")


def mask_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Extrai a faixa interna de borda conforme a geometria do Boundary IoU."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("máscara 2D esperada")
    if dilation_ratio <= 0:
        raise ValueError("dilation_ratio deve ser positivo")
    if not binary.any():
        return np.zeros_like(binary)
    height, width = binary.shape
    radius = max(1, int(round(dilation_ratio * math.hypot(height, width))))
    padded = np.pad(binary.astype(np.uint8), radius, constant_values=0)
    integral = np.pad(padded, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)
    kernel = 2 * radius + 1
    window_sum = (
        integral[kernel:, kernel:]
        - integral[:-kernel, kernel:]
        - integral[kernel:, :-kernel]
        + integral[:-kernel, :-kernel]
    )
    eroded = window_sum == kernel * kernel
    return binary & ~eroded


def boundary_iou(
    prediction: np.ndarray, reference: np.ndarray, dilation_ratio: float = 0.02
) -> float:
    pred_boundary = mask_boundary(prediction, dilation_ratio)
    ref_boundary = mask_boundary(reference, dilation_ratio)
    union = int((pred_boundary | ref_boundary).sum())
    return float((pred_boundary & ref_boundary).sum() / union) if union else 1.0


def aggregate_variant_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    if set(grouped) != set(VARIANTS):
        raise ValueError("conjunto de variantes incompleto")
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        group = grouped[variant]
        item: dict[str, Any] = {
            "variant": variant,
            "images": len(group),
            "zero_dice_images": sum(float(row["dice"]) == 0.0 for row in group),
            "mean_network_seconds": statistics.fmean(
                float(row["network_seconds"]) for row in group
            ),
            "mean_sam_seconds": statistics.fmean(float(row["sam_seconds"]) for row in group),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 0:
        raise ValueError("lista vazia")
    position = (len(sorted_values) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_ci(
    differences: Sequence[float], samples: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values) or samples < 1:
        raise ValueError("bootstrap requer diferenças e amostras")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = 1000
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    means.sort()
    return percentile(means, 0.025), percentile(means, 0.975)


def paired_sign_flip_pvalue(differences: Sequence[float], samples: int, seed: int) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values) or samples < 1:
        raise ValueError("permutação requer diferenças e amostras")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 1000
    for start in range(0, samples, chunk):
        current = min(chunk, samples - start)
        signs = rng.integers(0, 2, size=(current, len(values)), dtype=np.int8)
        signed_means = ((signs * 2 - 1) * values).mean(axis=1)
        extreme += int((np.abs(signed_means) >= observed - 1e-15).sum())
    return float((extreme + 1) / (samples + 1))


def holm_adjust(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (total - rank) * float(value))
        adjusted[name] = min(1.0, running)
    return adjusted


def index_rows(
    rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Mapping[str, Any]]]]:
    by_variant: dict[str, dict[str, Mapping[str, Any]]] = {variant: {} for variant in VARIANTS}
    order: list[str] = []
    for row in rows:
        variant, canonical_id = str(row["variant"]), str(row["canonical_id"])
        if variant not in by_variant:
            raise ValueError(f"variante desconhecida: {variant}")
        if canonical_id in by_variant[variant]:
            raise ValueError(f"par duplicado: {canonical_id}/{variant}")
        by_variant[variant][canonical_id] = row
        if variant == BASELINE_VARIANT:
            order.append(canonical_id)
    if len(order) != len(set(order)):
        raise ValueError("IDs duplicados no baseline")
    for variant in VARIANTS:
        if set(by_variant[variant]) != set(order):
            raise ValueError(f"IDs de {variant} diferem do baseline")
    return order, by_variant


def paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids, indexed = index_rows(rows)
    comparisons: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = [{"canonical_id": item} for item in ids]
    raw_p: dict[str, float] = {}
    for variant_index, variant in enumerate(SAM_VARIANTS):
        comparison = f"{variant}_minus_{BASELINE_VARIANT}"
        for metric_index, metric in enumerate(METRICS):
            differences = [
                float(indexed[variant][item][metric])
                - float(indexed[BASELINE_VARIANT][item][metric])
                for item in ids
            ]
            low, high = paired_bootstrap_ci(
                differences,
                bootstrap_samples,
                seed + variant_index * 100 + metric_index,
            )
            row = {
                "comparison": comparison,
                "metric": metric,
                "mean_difference": statistics.fmean(differences),
                "median_difference": statistics.median(differences),
                "ci95_low": low,
                "ci95_high": high,
                "improved": sum(value > 0 for value in differences),
                "equal": sum(value == 0 for value in differences),
                "worsened": sum(value < 0 for value in differences),
                "permutation_p_raw": None,
                "permutation_p_holm": None,
            }
            if metric == PRIMARY_METRIC:
                p_value = paired_sign_flip_pvalue(
                    differences,
                    permutation_samples,
                    seed + 10_000 + variant_index,
                )
                row["permutation_p_raw"] = p_value
                raw_p[comparison] = p_value
            comparisons.append(row)
            for item_index, difference in enumerate(differences):
                per_image[item_index][f"{metric}_{comparison}"] = difference
    adjusted = holm_adjust(raw_p)
    for row in comparisons:
        if row["metric"] == PRIMARY_METRIC:
            row["permutation_p_holm"] = adjusted[row["comparison"]]
    return comparisons, per_image


def select_sam_variant(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    tie_break_order: Sequence[str],
) -> dict[str, Any]:
    scores = {
        str(row["variant"]): float(row["dice_mean"])
        for row in summary_rows
        if str(row["variant"]) in SAM_VARIANTS
    }
    if set(scores) != set(SAM_VARIANTS):
        raise ValueError("resumo não contém os três candidatos SAM")
    if tuple(tie_break_order) != SAM_VARIANTS:
        raise ValueError("ordem de desempate não está congelada")
    best = max(scores.values())
    plateau = [variant for variant in tie_break_order if best - scores[variant] <= tolerance]
    selected = plateau[0]
    return {
        "selection_rule": "highest_validation_macro_dice",
        "plateau_tolerance": float(tolerance),
        "tie_break_order": list(tie_break_order),
        "scores": scores,
        "best_observed_score": best,
        "plateau_candidates": plateau,
        "selected_variant": selected,
    }


def subgroup_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), int(row["foreground_area_bin"]))].append(row)
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for area_bin in range(5):
            group = grouped.get((variant, area_bin), [])
            if not group:
                raise ValueError(f"estrato ausente: {variant}/área {area_bin}")
            item: dict[str, Any] = {
                "variant": variant,
                "foreground_area_bin": area_bin,
                "images": len(group),
            }
            for metric in METRICS:
                item[f"{metric}_mean"] = statistics.fmean(float(row[metric]) for row in group)
            output.append(item)
    return output
