"""Configuração e análise estatística da ablação MedSAM na validação.

O módulo mantém a inferência separada das decisões estatísticas. A DR-IFT,
os três modos do SAM 1 e o MedSAM com caixa são avaliados nas mesmas imagens.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from tcc_segmentation.sam_validation import (
    METRICS,
    holm_adjust,
    paired_bootstrap_ci,
    paired_sign_flip_pvalue,
)

BASELINE_VARIANT = "DR-IFT_without_SAM"
GENERIC_SAM_VARIANTS = ("SAM_box", "SAM_mask_input", "SAM_box_plus_mask")
MEDSAM_VARIANT = "MedSAM_box"
VARIANTS = (BASELINE_VARIANT, *GENERIC_SAM_VARIANTS, MEDSAM_VARIANT)
PRIMARY_METRIC = "dice"

SAM_REPOSITORY_COMMIT = "6fdee8f2727f4506cfbbe553e23b895e27956588"
SAM_CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
MEDSAM_CHECKPOINT_FILENAME = "medsam_vit_b.pth"
MEDSAM_CHECKPOINT_SHA256 = "34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74"


def validate_day13b_config(config: Mapping[str, Any]) -> None:
    if config.get("stage") != "day13b_medsam_validation_ablation":
        raise ValueError("Etapa inválida para a avaliação do MedSAM")
    if int(config.get("protocol_version", -1)) != 1:
        raise ValueError("versão do protocolo da avaliação do MedSAM na validação inválida")

    network = config.get("network", {})
    if network.get("experiment_id") != "DR-IFT" or int(network.get("seed", -1)) != 42:
        raise ValueError(
            "A avaliação do MedSAM na validação exige o DR-IFT representativo da seed 42"
        )
    if int(network.get("checkpoint_epoch", -1)) != 56:
        raise ValueError("checkpoint DR-IFT deve permanecer na época 56")
    if not math.isclose(float(network.get("threshold", -1)), 0.18, abs_tol=1e-12):
        raise ValueError("threshold DR-IFT deve permanecer congelado em 0,18")

    sam = config.get("sam", {})
    if sam.get("family") != "SAM 1" or sam.get("model_type") != "vit_b":
        raise ValueError("comparador genérico deve ser SAM 1 ViT-B")
    if sam.get("repository_commit") != SAM_REPOSITORY_COMMIT:
        raise ValueError("commit do SAM difere da versão congelada")
    if sam.get("checkpoint_sha256") != SAM_CHECKPOINT_SHA256:
        raise ValueError("hash do checkpoint SAM genérico foi alterado")
    if not bool(sam.get("weights_frozen")) or bool(sam.get("fine_tuning")):
        raise ValueError("pesos do SAM devem permanecer congelados")

    medsam = config.get("medsam", {})
    if medsam.get("family") != "MedSAM" or medsam.get("model_type") != "vit_b":
        raise ValueError("A avaliação do MedSAM na validação exige MedSAM ViT-B")
    if medsam.get("checkpoint_filename") != MEDSAM_CHECKPOINT_FILENAME:
        raise ValueError("checkpoint MedSAM não reconhecido")
    if medsam.get("checkpoint_sha256") != MEDSAM_CHECKPOINT_SHA256:
        raise ValueError("hash do checkpoint MedSAM foi alterado")
    if medsam.get("prompt_mode") != "box":
        raise ValueError("MedSAM deve usar somente o prompt oficial por caixa")
    if medsam.get("preprocessing") != "official_square_1024_minmax":
        raise ValueError("pré-processamento oficial do MedSAM foi alterado")
    if not bool(medsam.get("weights_frozen")) or bool(medsam.get("fine_tuning")):
        raise ValueError("pesos do MedSAM devem permanecer congelados")

    prompts = config.get("prompts", {})
    if prompts.get("source") != "DR-IFT prediction only" or bool(
        prompts.get("ground_truth_allowed")
    ):
        raise ValueError("prompts devem vir somente da DR-IFT")
    if not math.isclose(float(prompts.get("threshold_for_components", -1)), 0.18, abs_tol=1e-12):
        raise ValueError("threshold dos componentes deve acompanhar a DR-IFT")
    if prompts.get("multiple_components") != "one_prompt_per_component_union_outputs":
        raise ValueError("política de múltiplos componentes inválida")
    if prompts.get("empty_prediction") != "empty_output_without_model_call":
        raise ValueError("política de predição vazia inválida")
    if int(prompts.get("max_components_safety", 0)) != 64:
        raise ValueError("limite de segurança de componentes foi alterado")

    data = config.get("data", {})
    if data.get("split") != "validation" or int(data.get("expected_records", -1)) != 294:
        raise ValueError(
            "A avaliação do MedSAM na validação exige as 294 imagens da validação congelada"
        )
    for key in ("manifest", "fixed_visualization_manifest"):
        value = str(data.get(key, "")).casefold()
        if any(marker in value for marker in ("test", "external", "fuseg")):
            raise ValueError("partição protegida bloqueada")
    if int(data.get("expected_fixed_cases", -1)) != 12:
        raise ValueError("A avaliação do MedSAM na validação exige os 12 casos visuais congelados")

    evaluation = config.get("evaluation", {})
    if tuple(evaluation.get("variants", ())) != VARIANTS:
        raise ValueError(
            "variantes da avaliação do MedSAM na validação diferem das pré-registradas"
        )
    if evaluation.get("primary_metric") != "macro_dice_per_image":
        raise ValueError("métrica primária deve ser Dice macro por imagem")
    if bool(evaluation.get("threshold_optimization_allowed")):
        raise ValueError("nenhum limiar pode ser ajustado na avaliação do MedSAM")
    if not bool(evaluation.get("reference_used_only_after_prediction")):
        raise ValueError("a referência só pode entrar depois da predição")
    if not bool(evaluation.get("final_effect_requires_locked_test")):
        raise ValueError("efeito final deve permanecer reservado ao teste")
    if not math.isclose(float(evaluation.get("boundary_dilation_ratio", -1)), 0.02):
        raise ValueError("Boundary IoU deve usar faixa de 2% da diagonal")

    advance = config.get("advance_rule", {})
    if advance.get("candidate") != MEDSAM_VARIANT:
        raise ValueError("regra de avanço deve se referir ao MedSAM_box")
    if advance.get("reference") != BASELINE_VARIANT:
        raise ValueError("regra de avanço deve comparar com a DR-IFT")
    if not math.isclose(float(advance.get("minimum_mean_dice_gain", -1)), 0.01):
        raise ValueError("ganho mínimo prá-registrado deve ser 0,01")
    if not bool(advance.get("require_ci95_low_above_zero")):
        raise ValueError("limite inferior do IC 95% deve ser maior que zero")

    stats = config.get("statistics", {})
    if stats.get("statistical_unit") != "canonical_image":
        raise ValueError("unidade estatística deve ser a imagem canônica")
    if int(stats.get("bootstrap_samples", -1)) != 10_000:
        raise ValueError("quantidade de reamostragens bootstrap foi alterada")
    if int(stats.get("permutation_samples", -1)) != 50_000:
        raise ValueError("quantidade de permutações foi alterada")
    if stats.get("multiplicity") != "Holm_across_four_SAM_vs_DRIFT_Dice_contrasts":
        raise ValueError("correção de multiplicidade da avaliação do MedSAM na validação inválida")


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
            "mean_postprocess_seconds": statistics.fmean(
                float(row["postprocess_seconds"]) for row in group
            ),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_median"] = statistics.median(values)
            item[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Mapping[str, Any]]]]:
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {variant: {} for variant in VARIANTS}
    order: list[str] = []
    for row in rows:
        variant = str(row["variant"])
        canonical_id = str(row["canonical_id"])
        if variant not in indexed:
            raise ValueError(f"variante desconhecida: {variant}")
        if canonical_id in indexed[variant]:
            raise ValueError(f"par duplicado: {canonical_id}/{variant}")
        indexed[variant][canonical_id] = row
        if variant == BASELINE_VARIANT:
            order.append(canonical_id)
    if len(order) != len(set(order)):
        raise ValueError("IDs duplicados no baseline")
    for variant in VARIANTS:
        if set(indexed[variant]) != set(order):
            raise ValueError(f"IDs de {variant} diferem do baseline")
    return order, indexed


def paired_comparisons(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids, indexed = _index_rows(rows)
    comparison_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = [{"canonical_id": item} for item in ids]
    raw_primary: dict[str, float] = {}

    comparison_pairs = [
        (variant, BASELINE_VARIANT) for variant in VARIANTS if variant != BASELINE_VARIANT
    ] + [
        (MEDSAM_VARIANT, "SAM_box"),
        (MEDSAM_VARIANT, "SAM_box_plus_mask"),
    ]
    for comparison_index, (candidate, reference) in enumerate(comparison_pairs):
        comparison = f"{candidate}_minus_{reference}"
        for metric_index, metric in enumerate(METRICS):
            differences = [
                float(indexed[candidate][item][metric]) - float(indexed[reference][item][metric])
                for item in ids
            ]
            low, high = paired_bootstrap_ci(
                differences,
                bootstrap_samples,
                seed + comparison_index * 100 + metric_index,
            )
            row = {
                "comparison": comparison,
                "candidate": candidate,
                "reference": reference,
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
                "primary_family": reference == BASELINE_VARIANT,
            }
            if metric == PRIMARY_METRIC:
                p_value = paired_sign_flip_pvalue(
                    differences,
                    permutation_samples,
                    seed + 10_000 + comparison_index,
                )
                row["permutation_p_raw"] = p_value
                if reference == BASELINE_VARIANT:
                    raw_primary[comparison] = p_value
            comparison_rows.append(row)
            for item_index, difference in enumerate(differences):
                difference_rows[item_index][f"{metric}_{comparison}"] = difference

    adjusted = holm_adjust(raw_primary)
    for row in comparison_rows:
        if row["metric"] == PRIMARY_METRIC and row["primary_family"]:
            row["permutation_p_holm"] = adjusted[row["comparison"]]
    return comparison_rows, difference_rows


def apply_advance_rule(
    comparison_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    comparison_name = f"{MEDSAM_VARIANT}_minus_{BASELINE_VARIANT}"
    effect = next(
        row
        for row in comparison_rows
        if row["comparison"] == comparison_name and row["metric"] == PRIMARY_METRIC
    )
    rule = config["advance_rule"]
    gain_ok = float(effect["mean_difference"]) >= float(rule["minimum_mean_dice_gain"])
    ci_ok = float(effect["ci95_low"]) > 0.0
    advance = gain_ok and ci_ok
    return {
        "candidate": MEDSAM_VARIANT,
        "reference": BASELINE_VARIANT,
        "metric": "macro_dice_per_image",
        "minimum_mean_dice_gain": float(rule["minimum_mean_dice_gain"]),
        "require_ci95_low_above_zero": True,
        "observed_effect": dict(effect),
        "gain_criterion_met": gain_ok,
        "uncertainty_criterion_met": ci_ok,
        "advance_to_locked_internal_test": advance,
        "if_not_advanced": "retain_as_validation_only_exploratory_ablation",
        "protected_partitions_accessed": False,
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
