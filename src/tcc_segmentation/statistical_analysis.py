"""Funções de análise estatística dos resultados de segmentação.

Esta etapa trabalha apenas com as tabelas finais ja produzidas. Ela nao abre
imagens, nao executa inferencia e nao altera thresholds, checkpoints ou modelos.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


COUNT_FIELDS = ("tp", "fp", "fn", "tn")


def discover_result_directory(
    search_roots: Sequence[str | Path],
    *,
    marker: str,
    required_files: Sequence[str],
) -> Path:
    """Encontra a execução completa mais recente sem assumir o nome da pasta."""
    candidates: list[Path] = []
    seen: set[Path] = set()
    for value in search_roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            continue
        for marker_path in root.rglob(marker):
            directory = marker_path.parent.resolve()
            if directory in seen:
                continue
            seen.add(directory)
            if all((directory / filename).is_file() for filename in required_files):
                candidates.append(directory)
    if not candidates:
        roots = ", ".join(str(Path(value)) for value in search_roots)
        raise FileNotFoundError(
            f"Execucao com {marker} nao encontrada em: {roots}. "
            "Confirme que o pacote final foi extraido."
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _divide(numerator: float, denominator: float, fallback: float = 1.0) -> float:
    return float(numerator / denominator) if denominator else float(fallback)


def micro_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Calcula metricas data-based apos agregar as contagens das unidades."""
    if not rows:
        raise ValueError("Nao e possivel agregar uma colecao vazia")
    totals = {field: float(sum(float(row[field]) for row in rows)) for field in COUNT_FIELDS}
    tp, fp, fn, tn = (totals[field] for field in COUNT_FIELDS)
    precision = _divide(tp, tp + fp, 0.0 if tp + fn else 1.0)
    recall = _divide(tp, tp + fn)
    specificity = _divide(tn, tn + fp)
    return {
        **totals,
        "dice": _divide(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": _divide(tp, tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": _divide(tp + tn, tp + fp + fn + tn),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def bootstrap_macro_micro_dice_ci(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, tuple[float, float]]:
    """IC percentil por reamostragem da unidade estatistica."""
    if not rows or samples < 1:
        raise ValueError("Amostra ou numero de reamostragens invalido")
    dice = np.asarray([float(row["dice"]) for row in rows], dtype=np.float64)
    counts = np.asarray(
        [[float(row[field]) for field in COUNT_FIELDS] for row in rows],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    macro = np.empty(samples, dtype=np.float64)
    micro = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        indices = rng.integers(0, len(rows), size=(size, len(rows)))
        macro[start : start + size] = dice[indices].mean(axis=1)
        sampled = counts[indices].sum(axis=1)
        tp, fp, fn = sampled[:, 0], sampled[:, 1], sampled[:, 2]
        denominator = 2.0 * tp + fp + fn
        micro[start : start + size] = np.divide(
            2.0 * tp,
            denominator,
            out=np.ones_like(tp),
            where=denominator != 0,
        )
    macro_ci = tuple(float(value) for value in np.quantile(macro, [0.025, 0.975]))
    micro_ci = tuple(float(value) for value in np.quantile(micro, [0.025, 0.975]))
    return {"macro": macro_ci, "micro": micro_ci}


def independent_macro_gap_ci(
    internal_rows: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """IC do gap externo menos interno, com reamostragem independente."""
    if not internal_rows or not external_rows or samples < 1:
        raise ValueError("Entradas invalidas para o bootstrap independente")
    internal = np.asarray([float(row["dice"]) for row in internal_rows])
    external = np.asarray([float(row["dice"]) for row in external_rows])
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        size = min(256, samples - start)
        internal_indices = rng.integers(0, len(internal), size=(size, len(internal)))
        external_indices = rng.integers(0, len(external), size=(size, len(external)))
        differences[start : start + size] = external[external_indices].mean(axis=1) - internal[
            internal_indices
        ].mean(axis=1)
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def relative_area_quintiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Atribui quintis por posto, independentemente do desempenho do modelo."""
    fractions: dict[str, float] = {}
    for row in rows:
        identifier = str(row["canonical_id"])
        value = float(row["foreground_fraction"])
        if identifier in fractions and not np.isclose(fractions[identifier], value):
            raise ValueError(f"Area inconsistente para {identifier}")
        fractions[identifier] = value
    ordered = sorted(fractions, key=lambda item: (fractions[item], item))
    size = len(ordered)
    if not size:
        raise ValueError("Nao ha unidades para estratificar")
    return {
        identifier: min(5, int(index * 5 / size) + 1) for index, identifier in enumerate(ordered)
    }


def error_category(row: Mapping[str, Any]) -> str:
    """Taxonomia descritiva baseada apenas nas metricas da unidade.

    Os rotulos 'recall limitado' e 'precisao limitada' indicam tendencia, nao uma
    adjudicacao clinica do mecanismo do erro.
    """
    prediction_pixels = float(row.get("prediction_pixels", 0.0))
    tp = float(row["tp"])
    dice = float(row["dice"])
    precision = float(row["precision"])
    recall = float(row["recall"])
    if prediction_pixels == 0:
        return "predicao_vazia"
    if tp == 0:
        return "sem_sobreposicao"
    if dice >= 0.75:
        return "alta_sobreposicao"
    if dice >= 0.50:
        return "sobreposicao_parcial"
    if recall < precision:
        return "baixo_dice_recall_limitado"
    if precision < recall:
        return "baixo_dice_precisao_limitada"
    return "baixo_dice_misto"


def summarize_rows(rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    """Retorna a sintese macro e micro de um conjunto de unidades."""
    if not rows:
        raise ValueError("Grupo vazio")
    micro = micro_metrics(rows)
    ci = bootstrap_macro_micro_dice_ci(rows, samples=samples, seed=seed)
    return {
        "units": len(rows),
        "zero_dice_units": sum(float(row["dice"]) == 0.0 for row in rows),
        "dice_macro": statistics.fmean(float(row["dice"]) for row in rows),
        "dice_macro_median": statistics.median(float(row["dice"]) for row in rows),
        "dice_macro_ci95_low": ci["macro"][0],
        "dice_macro_ci95_high": ci["macro"][1],
        "dice_micro": micro["dice"],
        "dice_micro_ci95_low": ci["micro"][0],
        "dice_micro_ci95_high": ci["micro"][1],
        "iou_macro": statistics.fmean(float(row["iou"]) for row in rows),
        "iou_micro": micro["iou"],
        "precision_macro": statistics.fmean(float(row["precision"]) for row in rows),
        "precision_micro": micro["precision"],
        "recall_macro": statistics.fmean(float(row["recall"]) for row in rows),
        "recall_micro": micro["recall"],
        "balanced_accuracy_macro": statistics.fmean(
            float(row["balanced_accuracy"]) for row in rows
        ),
        "balanced_accuracy_micro": micro["balanced_accuracy"],
        "boundary_iou_macro": statistics.fmean(float(row["boundary_iou"]) for row in rows),
    }
