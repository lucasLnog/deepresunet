#!/usr/bin/env python3
"""Consolida estatistica e erros dos resultados finais DFUC e FUSeg.

O script le apenas CSVs finais. Nao acessa datasets, imagens, pesos ou codigo de
inferencia e nao realiza selecao de modelos ou thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from tcc_segmentation.statistical_analysis import (
    error_category,
    independent_macro_gap_ci,
    relative_area_quintiles,
    summarize_rows,
)


COMMON_GROUPS = (
    ("DRIFT", "selected"),
    ("DRIFT", "fixed_0_5"),
    ("DRIFT_SAM_BPM", "selected"),
)
LABELS = {
    ("DRIFT", "selected"): "DR-IFT (threshold congelado)",
    ("DRIFT", "fixed_0_5"): "DR-IFT (0,5; sensibilidade)",
    ("DRIFT_SAM_BPM", "selected"): "DR-IFT + SAM",
}
COLORS = {"DFUC2022 interno": "#6d28d9", "FUSeg externo": "#0891b2"}


def font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise RuntimeError(
        "Fonte TrueType escalavel nao encontrada. "
        "Instale DejaVu Sans, Liberation Sans ou Noto Sans antes de gerar as figuras."
    )


def centered(
    draw: ImageDraw.ImageDraw, x: float, y: float, text: str, used_font, fill: str
) -> None:
    box = draw.textbbox((0, 0), text, font=used_font)
    draw.text(
        (x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=used_font, fill=fill
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Sem linhas para salvar em {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(rows: list[dict[str, str]], arm: str, policy: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["arm_id"] == arm and row["threshold_policy"] == policy]
    ids = [row["canonical_id"] for row in selected]
    if not selected or len(ids) != len(set(ids)):
        raise ValueError(f"Grupo ausente ou duplicado: {arm}/{policy}")
    return selected


def save_metrics_figure(summary: list[dict[str, Any]], path: Path) -> None:
    common = [row for row in summary if (row["arm_id"], row["threshold_policy"]) in COMMON_GROUPS]
    width, height = 2500, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "Avaliação final interna e externa", font=font(48, True), fill="#0f172a")
    draw.text(
        (70, 108),
        "Barras: estimativa. Hastes: IC 95% por bootstrap da unidade canônica.",
        font=font(27),
        fill="#475569",
    )
    panel_width, panel_top, panel_bottom = 1010, 235, 810
    for panel_index, (key, title) in enumerate(
        (("dice_macro", "Dice macro por imagem"), ("dice_micro", "Dice micro (data-based)"))
    ):
        left = 120 + panel_index * 1210
        right = left + panel_width
        centered(draw, (left + right) / 2, 195, title, font(32, True), "#0f172a")
        for tick in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
            y = panel_bottom - (tick - 0.65) / 0.25 * (panel_bottom - panel_top)
            draw.line((left, y, right, y), fill="#e2e8f0", width=2)
            draw.text((left - 72, y - 14), f"{tick:.2f}", font=font(21), fill="#64748b")
        bar_width, gap = 105, 42
        x = left + 90
        for group_index, (arm, policy) in enumerate(COMMON_GROUPS):
            group_start = x
            for dataset in ("DFUC2022 interno", "FUSeg externo"):
                row = next(
                    item
                    for item in common
                    if item["dataset"] == dataset
                    and item["arm_id"] == arm
                    and item["threshold_policy"] == policy
                )
                value = float(row[key])
                low, high = float(row[f"{key}_ci95_low"]), float(row[f"{key}_ci95_high"])
                y = panel_bottom - (value - 0.65) / 0.25 * (panel_bottom - panel_top)
                y_low = panel_bottom - (low - 0.65) / 0.25 * (panel_bottom - panel_top)
                y_high = panel_bottom - (high - 0.65) / 0.25 * (panel_bottom - panel_top)
                draw.rectangle((x, y, x + bar_width, panel_bottom), fill=COLORS[dataset])
                middle = x + bar_width / 2
                draw.line((middle, y_high, middle, y_low), fill="#111827", width=4)
                draw.line((middle - 13, y_high, middle + 13, y_high), fill="#111827", width=4)
                draw.line((middle - 13, y_low, middle + 13, y_low), fill="#111827", width=4)
                centered(draw, middle, y - 24, f"{value:.3f}", font(20, True), "#0f172a")
                x += bar_width + gap
            centered(
                draw,
                group_start + bar_width + gap / 2,
                865,
                LABELS[(arm, policy)]
                .replace(" (threshold congelado)", "\ncongelado")
                .replace(" (0,5; sensibilidade)", "\n0,5")
                .replace("DR-IFT + SAM", "DR-IFT + SAM"),
                font(20, True),
                "#334155",
            )
            x += 70
    draw.rectangle((700, 955, 735, 990), fill=COLORS["DFUC2022 interno"])
    draw.text((750, 955), "DFUC2022 interno", font=font(23), fill="#334155")
    draw.rectangle((1240, 955, 1275, 990), fill=COLORS["FUSeg externo"])
    draw.text((1290, 955), "FUSeg externo", font=font(23), fill="#334155")
    image.save(path)


def save_area_figure(rows: list[dict[str, Any]], path: Path) -> None:
    width, height = 2400, 980
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42), "Desempenho por tamanho relativo da lesão", font=font(46, True), fill="#0f172a"
    )
    draw.text(
        (70, 102),
        "Q1 reúne as menores lesões de cada dataset; Q5, as maiores.",
        font=font(27),
        fill="#475569",
    )
    y_min, y_max = 0.30, 0.90
    for panel_index, dataset in enumerate(("DFUC2022 interno", "FUSeg externo")):
        left, right, top, bottom = 150 + panel_index * 1180, 1060 + panel_index * 1180, 250, 790
        centered(draw, (left + right) / 2, 195, dataset, font(31, True), "#0f172a")
        for tick in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            y = bottom - (tick - y_min) / (y_max - y_min) * (bottom - top)
            draw.line((left, y, right, y), fill="#e2e8f0", width=2)
            if panel_index == 0:
                draw.text((left - 64, y - 13), f"{tick:.1f}", font=font(20), fill="#64748b")
        subset = [row for row in rows if row["dataset"] == dataset]
        for arm, color in (("DRIFT", "#6d28d9"), ("DRIFT_SAM_BPM", "#0891b2")):
            arm_rows = sorted(
                [row for row in subset if row["arm_id"] == arm],
                key=lambda row: int(row["area_quintile"]),
            )
            points = []
            for row in arm_rows:
                quintile, value = int(row["area_quintile"]), float(row["dice_macro"])
                x = left + (quintile - 1) / 4 * (right - left)
                y = bottom - (value - y_min) / (y_max - y_min) * (bottom - top)
                points.append((x, y))
            draw.line(points, fill=color, width=7)
            for x, y in points:
                draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)
        for quintile in range(1, 6):
            x = left + (quintile - 1) / 4 * (right - left)
            centered(draw, x, 835, f"Q{quintile}", font(23, True), "#334155")
    draw.rectangle((720, 900, 755, 935), fill="#6d28d9")
    draw.text((770, 900), "DR-IFT", font=font(23), fill="#334155")
    draw.rectangle((1160, 900, 1195, 935), fill="#0891b2")
    draw.text((1210, 900), "DR-IFT + SAM", font=font(23), fill="#334155")
    image.save(path)


def save_error_figure(rows: list[dict[str, Any]], path: Path) -> None:
    categories = [
        "predicao_vazia",
        "sem_sobreposicao",
        "baixo_dice_recall_limitado",
        "baixo_dice_precisao_limitada",
        "baixo_dice_misto",
        "sobreposicao_parcial",
        "alta_sobreposicao",
    ]
    labels = [
        "Vazia",
        "Sem\nsobreposição",
        "Recall\nlimitado",
        "Precisão\nlimitada",
        "Misto",
        "Parcial",
        "Alta",
    ]
    width, height = 2700, 1120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "Taxonomia descritiva dos resultados", font=font(46, True), fill="#0f172a")
    draw.text(
        (70, 103),
        "Categorias calculadas por unidade; não representam adjudicação clínica.",
        font=font(27),
        fill="#475569",
    )
    for panel_index, dataset in enumerate(("DFUC2022 interno", "FUSeg externo")):
        left, right, top, bottom = 150 + panel_index * 1330, 1270 + panel_index * 1330, 250, 850
        centered(draw, (left + right) / 2, 195, dataset, font(31, True), "#0f172a")
        for tick in (0, 20, 40, 60, 80, 100):
            y = bottom - tick / 100 * (bottom - top)
            draw.line((left, y, right, y), fill="#e2e8f0", width=2)
            if panel_index == 0:
                draw.text((left - 67, y - 13), f"{tick}%", font=font(20), fill="#64748b")
        subset = [row for row in rows if row["dataset"] == dataset]
        group_width = (right - left) / len(categories)
        for index, category in enumerate(categories):
            center_x = left + (index + 0.5) * group_width
            for arm, offset, color in (("DRIFT", -30, "#6d28d9"), ("DRIFT_SAM_BPM", 30, "#0891b2")):
                by_category = {
                    row["error_category"]: float(row["percentage"])
                    for row in subset
                    if row["arm_id"] == arm
                }
                value = by_category.get(category, 0.0)
                y = bottom - value / 100 * (bottom - top)
                draw.rectangle(
                    (center_x + offset - 27, y, center_x + offset + 27, bottom), fill=color
                )
            for line_index, line in enumerate(labels[index].split("\n")):
                centered(draw, center_x, 895 + line_index * 27, line, font(18, True), "#334155")
    draw.rectangle((920, 1030, 955, 1065), fill="#6d28d9")
    draw.text((970, 1030), "DR-IFT", font=font(23), fill="#334155")
    draw.rectangle((1370, 1030, 1405, 1065), fill="#0891b2")
    draw.text((1420, 1030), "DR-IFT + SAM", font=font(23), fill="#334155")
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-results", type=Path, required=True)
    parser.add_argument("--external-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    internal_dir = args.internal_results.resolve()
    external_dir = args.external_results.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)

    internal = read_csv(internal_dir / "test_per_image_all_arms.csv")
    external = read_csv(external_dir / "external_per_unit.csv")
    datasets = {"DFUC2022 interno": internal, "FUSeg externo": external}

    summary_rows: list[dict[str, Any]] = []
    grouped_rows: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    group_index = 0
    for dataset, source in datasets.items():
        groups = sorted({(row["arm_id"], row["threshold_policy"]) for row in source})
        for arm, policy in groups:
            group = select(source, arm, policy)
            grouped_rows[(dataset, arm, policy)] = group
            summary = summarize_rows(
                group,
                samples=args.bootstrap_samples,
                seed=args.seed + group_index * 100,
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "arm_id": arm,
                    "threshold_policy": policy,
                    **summary,
                }
            )
            group_index += 1
    write_csv(output / "dataset_summary_macro_micro.csv", summary_rows)

    cross_rows: list[dict[str, Any]] = []
    for index, (arm, policy) in enumerate(COMMON_GROUPS):
        internal_group = grouped_rows[("DFUC2022 interno", arm, policy)]
        external_group = grouped_rows[("FUSeg externo", arm, policy)]
        internal_macro = sum(float(row["dice"]) for row in internal_group) / len(internal_group)
        external_macro = sum(float(row["dice"]) for row in external_group) / len(external_group)
        low, high = independent_macro_gap_ci(
            internal_group,
            external_group,
            samples=args.bootstrap_samples,
            seed=args.seed + 50_000 + index,
        )
        cross_rows.append(
            {
                "arm_id": arm,
                "threshold_policy": policy,
                "internal_dice_macro": internal_macro,
                "external_dice_macro": external_macro,
                "external_minus_internal_macro": external_macro - internal_macro,
                "gap_ci95_low": low,
                "gap_ci95_high": high,
                "interpretation": "descriptive_domain_gap_not_paired",
            }
        )
    write_csv(output / "cross_dataset_macro_comparison.csv", cross_rows)

    area_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    for dataset, source in datasets.items():
        quintiles = relative_area_quintiles(source)
        for arm, policy in (("DRIFT", "selected"), ("DRIFT_SAM_BPM", "selected")):
            group = grouped_rows[(dataset, arm, policy)]
            for quintile in range(1, 6):
                stratum = [row for row in group if quintiles[row["canonical_id"]] == quintile]
                values = [float(row["dice"]) for row in stratum]
                area_rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm,
                        "threshold_policy": policy,
                        "area_quintile": quintile,
                        "units": len(stratum),
                        "foreground_fraction_min": min(
                            float(row["foreground_fraction"]) for row in stratum
                        ),
                        "foreground_fraction_max": max(
                            float(row["foreground_fraction"]) for row in stratum
                        ),
                        "dice_macro": sum(values) / len(values),
                        "dice_median": statistics.median(values),
                        "zero_dice_units": sum(value == 0.0 for value in values),
                    }
                )
            counts = Counter(error_category(row) for row in group)
            for category, count in sorted(counts.items()):
                taxonomy_rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm,
                        "threshold_policy": policy,
                        "error_category": category,
                        "units": count,
                        "percentage": 100.0 * count / len(group),
                    }
                )
            ordered = sorted(group, key=lambda row: (float(row["dice"]), row["canonical_id"]))
            chosen = [("lowest", row) for row in ordered[:12]] + [
                ("highest", row) for row in reversed(ordered[-12:])
            ]
            for rank, (extreme, row) in enumerate(chosen, start=1):
                extreme_rows.append(
                    {
                        "dataset": dataset,
                        "arm_id": arm,
                        "threshold_policy": policy,
                        "extreme": extreme,
                        "rank_within_extreme": rank if rank <= 12 else rank - 12,
                        "canonical_id": row["canonical_id"],
                        "foreground_fraction": row["foreground_fraction"],
                        "dice": row["dice"],
                        "iou": row["iou"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "boundary_iou": row["boundary_iou"],
                        "error_category": error_category(row),
                    }
                )
    write_csv(output / "dice_by_relative_area_quintile.csv", area_rows)
    write_csv(output / "error_taxonomy.csv", taxonomy_rows)
    write_csv(output / "extreme_cases_for_review.csv", extreme_rows)

    save_metrics_figure(summary_rows, figures / "macro_micro_comparison.png")
    save_area_figure(area_rows, figures / "dice_by_area_quintile.png")
    save_error_figure(taxonomy_rows, figures / "error_taxonomy.png")

    report = {
        "status": "completed",
        "analysis": "day16_statistical_consolidation",
        "internal_run_id": internal_dir.name,
        "external_run_id": external_dir.name,
        "bootstrap": {
            "method": "unit_level_percentile",
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "aggregation": {
            "standardized_cross_dataset": "macro_per_canonical_unit",
            "DFUC2022_native": "macro_per_image",
            "FUSeg_native": "micro_data_based_over_canonical_units",
            "external_multiple_references": "mean_counts_and_metrics_within_unit_before_aggregation",
        },
        "area_analysis": "within_dataset_rank_quintiles_independent_of_performance",
        "error_taxonomy": {
            "type": "descriptive_metric_based_not_clinical_adjudication",
            "high_overlap": "dice >= 0.75",
            "partial_overlap": "0.50 <= dice < 0.75",
            "low_dice_tendency": "dice < 0.50; precision versus recall identifies the limiting metric",
        },
        "scientific_guards": {
            "training_performed": False,
            "inference_performed": False,
            "threshold_optimization_performed": False,
            "model_selection_performed": False,
            "samples_removed_post_hoc": False,
        },
        "artifacts": {
            "summary": "dataset_summary_macro_micro.csv",
            "cross_dataset": "cross_dataset_macro_comparison.csv",
            "area": "dice_by_relative_area_quintile.csv",
            "taxonomy": "error_taxonomy.csv",
            "extremes": "extreme_cases_for_review.csv",
            "figures": [
                "figures/macro_micro_comparison.png",
                "figures/dice_by_area_quintile.png",
                "figures/error_taxonomy.png",
            ],
        },
    }
    (output / "day16_analysis_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
