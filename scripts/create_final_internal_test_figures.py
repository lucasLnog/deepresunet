#!/usr/bin/env python3
"""Gera figuras reproduziveis do teste interno final congelado.

O script apenas le os CSVs produzidos pelo runner final. Ele nao recalcula
thresholds, nao escolhe modelos e nao acessa imagens ou mascaras do dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ARM_ORDER = [
    "D0",
    "P1",
    "A1",
    "P1A1",
    "DRRT",
    "DRIT",
    "DRIF",
    "DRIFT",
    "DRIFT_SAM_BPM",
    "DRIFTB",
]

ARM_LABELS = {
    "D0": "D0",
    "P1": "P1",
    "A1": "A1",
    "P1A1": "P1+A1",
    "DRRT": "DR-RT",
    "DRIT": "DR-IT",
    "DRIF": "DR-IF",
    "DRIFT": "DR-IFT (principal)",
    "DRIFT_SAM_BPM": "DR-IFT + SAM",
    "DRIFTB": "DR-IFT-B (exploratorio)",
}

ARM_COLORS = {
    "D0": "#64748b",
    "P1": "#60a5fa",
    "A1": "#22c55e",
    "P1A1": "#16a34a",
    "DRRT": "#f59e0b",
    "DRIT": "#f97316",
    "DRIF": "#a78bfa",
    "DRIFT": "#6d28d9",
    "DRIFT_SAM_BPM": "#0891b2",
    "DRIFTB": "#b45309",
}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def centered(draw: ImageDraw.ImageDraw, x: float, y: float, text: str, fnt, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def save_arm_performance(summary: list[dict[str, str]], output: Path) -> None:
    selected = {row["arm_id"]: row for row in summary if row["threshold_policy"] == "selected"}
    metrics = [
        ("Dice macro", "dice_mean", 0.50, 0.80),
        ("IoU macro", "iou_mean", 0.45, 0.70),
        ("Boundary IoU", "boundary_iou_mean", 0.40, 0.63),
    ]
    width, height = 2380, 1320
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "Teste interno final do DFUC2022", font=font(48, True), fill="#0f172a")
    draw.text(
        (70, 108),
        "Medias macro por imagem no threshold congelado da validacao (n=294)",
        font=font(28),
        fill="#334155",
    )

    panel_lefts = [430, 1115, 1800]
    panel_width = 500
    top, row_gap = 235, 88
    for panel_left, (title, key, low, high) in zip(panel_lefts, metrics):
        centered(draw, panel_left + panel_width / 2, 190, title, font(30, True), "#0f172a")
        for tick_index in range(4):
            value = low + (high - low) * tick_index / 3
            x = panel_left + panel_width * (value - low) / (high - low)
            draw.line(
                (x, top - 18, x, top + row_gap * len(ARM_ORDER) - 12), fill="#e2e8f0", width=2
            )
            centered(
                draw, x, top + row_gap * len(ARM_ORDER) + 22, f"{value:.2f}", font(21), "#475569"
            )

        for index, arm in enumerate(ARM_ORDER):
            value = float(selected[arm][key])
            y = top + index * row_gap
            x0 = panel_left + panel_width * (max(value, low) - low) / (high - low)
            draw.line((panel_left, y, panel_left + panel_width, y), fill="#f1f5f9", width=24)
            draw.line((panel_left, y, x0, y), fill=ARM_COLORS[arm], width=24)
            draw.ellipse((x0 - 10, y - 10, x0 + 10, y + 10), fill=ARM_COLORS[arm])
            draw.text((x0 + 16, y - 15), f"{value:.3f}", font=font(22, True), fill=ARM_COLORS[arm])

    for index, arm in enumerate(ARM_ORDER):
        y = top + index * row_gap
        label_font = font(25, arm == "DRIFT")
        draw.text((60, y - 27), ARM_LABELS[arm], font=label_font, fill="#0f172a")
        zeros = int(selected[arm]["zero_dice_images"])
        draw.text((60, y + 8), f"Dice zero: {zeros}", font=font(17), fill="#64748b")

    draw.text(
        (70, 1195),
        "DR-IFT e o modelo principal pre-definido. DR-IFT-B permanece exploratorio, mesmo com o maior Dice bruto.",
        font=font(25),
        fill="#334155",
    )
    draw.text(
        (70, 1240),
        "As conclusoes sobre efeito usam os contrastes pareados e intervalos de confianca da figura seguinte.",
        font=font(25),
        fill="#334155",
    )
    image.save(output)


def save_effects(comparisons: list[dict[str, str]], output: Path) -> None:
    rows = [
        row
        for row in comparisons
        if row["threshold_policy"] == "selected" and row["metric"] == "dice"
    ]
    order = [
        "preprocessing",
        "augmentation",
        "transfer_initialization",
        "progressive_fine_tuning",
        "SAM_refinement",
        "preprocessing_augmentation_combination",
        "total_pipeline_change_descriptive",
        "boundary_loss",
    ]
    by_id = {row["contrast_id"]: row for row in rows}
    labels = {
        "preprocessing": "P1 - D0",
        "augmentation": "A1 - D0",
        "transfer_initialization": "DR-IT - DR-RT",
        "progressive_fine_tuning": "DR-IFT - DR-IF",
        "SAM_refinement": "DR-IFT+SAM - DR-IFT",
        "preprocessing_augmentation_combination": "P1+A1 - A1",
        "total_pipeline_change_descriptive": "DR-IFT - D0",
        "boundary_loss": "DR-IFT-B - DR-IFT",
    }

    width, height = 2200, 1310
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 42),
        "Diferencas pareadas no Dice do teste interno",
        font=font(46, True),
        fill="#0f172a",
    )
    draw.text(
        (70, 102),
        "Ponto: diferenca media. Linha: IC 95% bootstrap pareado. Unidade estatistica: imagem canonica.",
        font=font(27),
        fill="#334155",
    )

    left, right, top, bottom = 690, 1750, 230, 1080
    x_min, x_max = -0.025, 0.155

    def scale_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    for tick in [-0.02, 0.00, 0.04, 0.08, 0.12, 0.15]:
        x = scale_x(tick)
        draw.line((x, top - 20, x, bottom), fill="#e2e8f0", width=2)
        centered(draw, x, bottom + 34, f"{tick:+.2f}", font(22), "#475569")
    draw.line((scale_x(0), top - 20, scale_x(0), bottom), fill="#0f172a", width=4)

    row_gap = (bottom - top) / len(order)
    for index, contrast_id in enumerate(order):
        row = by_id[contrast_id]
        family = row["family"]
        effect = float(row["mean_difference"])
        ci_low = float(row["ci95_low"])
        ci_high = float(row["ci95_high"])
        confirmed = (
            family == "primary"
            and row["permutation_p_holm"]
            and float(row["permutation_p_holm"]) < 0.05
        )
        color = "#15803d" if confirmed else ("#b45309" if family == "exploratory" else "#64748b")
        y = top + (index + 0.5) * row_gap
        draw.text((72, y - 20), labels[contrast_id], font=font(27, True), fill="#0f172a")
        family_label = {
            "primary": "primario",
            "secondary": "secundario",
            "exploratory": "exploratorio",
        }[family]
        draw.text((390, y - 15), family_label, font=font(20), fill=color)
        draw.line((scale_x(ci_low), y, scale_x(ci_high), y), fill=color, width=7)
        draw.line((scale_x(ci_low), y - 13, scale_x(ci_low), y + 13), fill=color, width=4)
        draw.line((scale_x(ci_high), y - 13, scale_x(ci_high), y + 13), fill=color, width=4)
        draw.ellipse((scale_x(effect) - 11, y - 11, scale_x(effect) + 11, y + 11), fill=color)
        draw.text((1790, y - 17), f"{effect:+.3f}", font=font(25, True), fill=color)
        if family == "primary":
            p_holm = float(row["permutation_p_holm"])
            p_text = "pHolm<0,001" if p_holm < 0.001 else f"pHolm={p_holm:.3f}".replace(".", ",")
            draw.text((1950, y - 15), p_text, font=font(20), fill="#475569")

    draw.line((55, top + 5 * row_gap, 2145, top + 5 * row_gap), fill="#cbd5e1", width=3)
    draw.line((55, top + 7 * row_gap, 2145, top + 7 * row_gap), fill="#cbd5e1", width=3)
    centered(
        draw,
        (left + right) / 2,
        1155,
        "Diferenca no Dice macro por imagem",
        font(27, True),
        "#0f172a",
    )
    draw.text(
        (70, 1220),
        "Confirmados apos Holm: augmentation e fine-tuning progressivo. O contraste total DR-IFT-D0 e descritivo.",
        font=font(24),
        fill="#334155",
    )
    draw.text(
        (70, 1260),
        "DR-IFT-B nao substitui DR-IFT: o intervalo exploratorio inclui ausencia de ganho.",
        font=font(24),
        fill="#334155",
    )
    image.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_csv(results_dir / "test_summary_all_arms.csv")
    comparisons = load_csv(results_dir / "paired_comparisons.csv")
    save_arm_performance(summary, output_dir / "final_internal_arm_performance.png")
    save_effects(comparisons, output_dir / "final_internal_paired_effects.png")
    print(output_dir / "final_internal_arm_performance.png")
    print(output_dir / "final_internal_paired_effects.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
