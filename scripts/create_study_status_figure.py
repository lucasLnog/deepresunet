#!/usr/bin/env python3
"""Gera uma figura reproduzível do estado atual dos baselines na validação."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = {
    "B0 — U-Net/ResNet34": "B0_unet_resnet34_random_seed42_20260828T161032Z",
    "D0 — DeepResUNet": "BH_deep_res_unet_historical_seed42_20260828T183513Z",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_pillow_fallback(rows: list[dict], figure_path: Path) -> None:
    """Fallback sem Matplotlib para ambientes locais com NumPy incompatível."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1600, 760), "white")
    draw = ImageDraw.Draw(canvas)
    windows_font = Path("C:/Windows/Fonts/segoeui.ttf")
    if windows_font.exists():
        font = ImageFont.truetype(str(windows_font), size=25)
        small = ImageFont.truetype(str(windows_font), size=19)
    else:
        font = ImageFont.load_default(size=22)
        small = ImageFont.load_default(size=17)
    draw.text(
        (55, 30),
        "Estado atual — desenvolvimento DFUC (1 seed; teste congelado não acessado)",
        fill="#172554",
        font=font,
    )
    colors = ("#2171b5", "#d95f0e")
    metrics = (
        ("Dice em 0,5", "dice_at_0_5"),
        ("Dice selecionado", "dice_selected"),
        ("IoU selecionado", "iou_selected"),
        ("Precisão*", "precision_selected_original_convention"),
        ("Recall", "recall_selected"),
    )
    left, top, plot_width, plot_height = 110, 130, 1370, 470
    draw.line((left, top, left, top + plot_height), fill="#475569", width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill="#475569",
        width=2,
    )
    for tick in range(0, 9):
        value = tick / 10
        y = top + plot_height - int(value * plot_height)
        draw.line((left - 8, y, left, y), fill="#475569", width=2)
        draw.text((52, y - 10), f"{value:.1f}", fill="#475569", font=small)
    group_width = plot_width / len(metrics)
    bar_width = 82
    for metric_index, (metric_label, key) in enumerate(metrics):
        center = left + group_width * (metric_index + 0.5)
        for model_index, row in enumerate(rows):
            value = float(row[key])
            x0 = int(center + (model_index - 1) * bar_width)
            x1 = x0 + bar_width
            y0 = top + plot_height - int(value * plot_height)
            draw.rectangle((x0, y0, x1, top + plot_height), fill=colors[model_index])
            draw.text(
                (x0 + 4, y0 - 24),
                f"{value:.3f}",
                fill=colors[model_index],
                font=small,
            )
        draw.text(
            (int(center - 80), top + plot_height + 18),
            metric_label,
            fill="#172554",
            font=small,
        )
    for index, row in enumerate(rows):
        draw.rectangle((150 + index * 520, 665, 185 + index * 520, 700), fill=colors[index])
        draw.text(
            (200 + index * 520, 668),
            f"{row['model']}  (t={row['threshold_selected']:.2f})",
            fill="#172554",
            font=small,
        )
    draw.text(
        (280, 725),
        "Comparação descritiva. *Precisão original será recalculada com a convenção corrigida.",
        fill="#7f1d1d",
        font=small,
    )
    canvas.save(figure_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    base = root / "results" / "development" / "baselines"
    output = root / "output" / "auditoria_metodologica"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, run_id in RUNS.items():
        selection = load_json(base / run_id / "threshold_selection.json")
        standard = selection["metrics_at_0.5"]
        selected = selection["metrics_at_selected_threshold"]
        rows.append(
            {
                "model": label,
                "run_id": run_id,
                "threshold_selected": selection["selected_threshold"],
                "dice_at_0_5": standard["dice"]["mean"],
                "dice_selected": selected["dice"]["mean"],
                "iou_selected": selected["iou"]["mean"],
                "precision_selected_original_convention": selected["precision"]["mean"],
                "recall_selected": selected["recall"]["mean"],
                "dice_std_selected": selected["dice"]["std"],
            }
        )

    csv_path = output / "status_baselines_validacao.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = output / "status_baselines_validacao.png"
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except (ImportError, AttributeError):
        create_pillow_fallback(rows, figure_path)
        print(figure_path)
        print(csv_path)
        return 0

    labels = [row["model"] for row in rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    width = 0.34
    axes[0].bar(
        x - width / 2,
        [row["dice_at_0_5"] for row in rows],
        width,
        label="Dice em 0,5",
        color="#6baed6",
    )
    axes[0].bar(
        x + width / 2,
        [row["dice_selected"] for row in rows],
        width,
        label="Dice no threshold selecionado",
        color="#2171b5",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.58, 0.70)
    axes[0].set_ylabel("Dice macro por imagem")
    axes[0].set_title("Desempenho atual na validação")
    axes[0].legend(loc="lower left")
    for index, row in enumerate(rows):
        axes[0].text(
            index + width / 2,
            row["dice_selected"] + 0.002,
            f"t={row['threshold_selected']:.2f}",
            ha="center",
            fontsize=9,
        )

    metric_names = ("Dice", "IoU", "Precisão*", "Recall")
    metric_keys = (
        "dice_selected",
        "iou_selected",
        "precision_selected_original_convention",
        "recall_selected",
    )
    metric_x = np.arange(len(metric_names))
    for index, row in enumerate(rows):
        axes[1].bar(
            metric_x + (index - 0.5) * width,
            [row[key] for key in metric_keys],
            width,
            label=row["model"],
        )
    axes[1].set_xticks(metric_x, metric_names)
    axes[1].set_ylim(0.45, 0.78)
    axes[1].set_ylabel("Média por imagem")
    axes[1].set_title("Perfil no threshold selecionado")
    axes[1].legend(loc="lower right")

    figure.suptitle(
        "Estado atual — desenvolvimento DFUC (1 seed; teste congelado não acessado)",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Comparação de arquiteturas é descritiva. *Precisão usa a convenção original e será recalculada.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(figure_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
