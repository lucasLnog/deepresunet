"""Cria tabelas e figuras da confirmação multiseed DR-IFT versus DR-IF.

Lê somente os CSVs/JSONs derivados da validação. Não acessa teste ou FUSeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selected-results", type=Path, required=True)
    p.add_argument(
        "--comparison", nargs=2, action="append", metavar=("SEED", "JSON"), required=True
    )
    p.add_argument(
        "--area", nargs=3, action="append", metavar=("SEED", "ARM", "CSV"), required=True
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def centered(draw, x, y, text, fnt, fill="#0f172a"):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_dice_plot(rows, output):
    w, h = 1600, 940
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text(
        (70, 40), "Confirmação multiseed: Dice por política", font=font(40, True), fill="#0f172a"
    )
    d.text(
        (70, 95),
        "Validação congelada, threshold selecionado dentro de cada execução",
        font=font(25),
        fill="#475569",
    )
    left, right, top, bottom = 145, 1510, 180, 760
    lo, hi = 0.67, 0.75
    for tick in (0.67, 0.69, 0.71, 0.73, 0.75):
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        d.line((left, y, right, y), fill="#dbe3ec", width=2)
        d.text((65, y - 14), f"{tick:.2f}", font=font(22), fill="#475569")
    step = (right - left) / len(rows)
    for i, row in enumerate(rows):
        cx = left + step * (i + 0.5)
        x1, x2 = cx - 85, cx + 85
        a, b = float(row["drif_dice"]), float(row["drift_dice"])
        y1 = bottom - (a - lo) / (hi - lo) * (bottom - top)
        y2 = bottom - (b - lo) / (hi - lo) * (bottom - top)
        d.line((x1, y1, x2, y2), fill="#64748b", width=5)
        d.ellipse((x1 - 12, y1 - 12, x1 + 12, y1 + 12), fill="#8b5cf6")
        d.ellipse((x2 - 12, y2 - 12, x2 + 12, y2 + 12), fill="#15803d")
        centered(d, x1, y1 + 34, f"{a:.3f}", font(21, True), "#7c3aed")
        centered(d, x2, y2 - 30, f"{b:.3f}", font(21, True), "#15803d")
        centered(d, cx, bottom + 40, f"seed {row['seed']}", font(25, True))
        centered(d, cx, bottom + 78, f"Δ {float(row['difference_dice']):+.3f}", font(22), "#15803d")
    d.rectangle((520, 855, 545, 880), fill="#8b5cf6")
    d.text((558, 852), "DR-IF", font=font(22), fill="#334155")
    d.rectangle((745, 855, 770, 880), fill="#15803d")
    d.text((783, 852), "DR-IFT", font=font(22), fill="#334155")
    d.text(
        (70, 900),
        "Cada linha liga execuções pareadas na mesma seed; todas apontam para ganho do fine-tuning progressivo.",
        font=font(21),
        fill="#475569",
    )
    im.save(output)


def effects_plot(comparisons, output):
    w, h = 1700, 900
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text(
        (70, 40),
        "Efeito pareado do fine-tuning progressivo por seed",
        font=font(40, True),
        fill="#0f172a",
    )
    d.text(
        (70, 95),
        "DR-IFT menos DR-IF no Dice macro por imagem; barras são IC 95% bootstrap",
        font=font(25),
        fill="#475569",
    )
    left, right, top, bottom = 430, 1500, 190, 690
    xmin, xmax = 0.0, 0.075

    def sx(v):
        return left + (v - xmin) / (xmax - xmin) * (right - left)

    for tick in (0.00, 0.02, 0.04, 0.06):
        x = sx(tick)
        d.line((x, top, x, bottom), fill="#dbe3ec", width=2)
        centered(d, x, bottom + 35, f"{tick:+.2f}", font(22), "#475569")
    d.line((sx(0), top, sx(0), bottom), fill="#0f172a", width=4)
    gap = (bottom - top) / len(comparisons)
    for i, row in enumerate(comparisons):
        y = top + gap * (i + 0.5)
        effect, low, high = row["effect"], row["low"], row["high"]
        d.text((90, y - 17), f"seed {row['seed']}", font=font(28, True), fill="#0f172a")
        d.line((sx(low), y, sx(high), y), fill="#15803d", width=7)
        d.line((sx(low), y - 13, sx(low), y + 13), fill="#15803d", width=4)
        d.line((sx(high), y - 13, sx(high), y + 13), fill="#15803d", width=4)
        d.ellipse((sx(effect) - 11, y - 11, sx(effect) + 11, y + 11), fill="#15803d")
        d.text((1520, y - 15), f"{effect:+.3f}", font=font(24, True), fill="#15803d")
    effects = [row["effect"] for row in comparisons]
    mean, sd = statistics.fmean(effects), statistics.stdev(effects)
    d.text(
        (90, 775),
        f"Média descritiva entre seeds: {mean:+.4f}   |   DP da diferença: {sd:.4f}",
        font=font(26, True),
        fill="#0f172a",
    )
    d.text(
        (90, 825),
        "Com apenas três seeds, média e DP descrevem estabilidade; não foi aplicado teste assintótico entre treinamentos.",
        font=font(22),
        fill="#475569",
    )
    im.save(output)


def area_summary(area_args, output):
    grouped = {}
    for raw_seed, arm, raw_path in area_args:
        for row in read_csv(Path(raw_path)):
            grouped.setdefault((int(row["foreground_area_bin"]), arm), []).append(
                float(row["dice_mean"])
            )
    result = []
    for area_bin in range(5):
        drif = grouped[(area_bin, "DR-IF")]
        drift = grouped[(area_bin, "DR-IFT")]
        result.append(
            {
                "foreground_area_bin": area_bin,
                "DR-IF_mean_across_seeds": statistics.fmean(drif),
                "DR-IFT_mean_across_seeds": statistics.fmean(drift),
                "difference_mean_across_seeds": statistics.fmean(
                    b - a for a, b in zip(drif, drift)
                ),
                "seeds_favoring_DR-IFT": sum(b > a for a, b in zip(drif, drift)),
            }
        )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.selected_results)
    comparisons = []
    for raw_seed, raw_path in args.comparison:
        summary = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        dice = summary["results"]["DR-IFT_minus_DR-IF"]["metrics"]["dice"]
        comparisons.append(
            {
                "seed": int(raw_seed),
                "effect": dice["mean_difference"],
                "low": dice["ci95_low"],
                "high": dice["ci95_high"],
            }
        )
    comparisons.sort(key=lambda row: row["seed"])
    paired_dice_plot(rows, args.output_dir / "day11_dice_by_seed.png")
    effects_plot(comparisons, args.output_dir / "day11_effects_by_seed.png")
    area_summary(args.area, args.output_dir / "day11_area_bin_multiseed.csv")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
