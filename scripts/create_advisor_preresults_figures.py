"""Gera figuras sintéticas para o resumo de pré-resultados ao orientador.

Usa somente Pillow para também funcionar no ambiente local sem Matplotlib. Os
valores são resultados de validação da seed 42 já auditados. O script não acessa
o teste interno nem o FUSeg.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def centered(draw, xy, text, fnt, fill="#0f172a"):
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill
    )


def save_progression() -> None:
    labels = ["B0", "D0", "P1", "P2", "A1", "A2", "P1+A1", "DR-RT", "DR-IT", "DR-IF", "DR-IFT"]
    values = [
        0.671759,
        0.645122,
        0.652252,
        0.634891,
        0.679915,
        0.657453,
        0.687214,
        0.701366,
        0.714372,
        0.690164,
        0.739212,
    ]
    colors = [
        "#94a3b8",
        "#2563eb",
        "#60a5fa",
        "#60a5fa",
        "#16a34a",
        "#86efac",
        "#22c55e",
        "#f59e0b",
        "#f97316",
        "#c084fc",
        "#7c3aed",
    ]
    w, h = 2100, 980
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text(
        (75, 42),
        "Progressão dos experimentos — resultados preliminares (seed 42)",
        font=font(42, True),
        fill="#0f172a",
    )
    left, top, right, bottom = 125, 145, 2030, 790
    lo, hi = 0.60, 0.76
    for tick in [0.60, 0.64, 0.68, 0.72, 0.76]:
        y = bottom - (tick - lo) / (hi - lo) * (bottom - top)
        d.line((left, y, right, y), fill="#dbe3ec", width=2)
        d.text((40, y - 15), f"{tick:.2f}", font=font(25), fill="#334155")
    d.line((left, top, left, bottom), fill="#334155", width=3)
    d.line((left, bottom, right, bottom), fill="#334155", width=3)
    step = (right - left) / len(labels)
    barw = step * 0.68
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = left + step * (i + 0.5)
        y = bottom - (value - lo) / (hi - lo) * (bottom - top)
        d.rounded_rectangle((cx - barw / 2, y, cx + barw / 2, bottom), radius=8, fill=color)
        centered(d, (cx, y - 25), f"{value:.3f}", font(23, True), color)
        centered(d, (cx, bottom + 33), label, font(25, True))
    d.text(
        (125, 875),
        "Leitura correta: as barras mostram desempenho observado. Comparações entre famílias são descritivas;",
        font=font(24),
        fill="#334155",
    )
    d.text(
        (125, 910),
        "afirmações de efeito usam apenas os contrastes controlados e pareados da figura seguinte.",
        font=font(24),
        fill="#334155",
    )
    im.save(OUTPUT_DIR / "preresults_experiment_progression.png")


def save_effects() -> None:
    rows = [
        ("P1 − D0", 0.007130, -0.007882, 0.023141),
        ("P2 − D0", -0.010230, -0.028440, 0.008378),
        ("A1 − D0", 0.034793, 0.015983, 0.053900),
        ("A2 − D0", 0.012331, -0.004655, 0.029588),
        ("P1+A1 − A1", 0.007299, -0.006569, 0.021336),
        ("DR-IT − DR-RT", 0.013006, -0.001247, 0.027525),
        ("DR-IFT − DR-IF", 0.049049, 0.030837, 0.067989),
    ]
    w, h = 2100, 1050
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text(
        (70, 40),
        "Efeitos dos contrastes controlados na validação — seed 42",
        font=font(42, True),
        fill="#0f172a",
    )
    left, right, top, bottom = 620, 1820, 160, 860
    xmin, xmax = -0.035, 0.075

    def sx(x):
        return left + (x - xmin) / (xmax - xmin) * (right - left)

    for tick in [-0.03, 0.00, 0.03, 0.06]:
        x = sx(tick)
        d.line((x, top, x, bottom), fill="#dbe3ec", width=2)
        centered(d, (x, bottom + 35), f"{tick:+.2f}", font(24), "#334155")
    d.line((sx(0), top, sx(0), bottom), fill="#0f172a", width=4)
    gap = (bottom - top) / len(rows)
    for i, (label, effect, low, high) in enumerate(rows):
        y = top + gap * (i + 0.5)
        color = "#15803d" if low > 0 else "#64748b"
        d.text((80, y - 18), label, font=font(28, True), fill="#0f172a")
        d.line((sx(low), y, sx(high), y), fill=color, width=6)
        d.line((sx(low), y - 12, sx(low), y + 12), fill=color, width=4)
        d.line((sx(high), y - 12, sx(high), y + 12), fill=color, width=4)
        d.ellipse((sx(effect) - 10, y - 10, sx(effect) + 10, y + 10), fill=color)
        d.text((1840, y - 16), f"{effect:+.3f}", font=font(25, True), fill=color)
    centered(
        d,
        ((left + right) / 2, 925),
        "Diferença no Dice macro por imagem (IC 95% bootstrap)",
        font(27, True),
    )
    d.text(
        (80, 990),
        "Verde: intervalo inteiramente acima de zero. Cinza: resultado compatível com ausência de ganho.",
        font=font(23),
        fill="#334155",
    )
    im.save(OUTPUT_DIR / "preresults_controlled_effects.png")


def save_workflow() -> None:
    stages = [
        ("Auditoria e\ncanonização", "done"),
        ("Split congelado\n70/15/15", "done"),
        ("DeepResUNet\ncentral (D0)", "done"),
        ("Pré-processamento\nP0/P1/P2", "done"),
        ("Augmentation\nA1 selecionado", "done"),
        ("Transfer learning\nDR-RT/DR-IT", "done"),
        ("Fine-tuning\nDR-IF/DR-IFT", "done"),
        ("Confirmação\nmultiseed", "done"),
        ("Refinamento SAM\nsem máscara GT", "future"),
        ("Teste interno\nabertura única", "future"),
        ("FUSeg\nvalidação externa", "future"),
    ]
    fill = {"done": "#dcfce7", "current": "#fef3c7", "future": "#e2e8f0"}
    edge = {"done": "#15803d", "current": "#b45309", "future": "#64748b"}
    w, h = 2200, 1050
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((70, 38), "Desenho experimental do TCC", font=font(44, True), fill="#0f172a")
    d.text(
        (70, 98),
        "Escolhas em treino/validação; teste interno e FUSeg permanecem bloqueados.",
        font=font(27),
        fill="#334155",
    )
    positions = [(70 + i * 415, 235) for i in range(5)]
    positions += [(1730 - i * 415, 555) for i in range(5)]
    positions += [(70, 835)]
    bw, bh = 335, 145
    for i, ((label, status), (x, y)) in enumerate(zip(stages, positions)):
        d.rounded_rectangle(
            (x, y, x + bw, y + bh), radius=22, fill=fill[status], outline=edge[status], width=5
        )
        parts = label.split("\n")
        centered(d, (x + bw / 2, y + 51), parts[0], font(25, True))
        centered(d, (x + bw / 2, y + 96), parts[1], font(24))
        if i < len(stages) - 1:
            nx, ny = positions[i + 1]
            if y == ny and nx > x:
                d.line((x + bw + 10, y + bh / 2, nx - 12, ny + bh / 2), fill="#475569", width=5)
                d.polygon(
                    [
                        (nx - 12, ny + bh / 2),
                        (nx - 30, ny + bh / 2 - 12),
                        (nx - 30, ny + bh / 2 + 12),
                    ],
                    fill="#475569",
                )
            elif y == ny:
                d.line((x - 10, y + bh / 2, nx + bw + 12, ny + bh / 2), fill="#475569", width=5)
                d.polygon(
                    [
                        (nx + bw + 12, ny + bh / 2),
                        (nx + bw + 30, ny + bh / 2 - 12),
                        (nx + bw + 30, ny + bh / 2 + 12),
                    ],
                    fill="#475569",
                )
            else:
                d.line((x + bw / 2, y + bh + 8, nx + bw / 2, ny - 12), fill="#475569", width=5)
                d.polygon(
                    [
                        (nx + bw / 2, ny - 12),
                        (nx + bw / 2 - 12, ny - 32),
                        (nx + bw / 2 + 12, ny - 32),
                    ],
                    fill="#475569",
                )
    for x, status, label in [(1630, "done", "concluído"), (1940, "future", "futuro")]:
        d.ellipse((x, 945, x + 25, 970), fill=fill[status], outline=edge[status], width=3)
        d.text((x + 37, 943), label, font=font(22), fill="#334155")
    im.save(OUTPUT_DIR / "preresults_workflow.png")


def main() -> None:
    save_progression()
    save_effects()
    save_workflow()
    print(f"Figuras salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
