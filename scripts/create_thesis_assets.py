from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tcc_segmentation.thesis_reporting import (
    as_float,
    fmt_metric,
    matching_rows,
    read_csv_rows,
    require_one,
    write_csv_rows,
    write_latex_table,
)


ARM_LABELS = {
    "D0": "DeepResUNet aleatória",
    "P1": "D0 + Shades-of-Gray",
    "A1": "D0 + augmentation A1",
    "P1A1": "D0 + P1 + A1",
    "DRRT": "DeepResUNet/ResNet-34 aleatória",
    "DRIT": "DR-IT (ImageNet)",
    "DRIF": "DR-IF (encoder congelado)",
    "DRIFT": "DR-IFT (rede central)",
    "DRIFTB": "DR-IFT-B (exploratória)",
    "DRIFT_SAM_BPM": "DR-IFT + SAM ViT-B",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else ["DejaVuSans.ttf", "arial.ttf"]
    roots = [
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("C:/Windows/Fonts"),
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
    raise RuntimeError("Nenhuma fonte TrueType foi encontrada para gerar a figura.")


def draw_pipeline(path: Path) -> None:
    width, height = 2400, 1160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = find_font(58, bold=True)
    subtitle = find_font(30)
    box_title = find_font(31, bold=True)
    box_body = find_font(25)
    small = find_font(22)
    draw.text((120, 70), "Pipeline experimental do estudo", fill="#18212f", font=title)
    draw.text(
        (120, 150),
        "Seleção restrita à validação; teste interno e FUSeg usados uma única vez após o lacre.",
        fill="#465569",
        font=subtitle,
    )

    boxes = [
        (
            100,
            285,
            540,
            570,
            "1. Curadoria",
            ["DFUC2022: 1.961 unidades", "FUSeg: 962 unidades", "Duplicatas agrupadas por OR"],
        ),
        (
            680,
            285,
            1120,
            570,
            "2. Desenvolvimento",
            ["Treino: 1.373", "Validação: 294", "Teste: 294 (congelado)"],
        ),
        (
            1260,
            285,
            1700,
            570,
            "3. Rede central",
            ["DeepResUNet + ResNet-34", "ImageNet + fine-tuning", "DR-IFT, seed 42, limiar 0,18"],
        ),
        (
            1840,
            285,
            2280,
            570,
            "4. Avaliação final",
            ["Teste interno DFUC", "Externo FUSeg sem ajuste", "Macro + micro + IC 95%"],
        ),
    ]
    colors = ["#e8f1fb", "#eaf7ef", "#fff4dc", "#f1ecfb"]
    outlines = ["#2b6cb0", "#2f855a", "#b7791f", "#6b46c1"]
    for (x0, y0, x1, y1, heading, body), fill, outline in zip(boxes, colors, outlines):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=28, fill=fill, outline=outline, width=5)
        draw.text((x0 + 34, y0 + 30), heading, fill="#18212f", font=box_title)
        y = y0 + 105
        for line in body:
            draw.ellipse((x0 + 36, y + 9, x0 + 48, y + 21), fill=outline)
            draw.text((x0 + 66, y), line, fill="#2d3748", font=box_body)
            y += 58
    for x in [590, 1170, 1750]:
        draw.line((x, 425, x + 40, 425), fill="#667085", width=8)
        draw.polygon([(x + 40, 410), (x + 70, 425), (x + 40, 440)], fill="#667085")

    draw.rounded_rectangle(
        (260, 710, 2140, 1010), radius=30, fill="#f8fafc", outline="#94a3b8", width=4
    )
    draw.text((320, 750), "Comparações planejadas", fill="#18212f", font=box_title)
    planned = [
        "Pré-processamento: P1 vs D0",
        "Augmentation: A1 vs D0",
        "Transfer learning: DR-IT vs DR-RT",
        "Fine-tuning progressivo: DR-IFT vs DR-IF",
        "Refinamento: DR-IFT + SAM vs DR-IFT",
    ]
    positions = [(320, 830), (980, 830), (1580, 830), (620, 925), (1370, 925)]
    for text, position in zip(planned, positions):
        draw.text(position, text, fill="#334155", font=small)
    image.save(path, optimize=True)


def build_tables(project: Path, tables_dir: Path) -> list[Path]:
    internal_dir = project / "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z"
    external_dir = project / "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z"
    analysis_dir = project / "results/day16_analysis/DAY16_FINAL_20260930"
    summary = read_csv_rows(analysis_dir / "dataset_summary_macro_micro.csv")
    contrasts = read_csv_rows(internal_dir / "paired_comparisons.csv")
    gaps = read_csv_rows(analysis_dir / "cross_dataset_macro_comparison.csv")
    external_contrasts = read_csv_rows(external_dir / "external_sam_comparison.csv")

    outputs: list[Path] = []
    datasets = [
        {
            "dataset": "DFUC2022",
            "raw_pairs": 2000,
            "canonical_units": 1961,
            "train": 1373,
            "validation": 294,
            "test_or_external": 294,
            "role": "desenvolvimento e teste interno",
        },
        {
            "dataset": "FUSeg",
            "raw_pairs": 1010,
            "canonical_units": 962,
            "train": "",
            "validation": "",
            "test_or_external": 962,
            "role": "avaliação externa sem adaptação",
        },
    ]
    p = tables_dir / "tabela_01_datasets_splits.csv"
    write_csv_rows(p, datasets, list(datasets[0]))
    outputs.append(p)
    write_latex_table(
        tables_dir / "tabela_01_datasets_splits.tex",
        ["Dataset", "Pares brutos", "Unidades", "Treino", "Validação", "Teste/externo", "Papel"],
        [
            [
                row[key]
                for key in [
                    "dataset",
                    "raw_pairs",
                    "canonical_units",
                    "train",
                    "validation",
                    "test_or_external",
                    "role",
                ]
            ]
            for row in datasets
        ],
        "Datasets e partições usados no estudo.",
        "tab:datasets-splits",
    )
    outputs.append(tables_dir / "tabela_01_datasets_splits.tex")

    arm_order = [
        "D0",
        "P1",
        "A1",
        "P1A1",
        "DRRT",
        "DRIT",
        "DRIF",
        "DRIFT",
        "DRIFTB",
        "DRIFT_SAM_BPM",
    ]
    ablation = []
    for arm in arm_order:
        row = require_one(
            summary, dataset="DFUC2022 interno", arm_id=arm, threshold_policy="selected"
        )
        ablation.append(
            {
                "arm_id": arm,
                "description": ARM_LABELS[arm],
                "threshold": {
                    "D0": 0.22,
                    "P1": 0.15,
                    "A1": 0.08,
                    "P1A1": 0.17,
                    "DRRT": 0.21,
                    "DRIT": 0.09,
                    "DRIF": 0.82,
                    "DRIFT": 0.18,
                    "DRIFTB": 0.05,
                    "DRIFT_SAM_BPM": 0.18,
                }[arm],
                "dice_macro": as_float(row, "dice_macro"),
                "dice_ci95_low": as_float(row, "dice_macro_ci95_low"),
                "dice_ci95_high": as_float(row, "dice_macro_ci95_high"),
                "dice_micro": as_float(row, "dice_micro"),
                "iou_macro": as_float(row, "iou_macro"),
                "precision_macro": as_float(row, "precision_macro"),
                "recall_macro": as_float(row, "recall_macro"),
                "boundary_iou_macro": as_float(row, "boundary_iou_macro"),
            }
        )
    p = tables_dir / "tabela_02_ablacao_teste_interno.csv"
    write_csv_rows(p, ablation, list(ablation[0]))
    outputs.append(p)
    write_latex_table(
        tables_dir / "tabela_02_ablacao_teste_interno.tex",
        [
            "Braço",
            "Descrição",
            "Limiar",
            "Dice macro",
            "IC 95%",
            "Dice micro",
            "IoU macro",
            "Boundary IoU",
        ],
        [
            [
                r["arm_id"],
                r["description"],
                fmt_metric(r["threshold"], 2),
                fmt_metric(r["dice_macro"]),
                f"[{fmt_metric(r['dice_ci95_low'])}; {fmt_metric(r['dice_ci95_high'])}]",
                fmt_metric(r["dice_micro"]),
                fmt_metric(r["iou_macro"]),
                fmt_metric(r["boundary_iou_macro"]),
            ]
            for r in ablation
        ],
        "Resultados dos braços no teste interno congelado do DFUC2022. A tabela descreve a avaliação final e não foi usada para selecionar modelos.",
        "tab:ablacao-teste-interno",
    )
    outputs.append(tables_dir / "tabela_02_ablacao_teste_interno.tex")

    contrast_rows = []
    keep_families = {"primary", "secondary", "exploratory"}
    for row in contrasts:
        if (
            row["metric"] != "dice"
            or row["threshold_policy"] != "selected"
            or row["family"] not in keep_families
        ):
            continue
        contrast_rows.append(
            {
                "family": row["family"],
                "contrast_id": row["contrast_id"],
                "candidate": row["candidate"],
                "control": row["control"],
                "mean_difference": as_float(row, "mean_difference"),
                "ci95_low": as_float(row, "ci95_low"),
                "ci95_high": as_float(row, "ci95_high"),
                "permutation_p_raw": row["permutation_p_raw"],
                "permutation_p_holm": row["permutation_p_holm"],
            }
        )
    p = tables_dir / "tabela_03_contrastes_pareados_dice.csv"
    write_csv_rows(p, contrast_rows, list(contrast_rows[0]))
    outputs.append(p)
    write_latex_table(
        tables_dir / "tabela_03_contrastes_pareados_dice.tex",
        ["Família", "Contraste", "Candidato", "Controle", "Δ Dice", "IC 95%", "p bruto", "p Holm"],
        [
            [
                r["family"],
                r["contrast_id"],
                r["candidate"],
                r["control"],
                fmt_metric(r["mean_difference"]),
                f"[{fmt_metric(r['ci95_low'])}; {fmt_metric(r['ci95_high'])}]",
                fmt_metric(r["permutation_p_raw"]),
                fmt_metric(r["permutation_p_holm"]),
            ]
            for r in contrast_rows
        ],
        "Contrastes pareados de Dice no teste interno. O ajuste de Holm aplica-se aos cinco contrastes primários.",
        "tab:contrastes-pareados",
    )
    outputs.append(tables_dir / "tabela_03_contrastes_pareados_dice.tex")

    final_rows = []
    for dataset in ["DFUC2022 interno", "FUSeg externo"]:
        for arm in ["DRIFT", "DRIFT_SAM_BPM"]:
            row = require_one(summary, dataset=dataset, arm_id=arm, threshold_policy="selected")
            gap = require_one(gaps, arm_id=arm, threshold_policy="selected")
            final_rows.append(
                {
                    "dataset": dataset,
                    "arm_id": arm,
                    "units": int(row["units"]),
                    "dice_macro": as_float(row, "dice_macro"),
                    "dice_macro_ci95_low": as_float(row, "dice_macro_ci95_low"),
                    "dice_macro_ci95_high": as_float(row, "dice_macro_ci95_high"),
                    "dice_micro": as_float(row, "dice_micro"),
                    "dice_micro_ci95_low": as_float(row, "dice_micro_ci95_low"),
                    "dice_micro_ci95_high": as_float(row, "dice_micro_ci95_high"),
                    "external_minus_internal_macro": as_float(gap, "external_minus_internal_macro")
                    if dataset == "FUSeg externo"
                    else "",
                    "gap_ci95_low": as_float(gap, "gap_ci95_low")
                    if dataset == "FUSeg externo"
                    else "",
                    "gap_ci95_high": as_float(gap, "gap_ci95_high")
                    if dataset == "FUSeg externo"
                    else "",
                }
            )
    p = tables_dir / "tabela_04_teste_interno_externo.csv"
    write_csv_rows(p, final_rows, list(final_rows[0]))
    outputs.append(p)
    write_latex_table(
        tables_dir / "tabela_04_teste_interno_externo.tex",
        [
            "Dataset",
            "Método",
            "n",
            "Dice macro",
            "IC 95% macro",
            "Dice micro",
            "IC 95% micro",
            "Gap externo-interno",
        ],
        [
            [
                r["dataset"],
                ARM_LABELS[r["arm_id"]],
                r["units"],
                fmt_metric(r["dice_macro"]),
                f"[{fmt_metric(r['dice_macro_ci95_low'])}; {fmt_metric(r['dice_macro_ci95_high'])}]",
                fmt_metric(r["dice_micro"]),
                f"[{fmt_metric(r['dice_micro_ci95_low'])}; {fmt_metric(r['dice_micro_ci95_high'])}]",
                fmt_metric(r["external_minus_internal_macro"]),
            ]
            for r in final_rows
        ],
        "Desempenho final no teste interno DFUC2022 e na avaliação externa FUSeg, sem adaptação externa.",
        "tab:teste-interno-externo",
    )
    outputs.append(tables_dir / "tabela_04_teste_interno_externo.tex")

    ext_dice = require_one(external_contrasts, metric="dice", threshold_policy="selected")
    ext_effect = [
        {
            "contrast": "DRIFT_SAM_BPM_minus_DRIFT",
            "units": int(ext_dice["images"]),
            "mean_difference": as_float(ext_dice, "mean_difference"),
            "ci95_low": as_float(ext_dice, "ci95_low"),
            "ci95_high": as_float(ext_dice, "ci95_high"),
            "improved": int(ext_dice["improved"]),
            "equal": int(ext_dice["equal"]),
            "worsened": int(ext_dice["worsened"]),
            "permutation_p_raw": as_float(ext_dice, "permutation_p_raw"),
        }
    ]
    p = tables_dir / "tabela_05_efeito_sam_externo.csv"
    write_csv_rows(p, ext_effect, list(ext_effect[0]))
    outputs.append(p)
    return outputs


def copy_figures(project: Path, figures_dir: Path) -> list[tuple[Path, Path]]:
    sources = [
        (
            project / "results/incoming/day10_20260903/DRIFT/training_curves.png",
            figures_dir / "figura_02_curvas_treinamento_dr_ift.png",
        ),
        (
            project / "results/incoming/day10_20260903/DRIFT/threshold_curves.png",
            figures_dir / "figura_03_sensibilidade_threshold_dr_ift.png",
        ),
        (
            project
            / "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/figures/final_internal_arm_performance.png",
            figures_dir / "figura_04_desempenho_bracos_teste_interno.png",
        ),
        (
            project
            / "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/figures/final_internal_paired_effects.png",
            figures_dir / "figura_05_efeitos_pareados_teste_interno.png",
        ),
        (
            project
            / "results/day16_analysis/DAY16_FINAL_20260930/figures/macro_micro_comparison.png",
            figures_dir / "figura_06_macro_micro_interno_externo.png",
        ),
        (
            project
            / "results/day16_analysis/DAY16_FINAL_20260930/figures/dice_by_area_quintile.png",
            figures_dir / "figura_07_dice_por_quintil_area.png",
        ),
        (
            project / "results/day16_analysis/DAY16_FINAL_20260930/figures/error_taxonomy.png",
            figures_dir / "figura_08_taxonomia_erros.png",
        ),
        (
            project
            / "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/fixed_test_panel_performance_blind.png",
            figures_dir / "figura_09_painel_qualitativo_dfuc.png",
        ),
        (
            project
            / "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z/fixed_external_panel_performance_blind.png",
            figures_dir / "figura_10_painel_qualitativo_fuseg.png",
        ),
    ]
    for source, destination in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
    return sources


def write_documentation(
    project: Path, output: Path, manifest_rows: list[dict[str, object]]
) -> None:
    text = """# Tabelas e figuras finais

Este diretório reúne material pronto para inserção no TCC. As tabelas foram apenas reorganizadas a partir dos resultados finais aprovados. Nenhum modelo, limiar, checkpoint ou análise estatística foi recalculado ou selecionado nesta etapa.

## Como usar

- `tabelas/*.csv`: versão auditável e editável.
- `tabelas/*.tex`: versão pronta para LaTeX; requer `\\usepackage{booktabs}` e `\\usepackage{graphicx}`.
- `figuras/*.png`: arquivos em resolução original, exceto o diagrama do pipeline, criado nesta etapa.
- `LEGENDAS_E_FONTES.md`: legendas sugeridas e origem exata de cada item.
- `manifesto_artefatos.csv`: inventário com SHA-256.

## Regra de interpretação

O Dice macro por unidade canônica é o comparador científico comum entre DFUC2022 e FUSeg. O Dice micro aparece como contextualização adicional, inclusive por corresponder à agregação global de pixels usada no FUSeg. Resultados do teste interno não foram usados para selecionar modelos; o FUSeg foi avaliado sem adaptação.
"""
    (output / "README.md").write_text(text, encoding="utf-8")

    legends = """# Legendas e fontes

## Figuras

1. **Pipeline experimental.** Fluxo de curadoria, desenvolvimento, definição da rede central e avaliações finais. Duplicatas exatas foram agrupadas antes do split; a máscara canônica resulta da união lógica (OR) das referências válidas do grupo. O teste interno e o conjunto externo permaneceram fora da seleção de modelos. Fonte: elaboração própria com base no protocolo congelado.
2. **Curvas de treinamento da DR-IFT.** Loss de treino e validação, Dice macro de validação no limiar 0,5 e taxa de aprendizado ao longo das épocas. A linha vertical indica o checkpoint escolhido exclusivamente pela validação. Fonte: execução `DRIFT_seed42_20260903T033716Z`.
3. **Sensibilidade ao limiar na validação.** Dice, IoU, precisão e recall macro em função do limiar aplicado às probabilidades da DR-IFT. A linha tracejada indica o limiar 0,18, escolhido pelo maior Dice macro na validação, com desempate pela proximidade de 0,5. Fonte: execução `DRIFT_seed42_20260903T033716Z`.
4. **Desempenho dos braços no teste interno.** Métricas dos braços pré-especificados nas 294 unidades canônicas do teste DFUC2022. A avaliação foi conjunta, após lacre, e não orientou seleção de modelo. Fonte: `FINAL_INTERNAL_TEST_20260929T195227Z`.
5. **Efeitos pareados no teste interno.** Diferença média por imagem e IC 95% por bootstrap pareado para os contrastes planejados. O zero indica ausência de diferença média. Fonte: `FINAL_INTERNAL_TEST_20260929T195227Z`.
6. **Comparação macro e micro entre datasets.** Dice macro por unidade e Dice micro global para DR-IFT e DR-IFT+SAM no teste interno DFUC2022 e na avaliação externa FUSeg. As barras representam IC 95% por bootstrap de unidades canônicas. Fonte: análise da análise estatística.
7. **Dice por quintil relativo de área da lesão.** Desempenho de DR-IFT e DR-IFT+SAM estratificado por quintis definidos separadamente em cada dataset, sem seleção por desempenho. Fonte: análise da análise estatística.
8. **Taxonomia métrica de erros.** Frequência de categorias descritivas de falha derivadas de métricas pré-especificadas, sem excluir casos. Fonte: análise da análise estatística.
9. **Painel qualitativo fixo do teste interno DFUC2022.** Para cada imagem: entrada, referência, probabilidade da DR-IFT, predição DR-IFT, refinamento DR-IFT+SAM e DR-IFT-B. Os casos foram fixados independentemente do desempenho e incluem acertos e limitações. Fonte: `FINAL_INTERNAL_TEST_20260929T195227Z`.
10. **Painel qualitativo fixo da avaliação externa FUSeg.** Comparação da DR-IFT e do refinamento DR-IFT+SAM nas mesmas imagens externas, sem adaptação ao FUSeg. Fonte: `FINAL_EXTERNAL_FUSEG_20260930T151950Z`.

## Tabelas

1. **Datasets e splits.** Contagens após a política canônica de duplicatas e referências.
2. **Ablação no teste interno.** Resultados descritivos de todos os braços pré-especificados no teste congelado; não usar esta tabela como justificativa de seleção retrospectiva.
3. **Contrastes pareados de Dice.** Comparações planejadas no teste interno, com diferença média e IC 95%.
4. **Teste interno e externo.** Síntese de DR-IFT e DR-IFT+SAM nos dois datasets, mantendo macro e micro explicitamente separados.
5. **Efeito externo do SAM.** Comparação pareada DR-IFT+SAM menos DR-IFT no FUSeg.
"""
    (output / "LEGENDAS_E_FONTES.md").write_text(legends, encoding="utf-8")
    write_csv_rows(
        output / "manifesto_artefatos.csv",
        manifest_rows,
        ["kind", "item", "path", "source", "sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("output/day17_final_assets"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else project / args.output_dir
    tables_dir = output / "tabelas"
    figures_dir = output / "figuras"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    table_paths = build_tables(project, tables_dir)
    pipeline = figures_dir / "figura_01_pipeline_experimental.png"
    draw_pipeline(pipeline)
    copied = copy_figures(project, figures_dir)

    manifest_rows: list[dict[str, object]] = []
    manifest_rows.append(
        {
            "kind": "figure",
            "item": "pipeline",
            "path": str(pipeline.relative_to(project)),
            "source": "configs/final_internal_test_protocol_v1.yaml; docs/CONTEXTO_ATUAL.md",
            "sha256": sha256(pipeline),
        }
    )
    for source, destination in copied:
        manifest_rows.append(
            {
                "kind": "figure",
                "item": destination.stem,
                "path": str(destination.relative_to(project)),
                "source": str(source.relative_to(project)),
                "sha256": sha256(destination),
            }
        )
    table_sources = {
        "tabela_01_datasets_splits": "docs/CONTEXTO_ATUAL.md; configs/final_internal_test_protocol_v1.yaml",
        "tabela_02_ablacao_teste_interno": "results/day16_analysis/DAY16_FINAL_20260930/dataset_summary_macro_micro.csv",
        "tabela_03_contrastes_pareados_dice": "results/final_internal_test/FINAL_INTERNAL_TEST_20260929T195227Z/paired_comparisons.csv",
        "tabela_04_teste_interno_externo": "results/day16_analysis/DAY16_FINAL_20260930/dataset_summary_macro_micro.csv; results/day16_analysis/DAY16_FINAL_20260930/cross_dataset_macro_comparison.csv",
        "tabela_05_efeito_sam_externo": "results/final_external_fuseg/FINAL_EXTERNAL_FUSEG_20260930T151950Z/external_sam_comparison.csv",
    }
    for path in table_paths:
        manifest_rows.append(
            {
                "kind": "table",
                "item": path.stem,
                "path": str(path.relative_to(project)),
                "source": table_sources[path.stem],
                "sha256": sha256(path),
            }
        )
    write_documentation(project, output, manifest_rows)

    report = {
        "status": "completed",
        "day": 17,
        "purpose": "final_tables_and_figures",
        "output_directory": str(output),
        "table_sets": 5,
        "table_files": len(table_paths),
        "figures": 1 + len(copied),
        "model_inference_performed": False,
        "threshold_search_performed": False,
        "model_selection_performed": False,
        "statistics_recomputed": False,
        "source_policy": "approved_final_artifacts_only",
    }
    (output / "day17_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
