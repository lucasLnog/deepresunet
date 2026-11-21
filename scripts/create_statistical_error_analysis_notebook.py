#!/usr/bin/env python3
"""Gera o notebook explicativo e executavel da análise estatística."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind: str, source: str) -> dict:
    item = {
        "cell_type": kind,
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


cells = [
    cell(
        "markdown",
        (
            '# Estatística e análise de erros\n'
            '\n'
            'Esta etapa reúne os resultados finais já congelados do **DFUC2022 interno** e'
            ' do **FUSeg externo**. Ela não treina, não executa inferência, não procura '
            'threshold e não escolhe modelos.\n'
            '\n'
            'A análise distingue dois objetivos:\n'
            '\n'
            '1. **Comparação descritiva entre conjuntos:** Dice macro nos dois conjuntos, '
            'dando peso igual a cada unidade canônica.\n'
            '2. **Contextualização das métricas usadas nos desafios:** Dice macro no '
            'DFUC2022 e Dice micro (*data-based*) no FUSeg. As partições avaliadas no '
            'estudo diferem dos testes privados dos desafios.\n'
            '\n'
            'O resultado principal continua sendo a DR-IFT no threshold congelado. O '
            'threshold 0,5 é apenas sensibilidade e o SAM é um refinador secundário.'
        ),
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import os, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
%cd {PROJECT_ROOT}

from tcc_segmentation.statistical_analysis import discover_result_directory

RAIZES = [PROJECT_ROOT / 'results/final_evaluation', PROJECT_ROOT / 'results']
INTERNAL = discover_result_directory(
    RAIZES,
    marker='test_per_image_all_arms.csv',
    required_files=[
        'test_per_image_all_arms.csv',
        'test_summary_all_arms.csv',
        'paired_comparisons.csv',
    ],
)
EXTERNAL = discover_result_directory(
    RAIZES,
    marker='external_per_unit.csv',
    required_files=['external_per_unit.csv', 'external_summary.csv', 'external_sam_comparison.csv'],
)
OUTPUT = PROJECT_ROOT / 'results/day16_analysis/DAY16_FINAL_20260930'
print('Teste interno:', INTERNAL)
print('Avaliação externa:', EXTERNAL)
print('Saída:', OUTPUT)""",
    ),
    cell(
        "markdown",
        """## 1. Testes dos cálculos

Os testes confirmam que o micro-Dice usa contagens agregadas, que o bootstrap é reproduzível, que os quintis dependem somente da área da referência e que a taxonomia não altera nenhuma amostra.""",
    ),
    cell(
        "code",
        """env = os.environ.copy()
env['PYTHONPATH'] = str(SRC_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_statistical_analysis.py'],
    text=True,
    capture_output=True,
    env=env,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()""",
    ),
    cell(
        "markdown",
        """## 2. Gerar a consolidação

O bootstrap reamostra a unidade canônica 10.000 vezes. No FUSeg, referências múltiplas já foram resumidas dentro da unidade antes desta etapa. O gap externo menos interno é descritivo e usa reamostragem independente, pois as imagens dos datasets não formam pares.""",
    ),
    cell(
        "code",
        """command = [
    sys.executable,
    'scripts/run_statistical_analysis.py',
    '--internal-results',
    str(INTERNAL),
    '--external-results',
    str(EXTERNAL),
    '--output-dir',
    str(OUTPUT),
    '--bootstrap-samples',
    '10000',
    '--seed',
    '20260930',
]
result = subprocess.run(command, text=True, capture_output=True, env=env)
print(result.stdout)
print(result.stderr)
result.check_returncode()""",
    ),
    cell(
        "markdown",
        """## 3. Macro e micro

- **Macro:** calcula a métrica por imagem e tira a média. É a medida padronizada para comparar DFUC e FUSeg.
- **Micro:** soma TP, FP e FN das unidades e calcula a métrica uma vez. É a agregação data-based usada para contextualizar o FUSeg.

Não usamos macro em um dataset e micro no outro para estimar a mudança de domínio.""",
    ),
    cell(
        "code",
        """import pandas as pd
from IPython.display import display, Image as DisplayImage

summary = pd.read_csv(OUTPUT / 'dataset_summary_macro_micro.csv')
common = summary[
    summary.apply(
        lambda r: (r.arm_id, r.threshold_policy)
        in {('DRIFT', 'selected'), ('DRIFT', 'fixed_0_5'), ('DRIFT_SAM_BPM', 'selected')},
        axis=1,
    )
]
display(
    common[
        [
            'dataset',
            'arm_id',
            'threshold_policy',
            'units',
            'dice_macro',
            'dice_macro_ci95_low',
            'dice_macro_ci95_high',
            'dice_micro',
            'dice_micro_ci95_low',
            'dice_micro_ci95_high',
        ]
    ].round(6)
)
display(DisplayImage(filename=str(OUTPUT / 'figures/macro_micro_comparison.png')))""",
    ),
    cell(
        "markdown",
        """## 4. Generalização externa

A tabela abaixo usa somente macro contra macro. O intervalo representa a incerteza do gap descritivo FUSeg menos DFUC. Ele mede a mudança observada entre datasets, mas não isola causalmente qual característica do domínio produziu a diferença.""",
    ),
    cell(
        "code",
        """gap = pd.read_csv(OUTPUT / 'cross_dataset_macro_comparison.csv')
display(gap.round(6))""",
    ),
    cell(
        "markdown",
        """## 5. Desempenho por tamanho da lesão

Cada dataset foi dividido em cinco grupos de tamanho relativo, sem consultar o desempenho. Q1 contém as menores lesões daquele dataset e Q5 as maiores. Essa análise mostra onde o modelo encontra mais dificuldade sem transformar o FUSeg em conjunto de ajuste.""",
    ),
    cell(
        "code",
        """area = pd.read_csv(OUTPUT / 'dice_by_relative_area_quintile.csv')
display(area.round(6))
display(DisplayImage(filename=str(OUTPUT / 'figures/dice_by_area_quintile.png')))""",
    ),
    cell(
        "markdown",
        """## 6. Taxonomia descritiva

Os rótulos são gerados pelas métricas, não por julgamento clínico:

- predição vazia;
- ausência de sobreposição;
- baixo Dice com recall limitante, associado a tendência de subsegmentação;
- baixo Dice com precisão limitante, associado a tendência de sobresegmentação;
- sobreposição parcial (`0,50 ≤ Dice < 0,75`);
- alta sobreposição (`Dice ≥ 0,75`).

Os casos extremos permanecem no estudo. A lista serve apenas para revisão visual e discussão.""",
    ),
    cell(
        "code",
        """taxonomy = pd.read_csv(OUTPUT / 'error_taxonomy.csv')
display(taxonomy.round(3))
display(DisplayImage(filename=str(OUTPUT / 'figures/error_taxonomy.png')))
extremes = pd.read_csv(OUTPUT / 'extreme_cases_for_review.csv')
display(extremes.head(24))""",
    ),
    cell(
        "markdown",
        """## 7. Painéis finais congelados

Os painéis abaixo foram definidos sem consultar o desempenho individual. Eles complementam as métricas e ajudam a observar contorno, falsos positivos, lesões pequenas e mudança de domínio.""",
    ),
    cell(
        "code",
        """display(DisplayImage(filename=str(INTERNAL / 'fixed_test_panel_performance_blind.png')))
display(DisplayImage(filename=str(EXTERNAL / 'fixed_external_panel_performance_blind.png')))""",
    ),
    cell(
        "markdown",
        (
            '## 8. O que deve ser enviado\n'
            '\n'
            'A pasta `results/day16_analysis/DAY16_FINAL_20260930` contém as tabelas, três'
            ' figuras e o relatório de auditoria. Nenhum dataset ou checkpoint é incluído.'
        ),
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "25_statistical_error_analysis.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/25_statistical_error_analysis.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
