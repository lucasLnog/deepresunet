#!/usr/bin/env python3
"""Gera o notebook de Boundary IoU pareado da avaliação de Boundary IoU."""

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
            '# Boundary IoU na validação congelada\n'
            '\n'
            '**Objetivo:** comparar a DR-IFT-B e a DR-IFT pelo Boundary IoU na partição de'
            ' validação.\n'
            '\n'
            'Esta etapa não treina modelos, não escolhe novos pesos e não procura outro '
            'threshold. Ela reutiliza os seis melhores checkpoints e o threshold que cada '
            'execução já havia escolhido na validação. O teste interno e o FUSeg '
            'permanecem fechados.\n'
            '\n'
            'O Boundary IoU mede a interseção sobre união somente em uma faixa ao redor '
            'dos contornos. Usaremos a regra já congelada de **2% da diagonal da imagem**,'
            ' baseada em Cheng et al. (CVPR 2021). O resultado é complementar ao Dice e '
            'não mudará sozinho o modelo principal.'
        ),
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import json, os, shutil, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
OUTPUT_ROOT = PROJECT_ROOT / 'results/analysis/day14/boundary_validation'
%cd {PROJECT_ROOT}
!pip -q install albumentations pyyaml pytest
source = PROJECT_ROOT / 'Dataset'
assert source.is_dir(), f'Dataset ausente: {source}'
shutil.copytree(source, DATA_ROOT / 'Dataset', dirs_exist_ok=True)
assert (DATA_ROOT / 'Dataset/DFUC2022_train_images/100001.jpg').is_file()
print('Dataset em cache e saída em:', OUTPUT_ROOT)""",
    ),
    cell(
        "markdown",
        """## 1. Auditoria do protocolo

Os testes confirmam: 294 imagens da validação, três seeds, faixa de 2%, pareamento por imagem e bloqueio de teste/FUSeg.""",
    ),
    cell(
        "code",
        """env = os.environ.copy()
env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
result = subprocess.run(
    [
        sys.executable,
        '-m',
        'pytest',
        '-q',
        'tests/test_boundary_iou_validation.py',
        'tests/test_sam_validation.py',
    ],
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
        """## 2. Localizar os seis experimentos concluídos

Cada linha deve apontar para uma pasta que contenha `best_checkpoint.pt`, `config_used.yaml`, `training_summary.json` e `validation_per_image_selected.csv`. Assim evitamos misturar execuções.""",
    ),
    cell(
        "code",
        """def latest_complete(parent, prefix):
    found = sorted(parent.glob(prefix + '_*')) if parent.is_dir() else []
    required = (
        'best_checkpoint.pt',
        'config_used.yaml',
        'training_summary.json',
        'validation_per_image_selected.csv',
    )
    complete = [run for run in found if all((run / name).is_file() for name in required)]
    assert complete, f'Execução completa ausente em {parent} com prefixo {prefix}'
    return complete[-1]


runs = {
    (42, 'DRIFT'): latest_complete(PROJECT_ROOT / 'results/day10/DRIFT', 'DRIFT_seed42'),
    (1337, 'DRIFT'): latest_complete(
        PROJECT_ROOT / 'results/day11/seed1337/DRIFT', 'DRIFT_seed1337'
    ),
    (2026, 'DRIFT'): latest_complete(
        PROJECT_ROOT / 'results/day11/seed2026/DRIFT', 'DRIFT_seed2026'
    ),
    (42, 'DRIFTB'): latest_complete(PROJECT_ROOT / 'results/day14/seed42/DRIFTB', 'DRIFTB_seed42'),
    (1337, 'DRIFTB'): latest_complete(
        PROJECT_ROOT / 'results/day14/seed1337/DRIFTB', 'DRIFTB_seed1337'
    ),
    (2026, 'DRIFTB'): latest_complete(
        PROJECT_ROOT / 'results/day14/seed2026/DRIFTB', 'DRIFTB_seed2026'
    ),
}
for key, value in runs.items():
    print(key, '->', value.name)""",
    ),
    cell(
        "markdown",
        """## 3. Reproduzir as máscaras e calcular Boundary IoU

A célula falha se o Dice recalculado não reproduzir o CSV original. Essa verificação garante que checkpoint, dados e threshold não foram trocados.""",
    ),
    cell(
        "code",
        """def run_stream(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='')
    code = process.wait()
    print(f'\\nreturncode={code}')
    if code:
        raise subprocess.CalledProcessError(code, command)


for (seed, arm), run in runs.items():
    destination = OUTPUT_ROOT / f'seed{seed}' / arm
    print(f'\\n=== seed {seed} | {arm} ===')
    run_stream(
        [
            sys.executable,
            'scripts/evaluate_boundary_iou_validation.py',
            '--project-root',
            str(PROJECT_ROOT),
            '--data-root',
            str(DATA_ROOT),
            '--run-dir',
            str(run),
            '--output-dir',
            str(destination),
            '--boundary-dilation-ratio',
            '0.02',
        ]
    )""",
    ),
    cell(
        "markdown",
        """## 4. Comparação pareada

Para cada seed, subtraímos `Boundary IoU da DR-IFT-B − Boundary IoU da DR-IFT` na mesma imagem. O intervalo de confiança bootstrap descreve a incerteza e o teste de troca de sinais verifica a compatibilidade com diferença média zero.""",
    ),
    cell(
        "code",
        """run_stream(
    [
        sys.executable,
        'scripts/compare_boundary_iou_validation.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--input-root',
        str(OUTPUT_ROOT),
        '--output-dir',
        str(OUTPUT_ROOT / 'comparison'),
    ]
)

import pandas as pd
from IPython.display import display

table = pd.read_csv(OUTPUT_ROOT / 'comparison/boundary_per_seed.csv')
display(table.round(6))
summary = json.loads(
    (OUTPUT_ROOT / 'comparison/boundary_comparison_summary.json').read_text(encoding='utf-8')
)
print('Diferença média nas três seeds:', round(summary['mean_difference_across_seeds'], 6))
print('Seeds favoráveis à DR-IFT-B:', summary['seeds_favoring_DR-IFT-B'], 'de 3')""",
    ),
    cell(
        "code",
        """import matplotlib.pyplot as plt

labels = [str(seed) for seed in table['seed']]
values = table['difference'].to_numpy()
low = values - table['ci95_low'].to_numpy()
high = table['ci95_high'].to_numpy() - values
colors = ['#2a9d8f' if value > 0 else '#e76f51' for value in values]
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(labels, values, yerr=[low, high], fmt='none', ecolor='#333333', capsize=5)
ax.scatter(labels, values, c=colors, s=70, zorder=3)
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.set(
    xlabel='Seed',
    ylabel='Δ Boundary IoU (DR-IFT-B − DR-IFT)',
    title='Comparação pareada de contorno na validação',
)
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
figure = OUTPUT_ROOT / 'comparison/boundary_iou_comparison.png'
plt.savefig(figure, dpi=180, bbox_inches='tight')
plt.show()""",
    ),
    cell(
        "markdown",
        """## 5. Baixar somente os resultados leves

O pacote abaixo contém CSV, JSON e a figura. Ele não inclui Dataset nem checkpoints.""",
    ),
    cell(
        "code",
        """archive = shutil.make_archive(
    '/content/boundary_iou_validation_results', 'zip', root_dir=OUTPUT_ROOT
)
print('Pacote pronto:', archive)
from google.colab import files

files.download(archive)""",
    ),
    cell(
        "markdown",
        (
            '### Interpretação das diferenças pareadas\n'
            '\n'
            '- `Δ > 0`: a DR-IFT-B delimitou melhor os contornos naquela seed.\n'
            '- Se o IC95% atravessar zero, o ganho observado ainda é compatível com '
            'ausência de diferença nessa validação.\n'
            '- O resultado será relatado junto ao Dice. Boundary IoU não autoriza escolher'
            ' outro threshold nem abrir o teste.\n'
            '\n'
            'O arquivo de saída é `boundary_iou_validation_results.zip`.'
        ),
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "20_boundary_iou_validation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/20_boundary_iou_validation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
