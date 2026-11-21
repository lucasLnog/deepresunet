#!/usr/bin/env python3
"""Gera o notebook didático da ablação DR-IFT-B."""

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


setup = """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import os, shutil, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
%cd {PROJECT_ROOT}
!pip -q install albumentations pyyaml pytest
source = PROJECT_ROOT / 'Dataset'
assert source.is_dir(), f'Dataset ausente: {source}'
shutil.copytree(source, DATA_ROOT / 'Dataset', dirs_exist_ok=True)
assert (DATA_ROOT / 'Dataset/DFUC2022_train_images/100001.jpg').is_file()
print('Cache preparado:', DATA_ROOT / 'Dataset')"""

helpers = """def run_stream(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='')
    code = process.wait()
    print(f'\\nreturncode={code}')
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_or_resume(config, output_root, prefix):
    root = PROJECT_ROOT / output_root
    candidates = sorted(root.glob(prefix + '_*')) if root.is_dir() else []
    incomplete = [
        p
        for p in candidates
        if (p / 'config_used.yaml').is_file() and not (p / 'final_summary.json').is_file()
    ]
    command = [
        sys.executable,
        'scripts/run_baseline_d0.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--data-root',
        str(DATA_ROOT),
        '--config',
        config,
    ]
    if incomplete:
        command += ['--resume-dir', str(incomplete[-1])]
        print('Retomando:', incomplete[-1].name)
    run_stream(command)"""

cells = [
    cell(
        "markdown",
        (
            '# DR-IFT-B: ablação controlada de loss de perímetro\n'
            '\n'
            '**Objetivo:** estimar o efeito do termo de perímetro mantendo a arquitetura e'
            ' as demais condições da DR-IFT.\n'
            '\n'
            'Esta é uma análise **exploratória pós-registro**, documentada antes de '
            'observar seus resultados. O teste interno e o FUSeg continuam bloqueados. Não'
            ' haverá busca do melhor peso: usamos uma única escolha, `0,10`.'
        ),
    ),
    cell("code", setup),
    cell("code", helpers),
    cell(
        "markdown",
        """## 1. O que muda — e o que não muda

A loss regional anterior continua sendo `0,5 × BCE + 0,5 × Soft Dice`. A DR-IFT-B usa:

`loss total = 0,90 × loss regional + 0,10 × discrepância relativa de perímetro`

Só a função objetivo muda. Permanecem iguais: imagens e splits, P0+A1, 384×512, batch, arquitetura DeepResUNet/ResNet-34, pesos ImageNet, liberação progressiva, learning rates, checkpoint, threshold e seeds.

O termo de perímetro compara o tamanho do contorno suave previsto com o contorno da referência. Ele complementa a sobreposição; isoladamente, não garante que a borda esteja no local correto.""",
    ),
    cell(
        "code",
        """import numpy as np, matplotlib.pyplot as plt, torch

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from tcc_segmentation.training.losses import SoftPerimeterLoss

mask = np.zeros((64, 96), dtype=np.float32)
mask[18:48, 28:70] = 1
values = torch.tensor(mask)[None, None]
eroded = -torch.nn.functional.max_pool2d(-values, 3, 1, 1)
contour = (values - eroded).clamp_min(0)[0, 0].numpy()
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].imshow(mask, cmap='gray')
ax[0].set_title('Região')
ax[1].imshow(contour, cmap='magma')
ax[1].set_title('Contorno usado pela loss')
for item in ax:
    item.axis('off')
plt.tight_layout()
plt.show()""",
    ),
    cell(
        "markdown",
        """## 2. Auditoria antes do treinamento

Os testes abaixo confirmam o peso único, as três seeds, a diferenciação da loss, a retropropagação e o bloqueio das partições protegidas.""",
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
        'tests/test_boundary_loss_ablation.py',
        'tests/test_fine_tuning.py',
        'tests/test_training_components.py',
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
        """## 3. Seed 42

Execute uma seed por vez. Se o Colab desconectar, a célula tenta retomar a execução incompleta mais recente.""",
    ),
    cell(
        "code",
        """run_or_resume('configs/boundary_loss_ablation_seed42.yaml', 'results/day14/seed42/DRIFTB', 'DRIFTB_seed42')""",
    ),
    cell("markdown", "## 4. Seed 1337"),
    cell(
        "code",
        """run_or_resume(
    'configs/boundary_loss_ablation_seed1337.yaml', 'results/day14/seed1337/DRIFTB', 'DRIFTB_seed1337'
)""",
    ),
    cell("markdown", "## 5. Seed 2026"),
    cell(
        "code",
        """run_or_resume(
    'configs/boundary_loss_ablation_seed2026.yaml', 'results/day14/seed2026/DRIFTB', 'DRIFTB_seed2026'
)""",
    ),
    cell(
        "markdown",
        (
            '## 6. Painel de validação e exportação dos resultados\n'
            '\n'
            'Os painéis continuam usando os mesmos casos fixos. Não escolha exemplos mais '
            'bonitos. A decisão será feita com as 294 imagens e as três seeds, e não pelo '
            'painel.'
        ),
    ),
    cell(
        "code",
        """from IPython.display import display, Image

runs = []
for seed in (42, 1337, 2026):
    found = sorted((PROJECT_ROOT / f'results/day14/seed{seed}/DRIFTB').glob(f'DRIFTB_seed{seed}_*'))
    assert found, f'Execução da seed {seed} ausente'
    run = found[-1]
    runs.append(run)
    assert (run / 'final_summary.json').is_file(), f'Execução incompleta: {run}'
    print('Seed', seed, run.name)
    display(Image(filename=str(run / 'training_curves.png'), width=1100))
    display(Image(filename=str(run / 'fixed_validation_panel.png'), width=1100))

archive = shutil.make_archive(
    '/content/boundary_loss_ablation_results', 'zip', root_dir=PROJECT_ROOT / 'results/day14'
)
print('Pacote pronto:', archive)
print(
    'O ZIP contém os resultados registrados da ablação.'
)""",
    ),
    cell(
        "markdown",
        """## Critério já congelado

A DR-IFT-B só avança como candidata exploratória se:

1. a média do ganho de Dice nas três seeds for pelo menos `+0,005`;
2. ela superar a DR-IFT original em pelo menos duas das três seeds;
3. qualquer melhora de borda não vier acompanhada de perda relevante no Dice.

Caso contrário, mantemos a DR-IFT original. Depois da execução, faremos a comparação pareada; **não abra o teste interno ainda**.""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "19_boundary_loss_ablation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/19_boundary_loss_ablation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
