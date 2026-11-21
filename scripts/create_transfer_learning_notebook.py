#!/usr/bin/env python3
"""Gera o notebook didático da comparação de inicialização do encoder para Google Colab."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
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
        "# Transfer learning: DR-RT × DR-IT\n\nObjetivo: medir somente o efeito da inicialização ImageNet. O decoder residual com attention gates da DeepResUNet é mantido. P0 + A1 e todos os demais controles são idênticos. Pare se uma verificação falhar; teste e FUSeg seguem bloqueados.",
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import shutil, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
%cd {PROJECT_ROOT}
!pip -q install albumentations pyyaml pytest
source_dataset = PROJECT_ROOT / 'Dataset'
assert source_dataset.is_dir(), f'Dataset não encontrado no Drive: {source_dataset}'
shutil.copytree(source_dataset, DATA_ROOT / 'Dataset', dirs_exist_ok=True)
required_image = DATA_ROOT / 'Dataset/DFUC2022_train_images/100001.jpg'
assert required_image.is_file(), f'Cache incompleto: {required_image}'
print('Cache local preparado:', DATA_ROOT / 'Dataset')""",
    ),
    cell(
        "markdown",
        "## 1. Gate técnico\n\nSem usar imagens, confirma formato, logits, gradientes, diferença dos encoders e igualdade do decoder.",
    ),
    cell(
        "code",
        """subprocess.run(
    [
        sys.executable,
        'scripts/check_deep_res_unet_resnet34.py',
        '--project-root',
        str(PROJECT_ROOT),
    ],
    check=True,
)""",
    ),
    cell(
        "markdown",
        "## 2. Contrato experimental\n\nOs testes impedem diferenças acidentais entre os braços.",
    ),
    cell(
        "code",
        """import os

test_env = os.environ.copy()
test_env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + test_env.get('PYTHONPATH', '')
result = subprocess.run(
    [
        sys.executable,
        '-m',
        'pytest',
        '-q',
        'tests/test_transfer_learning.py',
        'tests/test_deep_res_unet_resnet34.py',
    ],
    text=True,
    capture_output=True,
    env=test_env,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()""",
    ),
    cell("markdown", "## 3. DR-RT — encoder aleatório treinável"),
    cell(
        "code",
        """command = [
    sys.executable,
    'scripts/run_baseline_d0.py',
    '--project-root',
    str(PROJECT_ROOT),
    '--data-root',
    str(DATA_ROOT),
    '--config',
    'configs/transfer_learning_random_encoder.yaml',
]
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
for line in process.stdout:
    print(line, end='')
returncode = process.wait()
print(f'\\nreturncode={returncode}')
if returncode != 0:
    raise subprocess.CalledProcessError(returncode, command)""",
    ),
    cell(
        "markdown",
        "## 4. DR-IT — encoder ImageNet treinável\n\nA única intervenção científica é a inicialização do encoder.",
    ),
    cell(
        "code",
        """command = [
    sys.executable,
    'scripts/run_baseline_d0.py',
    '--project-root',
    str(PROJECT_ROOT),
    '--data-root',
    str(DATA_ROOT),
    '--config',
    'configs/transfer_learning_imagenet_encoder.yaml',
]
process = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
for line in process.stdout:
    print(line, end='')
returncode = process.wait()
print(f'\\nreturncode={returncode}')
if returncode != 0:
    raise subprocess.CalledProcessError(returncode, command)""",
    ),
    cell(
        "markdown",
        (
            '## 5. Saídas\n'
            '\n'
            'Os resultados ficam em `results/day9/DRRT` e `results/day9/DRIT`. O relatório'
            ' do encoder está em '
            '`results/architecture/deep_res_unet_resnet34_gate_report.json`. A comparação '
            'na validação é pareada por imagem.'
        ),
    ),
]
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "13_transfer_learning.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
destination = ROOT / "notebooks/13_transfer_learning.ipynb"
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(destination)
