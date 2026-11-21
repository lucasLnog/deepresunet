#!/usr/bin/env python3
"""Gera o notebook didático da comparação de ajuste fino."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": [x + "\n" for x in source.splitlines()]}
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
stream = """def run_stream(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='')
    code = process.wait()
    print(f'\\nreturncode={code}')
    if code:
        raise subprocess.CalledProcessError(code, command)"""
cells = [
    cell(
        "markdown",
        "# Política de fine-tuning\n\nDR-IF mantém o encoder ImageNet fixo. DR-IFT libera layer4 na época 16 e layer3 na época 31. O DR-IT anterior é a referência de liberação total imediata. Teste e FUSeg permanecem bloqueados.",
    ),
    cell("code", setup),
    cell("code", stream),
    cell(
        "markdown",
        "## 1. Validar contrato e transições\n\nOs testes verificam configurações, fases, parâmetros treináveis e BatchNorm.",
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
        'tests/test_fine_tuning.py',
        'tests/test_deep_res_unet_resnet34.py',
    ],
    text=True,
    capture_output=True,
    env=env,
)
print(result.stdout)
print(result.stderr)
result.check_returncode()""",
    ),
    cell("markdown", "## 2. Ver a agenda congelada"),
    cell(
        "code",
        """import sys, yaml

sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from tcc_segmentation.fine_tuning import stage_for_epoch

base = yaml.safe_load((PROJECT_ROOT / 'configs/augmentation_d0_a1.yaml').read_text())
raw = yaml.safe_load((PROJECT_ROOT / 'configs/fine_tuning_progressive.yaml').read_text())
raw.pop('base_config')


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = (
            merge(out[k], v) if k in out and isinstance(out[k], dict) and isinstance(v, dict) else v
        )
    return out


cfg = merge(base, raw)
for epoch in (1, 15, 16, 30, 31, 100):
    print(epoch, stage_for_epoch(cfg, epoch))""",
    ),
    cell(
        "markdown",
        "## 3. DR-IF — encoder fixo\n\nTreina somente adaptador, decoder residual e attention gates.",
    ),
    cell(
        "code",
        """run_stream(
    [
        sys.executable,
        'scripts/run_baseline_d0.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--data-root',
        str(DATA_ROOT),
        '--config',
        'configs/fine_tuning_frozen_encoder.yaml',
    ]
)""",
    ),
    cell(
        "markdown",
        "## 4. DR-IFT — liberação progressiva\n\nNão interrompa entre fases. Early stopping só pode atuar a partir da época 31.",
    ),
    cell(
        "code",
        """run_stream(
    [
        sys.executable,
        'scripts/run_baseline_d0.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--data-root',
        str(DATA_ROOT),
        '--config',
        'configs/fine_tuning_progressive.yaml',
    ]
)""",
    ),
    cell(
        "markdown",
        "## 5. Artefatos da execução\n\nEnvie as novas pastas de execução em `results/day10/DRIF` e `results/day10/DRIFT`. A análise considera DR-IFT−DR-IF como contraste primário e DR-IFT−DR-IT como comparação secundária.",
    ),
]
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "14_fine_tuning.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = ROOT / "notebooks/14_fine_tuning.ipynb"
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
