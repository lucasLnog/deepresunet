#!/usr/bin/env python3
"""Gera o notebook didático da confirmação multiseed da confirmação com múltiplas sementes."""

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


SETUP = """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import os, shutil, subprocess, sys, yaml

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
%cd {PROJECT_ROOT}
!pip -q install albumentations pyyaml pytest

assert PROJECT_ROOT.is_dir(), f'Projeto ausente: {PROJECT_ROOT}'
source = PROJECT_ROOT / 'Dataset'
assert source.is_dir(), f'Dataset ausente: {source}'
shutil.copytree(source, DATA_ROOT / 'Dataset', dirs_exist_ok=True)
assert (DATA_ROOT / 'Dataset/DFUC2022_train_images/100001.jpg').is_file()
print('Cache preparado:', DATA_ROOT / 'Dataset')
print(
    'GPU:',
    subprocess.run(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True
    ).stdout.strip(),
)"""


STREAM = """def run_stream(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        print(line, end='')
    code = process.wait()
    print(f'\\nreturncode={code}')
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_arm(config_path):
    run_stream(
        [
            sys.executable,
            'scripts/run_baseline_d0.py',
            '--project-root',
            str(PROJECT_ROOT),
            '--data-root',
            str(DATA_ROOT),
            '--config',
            config_path,
        ]
    )"""


cells = [
    cell(
        "markdown",
        "# Confirmação multiseed de DR-IF versus DR-IFT\n\n"
        "Repetiremos os dois braços nas seeds 1337 e 2026. A seed 42 já foi "
        "concluída. Dentro de cada seed, dados e aleatoriedade são pareados; entre "
        "seeds, eles variam. Teste interno e FUSeg continuam bloqueados.",
    ),
    cell("markdown", "## 1. Preparar Colab e cache local"),
    cell("code", SETUP),
    cell("code", STREAM),
    cell(
        "markdown",
        "## 2. Validar o contrato\n\n"
        "Esta célula verifica arquitetura, seeds, streams de augmentation, fases "
        "progressivas e a igualdade dos controles dentro de cada par.",
    ),
    cell(
        "code",
        "env = os.environ.copy()\n"
        "env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')\n"
        "result = subprocess.run([\n"
        "    sys.executable, '-m', 'pytest', '-q',\n"
        "    'tests/test_fine_tuning.py',\n"
        "    'tests/test_multiseed_confirmation.py',\n"
        "    'tests/test_deep_res_unet_resnet34.py',\n"
        "], text=True, capture_output=True, env=env)\n"
        "print(result.stdout); print(result.stderr); result.check_returncode()",
    ),
    cell(
        "markdown",
        "## 3. Seed 1337 — DR-IF\n\n"
        "Encoder fixo. Estimativa: aproximadamente 1,5–2 horas na mesma classe de GPU.",
    ),
    cell("code", "run_arm('configs/fine_tuning_frozen_encoder_seed1337.yaml')"),
    cell(
        "markdown",
        "## 4. Seed 1337 — DR-IFT\n\n"
        "Mesmo fluxo aleatório do braço anterior; somente a política de liberação muda.",
    ),
    cell("code", "run_arm('configs/fine_tuning_progressive_seed1337.yaml')"),
    cell(
        "markdown",
        "## 5. Seed 2026 — DR-IF\n\n"
        "Nova inicialização e nova sequência de augmentation, compartilhadas com seu par.",
    ),
    cell("code", "run_arm('configs/fine_tuning_frozen_encoder_seed2026.yaml')"),
    cell("markdown", "## 6. Seed 2026 — DR-IFT"),
    cell("code", "run_arm('configs/fine_tuning_progressive_seed2026.yaml')"),
    cell(
        "markdown",
        "## 7. Conferir e empacotar os quatro resultados\n\n"
        "A célula exige uma execução completa de cada braço e gera quatro ZIPs para envio.",
    ),
    cell(
        "code",
        "import json\n"
        "matrix = [\n"
        "    ('seed1337', 'DRIF'), ('seed1337', 'DRIFT'),\n"
        "    ('seed2026', 'DRIF'), ('seed2026', 'DRIFT'),\n"
        "]\n"
        "export_dir = PROJECT_ROOT / 'exports/day11'\n"
        "export_dir.mkdir(parents=True, exist_ok=True)\n"
        "for seed_name, arm in matrix:\n"
        "    root = PROJECT_ROOT / 'results/day11' / seed_name / arm\n"
        "    runs = sorted([p for p in root.glob('*') if p.is_dir()])\n"
        "    assert runs, f'Nenhuma execução: {root}'\n"
        "    run = runs[-1]\n"
        "    summary = json.loads((run / 'final_summary.json').read_text())\n"
        "    assert summary['status'] == 'completed'\n"
        "    assert not summary['protected_partitions_accessed']\n"
        "    destination = export_dir / f'{arm}_{seed_name}'\n"
        "    archive = shutil.make_archive(str(destination), 'zip', run)\n"
        "    print(archive)",
    ),
    cell(
        "markdown",
        "## 8. Artefatos da execução\n\n"
        "Arquive os quatro ZIPs de `exports/day11`. Não execute teste nem FUSeg. "
        "A síntese reúne as sementes 42, 1337 e 2026 sem tratar as 882 observações "
        "imagem-seed como independentes.",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {
            "name": "15_multiseed_confirmation.ipynb",
            "provenance": [],
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks" / "15_multiseed_confirmation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
