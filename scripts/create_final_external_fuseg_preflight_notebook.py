#!/usr/bin/env python3
"""Gera o notebook didatico de preflight da avaliacao externa FUSeg."""

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
            '# Avaliação externa FUSeg — preflight\n'
            '\n'
            'Este notebook **não executa o DRIFT, não abre pixels de imagens ou máscaras e'
            ' não calcula métricas**.\n'
            '\n'
            'A verificação confere os hashes dos artefatos do protocolo e a presença dos '
            '1.010 pares de nomes. O lacre produzido é exigido pelo notebook de avaliação '
            'externa.'
        ),
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import json, os, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
PROTOCOL = PROJECT_ROOT / 'configs/final_external_fuseg_protocol_v1.yaml'
OUTPUT = PROJECT_ROOT / 'results/reproduction/external_fuseg_preflight.json'
%cd {PROJECT_ROOT}
!pip -q install pyyaml pytest
print('Projeto:', PROJECT_ROOT)
print('Protocolo:', PROTOCOL)
print('Nenhum pixel será aberto neste notebook.')""",
    ),
    cell(
        "markdown",
        """## 1. Localizar somente as pastas

O código procura as pastas do FUSeg em `/content/tcc_data` e no próprio projeto. Ele apenas verifica os nomes dos diretórios.""",
    ),
    cell(
        "code",
        """candidates = [Path('/content/tcc_data'), PROJECT_ROOT]
DATA_ROOT = next(
    (
        item
        for item in candidates
        if (item / 'Dataset/FUSeg_Images').is_dir() and (item / 'Dataset/FUSeg_Masks').is_dir()
    ),
    None,
)
if DATA_ROOT is None:
    raise FileNotFoundError(
        'Não encontrei Dataset/FUSeg_Images e Dataset/FUSeg_Masks. Copie o Dataset para o Drive ou /content/tcc_data.'
    )
print('Raiz estrutural encontrada:', DATA_ROOT)""",
    ),
    cell(
        "markdown",
        """## 2. Testar as regras congeladas

Os testes confirmam que DRIFT é o único modelo principal, DRIFT+SAM é secundário, DRIFTB está excluído e nenhum ajuste no FUSeg é permitido.""",
    ),
    cell(
        "code",
        """env = os.environ.copy()
env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_final_external_fuseg_protocol.py'],
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
        """## 3. Criar o lacre externo

O programa calcula hashes e lista os nomes `0001.png` a `1010.png`. Ele não decodifica nenhum arquivo científico.""",
    ),
    cell(
        "code",
        """command = [
    sys.executable,
    'scripts/audit_final_external_fuseg_protocol.py',
    '--project-root',
    str(PROJECT_ROOT),
    '--data-root',
    str(DATA_ROOT),
    '--protocol',
    str(PROTOCOL),
    '--output',
    str(OUTPUT),
]
result = subprocess.run(command, text=True, capture_output=True, env=env)
print(result.stdout)
print(result.stderr)
result.check_returncode()""",
    ),
    cell(
        "markdown",
        (
            '## 4. Conferência legível\n'
            '\n'
            'O relatório deve registrar `passed`, 1.010 imagens, 1.010 máscaras, 1.010 '
            'nomes pareados e ausência de inferência ou de leitura do conteúdo das imagens'
            ' e máscaras.'
        ),
    ),
    cell(
        "code",
        """report = json.loads(OUTPUT.read_text(encoding='utf-8'))
print('Status:', report['status'])
print('Conteúdo científico aberto:', report['scientific_file_contents_accessed'])
print('Inferência executada:', report['inference_executed'])
print('Imagens:', report['dataset_structure']['image_files'])
print('Máscaras:', report['dataset_structure']['mask_files'])
print('Nomes pareados:', report['dataset_structure']['paired_names'])
print('Modelo principal:', report['primary_model']['id'])
print('Análise secundária:', report['secondary_model']['id'])
print('Hash do protocolo:', report['protocol_sha256'])
print('Hash do manifesto:', report['manifest']['sha256'])""",
    ),
    cell(
        "markdown",
        """## 5. Baixar o lacre

Guarde o JSON com o resultado da verificação. **Não execute ainda nenhuma avaliação no FUSeg.**""",
    ),
    cell(
        "code",
        """from google.colab import files

files.download(str(OUTPUT))""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "23_external_fuseg_preflight.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/23_external_fuseg_preflight.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)


if __name__ == "__main__":
    pass
