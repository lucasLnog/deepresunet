#!/usr/bin/env python3
"""Gera o notebook de auditoria anterior ao teste interno final."""

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
            '# Etapa final — preflight antes do teste interno\n'
            '\n'
            'Este notebook **não executa inferência e não abre imagens ou máscaras do '
            'teste**.\n'
            '\n'
            'A verificação confere os hashes do protocolo, do manifesto, das '
            'configurações, dos checkpoints e dos pesos do SAM. O lacre produzido é '
            'exigido pelo notebook de avaliação interna.'
        ),
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import json, os, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
PROTOCOL = PROJECT_ROOT / 'configs/final_internal_test_protocol_v1.yaml'
OUTPUT = PROJECT_ROOT / 'results/reproduction/internal_dfuc2022_preflight.json'
%cd {PROJECT_ROOT}
!pip -q install pyyaml pytest
print('Projeto:', PROJECT_ROOT)
print('Protocolo:', PROTOCOL)
print('Nenhuma imagem de teste será aberta neste notebook.')""",
    ),
    cell(
        "markdown",
        """## 1. Conferir as regras congeladas

Os testes abaixo verificam a matriz de dez saídas, a seed 42, os cinco contrastes primários, a separação da DRIFTB exploratória e a proibição de escolher algo pelo teste.""",
    ),
    cell(
        "code",
        """env = os.environ.copy()
env['PYTHONPATH'] = str(PROJECT_ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_final_internal_test_protocol.py'],
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
        """## 2. Selar os arquivos

O programa calcula hashes dos arquivos. Para o manifesto de teste, somente os bytes do próprio CSV são usados para conferir integridade; suas linhas não são lidas e nenhuma imagem ou máscara é carregada.""",
    ),
    cell(
        "code",
        """command = [
    sys.executable,
    'scripts/audit_final_internal_test_protocol.py',
    '--project-root',
    str(PROJECT_ROOT),
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
        """## 3. Resumo legível

A verificação deve indicar `passed`, dez braços, `scientific_data_accessed = false` e `inference_executed = false`. Se faltar um checkpoint, confira o caminho da execução registrado no protocolo.""",
    ),
    cell(
        "code",
        """report = json.loads(OUTPUT.read_text(encoding='utf-8'))
print('Status:', report['status'])
print('Dados científicos acessados:', report['scientific_data_accessed'])
print('Inferência executada:', report['inference_executed'])
print('Hash do protocolo:', report['protocol_sha256'])
print('Hash do manifesto:', report['test_manifest']['sha256'])
print('\\nArquivos selados:')
for arm in report['arms']:
    key = 'checkpoint_sha256' if 'checkpoint_sha256' in arm else 'sam_checkpoint_sha256'
    print(f"- {arm['id']}: {arm[key][:16]}…")""",
    ),
    cell(
        "markdown",
        (
            '## 4. Arquivo de verificação preliminar\n'
            '\n'
            'O arquivo JSON registra os hashes conferidos. A avaliação interna usa o mesmo'
            ' protocolo e exige esses hashes.'
        ),
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
        "colab": {"name": "21_internal_dfuc2022_preflight.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/21_internal_dfuc2022_preflight.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
