#!/usr/bin/env python3
"""Gera o notebook da avaliacao externa final e bloqueada no FUSeg."""

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
        """# Avaliação externa final — FUSeg

Esta é a abertura única das **962 unidades canônicas externas**. O modelo principal é o DRIFT congelado; DRIFT+SAM é uma análise secundária previamente definida.

O notebook não treina, não procura threshold, não escolhe checkpoint e não escolhe modelo. As referências duplicadas são avaliadas separadamente, suas métricas são promediadas dentro da unidade e só então cada imagem única recebe peso 1.

Execute as células em ordem. A célula **ABERTURA DO FUSEG** é o ponto irreversível desta avaliação.""",
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import json, os, shutil, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
PROTOCOL = PROJECT_ROOT / 'configs/final_external_fuseg_protocol_v1.yaml'
LOCK = PROJECT_ROOT / 'results/reproduction/external_fuseg_preflight.json'
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
%cd {PROJECT_ROOT}
!pip -q install pyyaml pytest
!pip -q install "git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588"
print('Protocolo:', PROTOCOL)
print('Lacre:', LOCK)
print('Código-fonte ativo:', SRC_ROOT)""",
    ),
    cell(
        "markdown",
        """## 1. Testes de proteção

Eles verificam o manifesto canônico, a política de múltiplas referências, o bootstrap e a ausência de treino ou busca de threshold no runner.""",
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
        'tests/test_final_external_fuseg_protocol.py',
        'tests/test_final_external_fuseg_evaluation.py',
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
        """## 2. Reconferir o lacre sem abrir resultados

Esta célula confirma que protocolo, manifesto, exclusões, auditoria, configuração e pesos continuam idênticos aos selados.""",
    ),
    cell(
        "code",
        """from tcc_segmentation.external_evaluation import verify_external_preflight_lock

protocol, lock = verify_external_preflight_lock(PROJECT_ROOT, PROTOCOL, LOCK)
print('Lacre externo válido.')
print('Modelo principal:', protocol['primary_model']['id'], protocol['primary_model']['run_id'])
print('Threshold congelado:', protocol['primary_model']['threshold'])
print('Unidades canônicas esperadas:', protocol['data']['expected_canonical_units'])
print('Política de referências:', protocol['data']['duplicate_reference_policy'])""",
    ),
    cell(
        "markdown",
        """## 3. ABERTURA DO FUSEG — execução única

Ao executar a próxima célula, os arquivos FUSeg serão copiados para o cache local e as 962 unidades serão processadas em lote. Nenhum resumo é calculado antes de todas as predições terminarem.

Não interrompa para inspecionar resultados parciais. Se ocorrer falha técnica, preserve a mensagem completa e não altere parâmetros científicos.""",
    ),
    cell(
        "code",
        """for folder in ('FUSeg_Images', 'FUSeg_Masks'):
    source = PROJECT_ROOT / 'Dataset' / folder
    destination = DATA_ROOT / 'Dataset' / folder
    assert source.is_dir(), f'Pasta ausente: {source}'
    shutil.copytree(source, destination, dirs_exist_ok=True)


def run_stream(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
    )
    for line in process.stdout:
        print(line, end='')
    code = process.wait()
    print(f'\\nreturncode={code}')
    if code:
        raise subprocess.CalledProcessError(code, command)


run_stream(
    [
        sys.executable,
        'scripts/run_final_external_fuseg.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--data-root',
        str(DATA_ROOT),
        '--protocol',
        str(PROTOCOL),
        '--preflight-lock',
        str(LOCK),
        '--confirm',
        'YES_I_UNDERSTAND_THIS_OPENS_FUSEG_EXTERNAL_EVALUATION',
    ]
)""",
    ),
    cell(
        "markdown",
        """## 4. Resultados externos

O primeiro quadro contém desempenho absoluto e IC95% por bootstrap. O segundo mostra o contraste pareado secundário DRIFT+SAM menos DRIFT. O resultado principal é o Dice macro do DRIFT em threshold 0,18.""",
    ),
    cell(
        "code",
        """import pandas as pd
from IPython.display import display, Image as DisplayImage

runs = sorted((PROJECT_ROOT / 'results/final_external_fuseg').glob('FINAL_EXTERNAL_FUSEG_*'))
complete = [run for run in runs if (run / 'final_external_report.json').is_file()]
assert complete, 'Execução externa completa não encontrada'
RUN = complete[-1]
summary = pd.read_csv(RUN / 'external_summary.csv')
comparison = pd.read_csv(RUN / 'external_sam_comparison.csv')
display(
    summary[
        [
            'arm_id',
            'threshold_policy',
            'images',
            'dice_mean',
            'dice_ci95_low',
            'dice_ci95_high',
            'iou_mean',
            'precision_mean',
            'recall_mean',
            'boundary_iou_mean',
        ]
    ].round(6)
)
display(
    comparison[['metric', 'mean_difference', 'ci95_low', 'ci95_high', 'permutation_p_raw']].round(6)
)
print('Execução:', RUN.name)""",
    ),
    cell(
        "markdown",
        """## 5. Painel visual sem cherry-picking

Os 12 IDs foram escolhidos por espaçamento uniforme na ordem lexical antes de qualquer desempenho ser conhecido. Em unidades com múltiplas referências, a intensidade vermelha representa a média das referências.""",
    ),
    cell(
        "code",
        """display(DisplayImage(filename=str(RUN / 'fixed_external_panel_performance_blind.png')))""",
    ),
    cell(
        "markdown",
        (
            '## 6. Exportação do pacote\n'
            '\n'
            'O ZIP contém métricas por unidade, intervalos, comparação SAM, auditoria de '
            'prompts, painel e relatório técnico; não contém o FUSeg nem checkpoints.'
        ),
    ),
    cell(
        "code",
        """archive = shutil.make_archive('/content/FINAL_EXTERNAL_FUSEG_RESULTS', 'zip', root_dir=RUN)
print('Pacote pronto:', archive)
from google.colab import files

files.download(archive)""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "24_external_fuseg_evaluation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/24_external_fuseg_evaluation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
