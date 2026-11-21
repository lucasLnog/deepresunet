#!/usr/bin/env python3
"""Gera o notebook da avaliação interna final, única e bloqueada."""

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
        """# Avaliação final — teste interno DFUC2022

Esta é a abertura única das **294 unidades canônicas do teste interno**.

O notebook não treina, não procura threshold e não escolhe o melhor modelo. Ele exige o protocolo e o preflight já lacrados, executa todos os braços antes de mostrar métricas e responde às comparações que foram definidas previamente.

Execute as células em ordem. A célula marcada como **ABERTURA DO TESTE** é o ponto irreversível do estudo.""",
    ),
    cell(
        "code",
        """from google.colab import drive

drive.mount('/content/drive')
from pathlib import Path
import json, os, shutil, subprocess, sys

PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')
DATA_ROOT = Path('/content/tcc_data')
PROTOCOL = PROJECT_ROOT / 'configs/final_internal_test_protocol_v1.yaml'
LOCK = PROJECT_ROOT / 'results/reproduction/internal_dfuc2022_preflight.json'
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
        """## 1. Testes do mecanismo de proteção

Esses testes não abrem o manifesto. Eles verificam a confirmação explícita, o pareamento por imagem, a correção de Holm, a seleção visual independente do desempenho e a recusa de arquivos alterados.""",
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
        'tests/test_final_internal_test_protocol.py',
        'tests/test_final_internal_evaluation.py',
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
        """## 2. Reconferir o lacre sem abrir o teste

Se qualquer configuração ou checkpoint tiver mudado depois do preflight, esta célula falhará.""",
    ),
    cell(
        "code",
        """from tcc_segmentation.final_evaluation import verify_preflight_lock

protocol, lock = verify_preflight_lock(PROJECT_ROOT, PROTOCOL, LOCK)
print('Lacre válido.')
print('Braços:', [arm['id'] for arm in protocol['arms']])
print('Contrastes primários:', [item['id'] for item in protocol['primary_contrasts']])
print('Thresholds congelados:', {arm['id']: arm['threshold'] for arm in protocol['arms']})""",
    ),
    cell(
        "markdown",
        """## 3. ABERTURA DO TESTE — execução única

Ao executar a próxima célula:

1. o Dataset será copiado para o cache local;
2. o manifesto de teste será aberto;
3. os nove checkpoints e o braço derivado DRIFT+SAM serão avaliados;
4. somente depois de todos terminarem serão gravados e exibidos os resumos.

Não interrompa para inspecionar resultados parciais. Se houver falha técnica, preserve a mensagem completa e não altere parâmetros.""",
    ),
    cell(
        "code",
        """source = PROJECT_ROOT / 'Dataset'
assert source.is_dir(), f'Dataset ausente: {source}'
shutil.copytree(source, DATA_ROOT / 'Dataset', dirs_exist_ok=True)


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
        'scripts/run_final_internal_test.py',
        '--project-root',
        str(PROJECT_ROOT),
        '--data-root',
        str(DATA_ROOT),
        '--protocol',
        str(PROTOCOL),
        '--preflight-lock',
        str(LOCK),
        '--confirm',
        'YES_I_UNDERSTAND_THIS_OPENS_THE_INTERNAL_TEST',
    ]
)""",
    ),
    cell(
        "markdown",
        """## 4. Resultado completo

Esta seção só é alcançada depois que todos os braços terminam. O primeiro quadro mostra desempenho absoluto; o segundo mostra os efeitos pareados congelados. Para os cinco contrastes primários de Dice, use `permutation_p_holm`, não apenas o valor bruto.""",
    ),
    cell(
        "code",
        """import pandas as pd
from IPython.display import display, Image as DisplayImage

runs = sorted((PROJECT_ROOT / 'results/final_evaluation').glob('FINAL_INTERNAL_TEST_*'))
complete = [run for run in runs if (run / 'final_test_report.json').is_file()]
assert complete, 'Execução final completa não encontrada'
RUN = complete[-1]
summary = pd.read_csv(RUN / 'test_summary_all_arms.csv')
comparisons = pd.read_csv(RUN / 'paired_comparisons.csv')
display(
    summary[
        [
            'arm_id',
            'threshold_policy',
            'images',
            'dice_mean',
            'iou_mean',
            'precision_mean',
            'recall_mean',
            'boundary_iou_mean',
        ]
    ].round(6)
)
display(
    comparisons[
        (comparisons['metric'] == 'dice') & (comparisons['threshold_policy'] == 'selected')
    ][
        [
            'family',
            'contrast_id',
            'candidate',
            'control',
            'mean_difference',
            'ci95_low',
            'ci95_high',
            'permutation_p_raw',
            'permutation_p_holm',
        ]
    ].round(
        6
    )
)
print('Execução:', RUN.name)""",
    ),
    cell(
        "markdown",
        """## 5. Painel visual sem cherry-picking

Os 12 casos foram escolhidos por espaçamento uniforme entre IDs ordenados, antes de consultar qualquer desempenho. Vermelho é referência e verde é predição.""",
    ),
    cell(
        "code",
        """display(DisplayImage(filename=str(RUN / 'fixed_test_panel_performance_blind.png')))""",
    ),
    cell(
        "markdown",
        """## 6. Baixar o pacote

O ZIP contém métricas por imagem, resumos, comparações, painel e o relatório técnico; não contém Dataset nem checkpoints.""",
    ),
    cell(
        "code",
        """archive = shutil.make_archive('/content/FINAL_INTERNAL_TEST_RESULTS', 'zip', root_dir=RUN)
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
        "colab": {"name": "22_internal_dfuc2022_evaluation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks/22_internal_dfuc2022_evaluation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
