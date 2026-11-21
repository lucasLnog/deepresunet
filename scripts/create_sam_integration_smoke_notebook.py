#!/usr/bin/env python3
"""Gera o notebook didático da integração técnica do SAM na integração técnica do SAM."""

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
        "# Integração técnica do SAM 1 ViT-B\n\n"
        "Hoje verificaremos se o SAM consegue receber prompts produzidos somente pela "
        "DR-IFT. Este é um teste técnico em quatro casos fixos da validação; ele não "
        "escolhe o melhor modo e não acessa teste ou FUSeg.\n\n"
        "Fluxo: **imagem → DR-IFT → componentes/box/logits → SAM → nova máscara**.",
    ),
    cell(
        "markdown",
        "## 1. Preparar o Colab\n\n"
        "A cópia do Dataset para o disco temporário acelera a leitura. O projeto e os "
        "resultados permanecem no Drive.",
    ),
    cell(
        "code",
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "from pathlib import Path\n"
        "import json, os, shutil, subprocess, sys, urllib.request, yaml\n\n"
        "PROJECT_ROOT = Path('/content/drive/MyDrive/Projeto_TCC_Lucas')\n"
        "DATA_ROOT = Path('/content/tcc_data')\n"
        "%cd {PROJECT_ROOT}\n"
        "assert PROJECT_ROOT.is_dir(), f'Projeto ausente: {PROJECT_ROOT}'\n"
        "source = PROJECT_ROOT / 'Dataset'\n"
        "assert source.is_dir(), f'Dataset ausente: {source}'\n"
        "shutil.copytree(source, DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
        "assert (DATA_ROOT/'Dataset/DFUC2022_train_images/100001.jpg').is_file()\n"
        "print('Cache local pronto')\n"
        "print('GPU:', subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], capture_output=True, text=True).stdout.strip())",
    ),
    cell(
        "markdown",
        "## 2. Instalar a versão congelada do SAM\n\n"
        "Usamos o repositório oficial fixado em um commit, e não uma versão móvel. O "
        "checkpoint ViT-B é baixado uma única vez para `models/sam` no Drive.",
    ),
    cell(
        "code",
        "SAM_COMMIT='6fdee8f2727f4506cfbbe553e23b895e27956588'\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q',"
        "f'git+https://github.com/facebookresearch/segment-anything.git@{SAM_COMMIT}'],check=True)\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q','pyyaml','pytest'],check=True)\n"
        "sam_dir=PROJECT_ROOT/'models/sam'; sam_dir.mkdir(parents=True,exist_ok=True)\n"
        "sam_checkpoint=sam_dir/'sam_vit_b_01ec64.pth'\n"
        "sam_url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'\n"
        "if not sam_checkpoint.is_file():\n"
        "    print('Baixando checkpoint oficial ViT-B...')\n"
        "    urllib.request.urlretrieve(sam_url,sam_checkpoint)\n"
        "print('SAM:',sam_checkpoint, f'{sam_checkpoint.stat().st_size/1024**2:.1f} MiB')",
    ),
    cell(
        "markdown",
        "## 3. Conferir os arquivos congelados\n\n"
        "A DR-IFT deve ser a execução da seed 42, melhor época 56 e threshold 0,18. "
        "Se algum item estiver ausente, pare aqui: não substitua por outra seed.",
    ),
    cell(
        "code",
        "CONFIG=PROJECT_ROOT/'configs/sam_vit_b_integration.yaml'\n"
        "cfg=yaml.safe_load(CONFIG.read_text())\n"
        "network_run=PROJECT_ROOT/cfg['network']['run_directory']\n"
        "network_checkpoint=network_run/cfg['network']['checkpoint_filename']\n"
        "required=[CONFIG,network_run/'config_used.yaml',network_checkpoint,sam_checkpoint,PROJECT_ROOT/cfg['data']['manifest']]\n"
        "for path in required:\n"
        "    assert path.is_file(), f'Arquivo ausente: {path}'\n"
        "print('DR-IFT:',cfg['network']['run_id'],'| época',cfg['network']['checkpoint_epoch'],'| threshold',cfg['network']['threshold'])\n"
        "print('Todos os arquivos obrigatórios foram encontrados.')",
    ),
    cell(
        "markdown",
        "## 4. Testar o contrato antes da GPU\n\n"
        "Os testes verificam versão, threshold, bloqueio do teste, componentes múltiplos, "
        "máscara vazia e os argumentos entregues ao SAM.",
    ),
    cell(
        "code",
        "env=os.environ.copy()\n"
        "env['PYTHONPATH']=str(PROJECT_ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')\n"
        "result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sam_refinement.py'],text=True,capture_output=True,env=env)\n"
        "print(result.stdout); print(result.stderr); result.check_returncode()",
    ),
    cell(
        "markdown",
        (
            '## 5. Executar o smoke test\n'
            '\n'
            'Para cada componente previsto, foram avaliadas três modalidades: caixa, '
            '`mask_input` e caixa com `mask_input`. O `mask_input` preserva o aspecto e o '
            'padding inferior/direito usados pelo SAM. A referência só é consultada depois'
            ' da predição, para métricas e painel.'
        ),
    ),
    cell(
        "code",
        "command=[sys.executable,'scripts/run_sam_integration_smoke.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config',str(CONFIG)]\n"
        "process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "for line in process.stdout: print(line,end='')\n"
        "code=process.wait(); print(f'\\nreturncode={code}')\n"
        "if code: raise subprocess.CalledProcessError(code,command)",
    ),
    cell(
        "markdown",
        (
            '## 6. Ver o resultado\n'
            '\n'
            'No mapa de erros: verde é acerto, laranja é falso positivo e vermelho é falso'
            ' negativo. Os quatro exemplos são usados na verificação técnica. A seleção da'
            ' modalidade usa a partição completa de validação.'
        ),
    ),
    cell(
        "code",
        "from IPython.display import display\n"
        "from PIL import Image\n"
        "runs=sorted((PROJECT_ROOT/'results/day12/smoke').glob('SAM1_ViTB_DRIFT_seed42_*'))\n"
        "assert runs, 'Nenhuma execução de integração técnica do SAM encontrada'\n"
        "RUN=runs[-1]\n"
        "report=json.loads((RUN/'technical_report.json').read_text())\n"
        "print(json.dumps(report,indent=2,ensure_ascii=False))\n"
        "display(Image.open(RUN/'sam_smoke_panel.png'))",
    ),
    cell(
        "markdown",
        "## 7. Empacotar o que será analisado\n\n"
        "O ZIP contém somente relatórios, métricas e painel do smoke test. Não precisa "
        "copiar o checkpoint SAM nem o checkpoint da rede.",
    ),
    cell(
        "code",
        "export_dir=PROJECT_ROOT/'exports/day12'; export_dir.mkdir(parents=True,exist_ok=True)\n"
        "archive=shutil.make_archive(str(export_dir/f'sam_integration_{RUN.name}'),'zip',RUN)\n"
        "print('Arquivo exportado:',archive)",
    ),
    cell(
        "markdown",
        "## Interpretação\n\n"
        "O SAM recebeu apenas localização e logits produzidos "
        "pela DR-IFT. A referência foi usada somente para avaliação. "
        "A etapa seguinte aplica o procedimento às 294 imagens de validação.",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "16_sam_integration_smoke.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks" / "16_sam_integration_smoke.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
