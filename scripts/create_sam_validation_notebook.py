#!/usr/bin/env python3
"""Gera o notebook didático da avaliação completa do SAM na avaliação do SAM na validação."""

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
        "# Avaliação do SAM na validação congelada\n\n"
        "O experimento aplica as três modalidades verificadas na integração técnica às **294 imagens da "
        "validação**. Esta etapa seleciona o protocolo SAM que será congelado antes "
        "do teste, mas ainda não mede o efeito final.\n\n"
        "Fluxo: **DR-IFT congelada → prompts automáticos → SAM congelado → comparação "
        "pareada por imagem**. Teste e FUSeg continuam fechados.",
    ),
    cell(
        "markdown",
        "## O que já está decidido\n\n"
        "- DR-IFT seed 42, época 56 e threshold 0,18;\n"
        "- SAM 1 ViT-B sem fine-tuning;\n"
        "- modos `box`, `mask_input` e `box+mask`;\n"
        "- prompts produzidos somente pela DR-IFT;\n"
        "- Dice macro por imagem como métrica primária;\n"
        "- IC 95% bootstrap pareado e Holm nos três contrastes de Dice;\n"
        "- escolha pela maior média de Dice, com tolerância de 0,0001 e desempate "
        "conservador: box, mask e box+mask.\n\n"
        "A regra foi registrada antes desta execução para que o resultado não possa "
        "mudar o critério de escolha.",
    ),
    cell(
        "markdown",
        "## 1. Preparar o Colab\n\n"
        "O Dataset será copiado para o disco temporário apenas para acelerar a leitura. "
        "O projeto e as saídas permanecem no Drive.",
    ),
    cell(
        "code",
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "from pathlib import Path\n"
        "import json, os, shutil, subprocess, sys, urllib.request, yaml\n\n"
        "PROJECT_ROOT=Path('/content/drive/MyDrive/Projeto_TCC_Lucas')\n"
        "DATA_ROOT=Path('/content/tcc_data')\n"
        "%cd {PROJECT_ROOT}\n"
        "assert PROJECT_ROOT.is_dir(), f'Projeto ausente: {PROJECT_ROOT}'\n"
        "source=PROJECT_ROOT/'Dataset'\n"
        "assert source.is_dir(), f'Dataset ausente: {source}'\n"
        "shutil.copytree(source,DATA_ROOT/'Dataset',dirs_exist_ok=True)\n"
        "assert (DATA_ROOT/'Dataset/DFUC2022_train_images/100001.jpg').is_file()\n"
        "gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()\n"
        "print('Cache local pronto | GPU:',gpu)",
    ),
    cell(
        "markdown",
        "## 2. Instalar a versão congelada do SAM\n\n"
        "O código oficial permanece fixado no mesmo commit usado no smoke test. O "
        "checkpoint já existente no Drive será reutilizado.",
    ),
    cell(
        "code",
        "SAM_COMMIT='6fdee8f2727f4506cfbbe553e23b895e27956588'\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q',f'git+https://github.com/facebookresearch/segment-anything.git@{SAM_COMMIT}'],check=True)\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q','pyyaml','pytest','matplotlib'],check=True)\n"
        "sam_dir=PROJECT_ROOT/'models/sam'; sam_dir.mkdir(parents=True,exist_ok=True)\n"
        "sam_checkpoint=sam_dir/'sam_vit_b_01ec64.pth'\n"
        "sam_url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'\n"
        "if not sam_checkpoint.is_file():\n"
        "    print('Baixando checkpoint oficial ViT-B...')\n"
        "    urllib.request.urlretrieve(sam_url,sam_checkpoint)\n"
        "print('SAM:',sam_checkpoint,f'{sam_checkpoint.stat().st_size/1024**2:.1f} MiB')",
    ),
    cell(
        "markdown",
        "## 3. Conferir o protocolo antes da inferência\n\n"
        "Esta célula verifica os arquivos, os hashes dos manifests, as 294 imagens e "
        "a identidade do checkpoint. Se falhar, não substitua arquivos nem altere a regra.",
    ),
    cell(
        "code",
        "CONFIG=PROJECT_ROOT/'configs/sam_validation.yaml'\n"
        "cfg=yaml.safe_load(CONFIG.read_text())\n"
        "network_run=PROJECT_ROOT/cfg['network']['run_directory']\n"
        "network_checkpoint=network_run/cfg['network']['checkpoint_filename']\n"
        "required=[CONFIG,network_run/'config_used.yaml',network_checkpoint,sam_checkpoint,PROJECT_ROOT/cfg['data']['manifest'],PROJECT_ROOT/cfg['data']['fixed_visualization_manifest']]\n"
        "for path in required: assert path.is_file(), f'Arquivo ausente: {path}'\n"
        "import csv, hashlib\n"
        "def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()\n"
        "manifest=PROJECT_ROOT/cfg['data']['manifest']\n"
        "with manifest.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))\n"
        "assert len(rows)==294 and all(r['split']=='validation' for r in rows)\n"
        "assert sha(manifest)==cfg['data']['manifest_sha256']\n"
        "print('Protocolo:',cfg['stage'])\n"
        "print('DR-IFT:',cfg['network']['run_id'],'| época',cfg['network']['checkpoint_epoch'],'| threshold',cfg['network']['threshold'])\n"
        "print('Validação congelada:',len(rows),'imagens | hashes aprovados')",
    ),
    cell(
        "markdown",
        "## 4. Rodar os testes\n\n"
        "Os testes de integração verificam a interface dos prompts. Os testes de validação verificam "
        "a seleção, o pareamento, o Boundary IoU e o bloqueio das partições finais.",
    ),
    cell(
        "code",
        "env=os.environ.copy()\n"
        "env['PYTHONPATH']=str(PROJECT_ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')\n"
        "result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sam_refinement.py','tests/test_sam_validation.py'],text=True,capture_output=True,env=env)\n"
        "print(result.stdout); print(result.stderr); result.check_returncode()",
    ),
    cell(
        "markdown",
        "## 5. Executar as 294 imagens\n\n"
        "Cada imagem passa uma vez pela DR-IFT e pelos três modos do SAM. A saída mostra "
        "o progresso a cada dez imagens. Não interrompa enquanto não aparecer "
        "`status: completed`.",
    ),
    cell(
        "code",
        "def run_stream(command):\n"
        "    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
        "    for line in process.stdout: print(line,end='')\n"
        "    code=process.wait(); print(f'\\nreturncode={code}')\n"
        "    if code: raise subprocess.CalledProcessError(code,command)\n"
        "    return code\n\n"
        "run_stream([sys.executable,'scripts/run_sam_validation.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config','configs/sam_validation.yaml'])",
    ),
    cell(
        "markdown",
        "## 6. Ler o resultado quantitativo\n\n"
        "A primeira tabela mostra o desempenho absoluto. A segunda mostra quanto cada "
        "modo mudou o Dice da mesma imagem em relação à DR-IFT. O intervalo que cruza "
        "zero indica incerteza sobre a direção média na validação.",
    ),
    cell(
        "code",
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "runs=sorted((PROJECT_ROOT/'results/day13/SAM_validation').glob('SAM1_ViTB_DRIFT_seed42_validation_*'))\n"
        "assert runs, 'Nenhuma execução de validação do SAM encontrada'\n"
        "RUN=runs[-1]\n"
        "summary=pd.read_csv(RUN/'variant_summary.csv')\n"
        "effects=pd.read_csv(RUN/'paired_comparisons.csv')\n"
        "selection=json.loads((RUN/'selection.json').read_text())\n"
        "display(summary[['variant','images','dice_mean','iou_mean','precision_mean','recall_mean','boundary_iou_mean','zero_dice_images']])\n"
        "display(effects[effects.metric=='dice'][['comparison','mean_difference','ci95_low','ci95_high','improved','equal','worsened','permutation_p_holm']])\n"
        "print(json.dumps(selection,indent=2,ensure_ascii=False))",
    ),
    cell(
        "markdown",
        "## 7. Ver os resultados\n\n"
        "O primeiro gráfico resume desempenho e efeitos pareados. O primeiro painel "
        "mantém os 12 casos congelados e todos os modos; o segundo destaca apenas o "
        "modo selecionado. Verde é acerto, laranja é falso positivo e vermelho é falso negativo.",
    ),
    cell(
        "code",
        "from PIL import Image\n"
        "display(Image.open(RUN/'sam_validation_summary.png'))\n"
        "display(Image.open(RUN/'fixed_validation_panel_all_modes.png'))\n"
        "display(Image.open(RUN/'fixed_validation_panel_selected.png'))",
    ),
    cell(
        "markdown",
        "## 8. Empacotar a execução\n\n"
        "O ZIP contém métricas, auditoria, gráficos e painéis. Checkpoints e Dataset não "
        "são copiados.",
    ),
    cell(
        "code",
        "export_dir=PROJECT_ROOT/'exports/day13'; export_dir.mkdir(parents=True,exist_ok=True)\n"
        "archive=shutil.make_archive(str(export_dir/f'sam_validation_{RUN.name}'),'zip',RUN)\n"
        "print('Arquivo exportado:',archive)",
    ),
    cell(
        "markdown",
        "## Interpretação\n\n"
        "A modalidade de SAM é escolhida na validação, "
        "e seu efeito final é estimado no teste interno. "
        "A modalidade selecionada é fixada antes da avaliação final.",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "17_sam_validation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

path = ROOT / "notebooks" / "17_sam_validation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
