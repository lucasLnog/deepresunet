#!/usr/bin/env python3
"""Gera o notebook didático da ablação controlada MedSAM."""

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
        "# MedSAM como refinador da DR-IFT\n\n"
        "Esta é uma **ablação controlada na validação congelada**. Repetiremos DR-IFT e os três modos do SAM 1 e acrescentaremos somente o `MedSAM_box`. Assim, isolamos o efeito do pré-treinamento médico.\n\n"
        "A arquitetura central continua sendo a **DR-IFT**: ela produz a probabilidade e todas as caixas. O MedSAM não vê a referência. Teste interno e FUSeg permanecem fechados.",
    ),
    cell(
        "markdown",
        "## Hipótese e regra definida antes do resultado\n\n"
        "O MedSAM pode interpretar melhor imagens médicas que o SAM genérico, mas suas imagens de treinamento são muito diferentes de fotografias de úlceras do pé. Por isso ele é uma hipótese, não uma melhoria presumida.\n\n"
        "Ele só avançará se, contra a DR-IFT, obtiver simultaneamente: ganho médio de Dice ≥ **0,01** e limite inferior do IC 95% pareado > **0**. Esse 0,01 é um critério operacional do estudo, não um limiar clínico universal.",
    ),
    cell(
        "markdown",
        "## 1. Preparar o Colab\n\nO Dataset é copiado ao disco temporário para acelerar a leitura; projeto e resultados continuam no Drive.",
    ),
    cell(
        "code",
        "from google.colab import drive\ndrive.mount('/content/drive')\n"
        "from pathlib import Path\nimport csv, hashlib, json, os, shutil, subprocess, sys, urllib.request, yaml\n"
        "PROJECT_ROOT=Path('/content/drive/MyDrive/Projeto_TCC_Lucas')\nDATA_ROOT=Path('/content/tcc_data')\n%cd {PROJECT_ROOT}\n"
        "assert PROJECT_ROOT.is_dir(), f'Projeto ausente: {PROJECT_ROOT}'\nsource=PROJECT_ROOT/'Dataset'\n"
        "assert source.is_dir(), f'Dataset ausente: {source}'\nshutil.copytree(source,DATA_ROOT/'Dataset',dirs_exist_ok=True)\n"
        "assert (DATA_ROOT/'Dataset/DFUC2022_train_images/100001.jpg').is_file()\n"
        "gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()\nprint('Cache local pronto | GPU:',gpu)",
    ),
    cell(
        "markdown",
        "## 2. Instalar a implementação congelada\n\n"
        "SAM e MedSAM usam a mesma arquitetura ViT-B do pacote oficial do SAM. O código fica fixado no commit já auditado. `scikit-image` é necessário para reproduzir o redimensionamento oficial do MedSAM.",
    ),
    cell(
        "code",
        "SAM_COMMIT='6fdee8f2727f4506cfbbe553e23b895e27956588'\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q',f'git+https://github.com/facebookresearch/segment-anything.git@{SAM_COMMIT}'],check=True)\n"
        "subprocess.run([sys.executable,'-m','pip','install','-q','pyyaml','pytest','matplotlib','scikit-image'],check=True)\nprint('Dependências prontas')",
    ),
    cell(
        "markdown",
        "## 3. Obter e verificar os dois checkpoints\n\n"
        "Os pesos ficam em cache no Drive. Mesmo se já existirem, calculamos SHA-256: um arquivo incompleto ou diferente interrompe a execução.",
    ),
    cell(
        "code",
        "def download_if_missing(url,path):\n    path.parent.mkdir(parents=True,exist_ok=True)\n    if not path.is_file():\n        print('Baixando',path.name,'...')\n        urllib.request.urlretrieve(url,path)\n\ndef sha(path):\n    digest=hashlib.sha256()\n    with path.open('rb') as stream:\n        for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)\n    return digest.hexdigest()\n\n"
        "sam=PROJECT_ROOT/'models/sam/sam_vit_b_01ec64.pth'\nmedsam=PROJECT_ROOT/'models/medsam/medsam_vit_b.pth'\n"
        "download_if_missing('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',sam)\n"
        "download_if_missing('https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1',medsam)\n"
        "expected={'sam_vit_b_01ec64.pth':'ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912','medsam_vit_b.pth':'34b34b78c1d18cb8c6bf84cf9c00e135d6d6c965699f3c0e31ef1bc9dcb5be74'}\n"
        "for path in (sam,medsam):\n    actual=sha(path); assert actual==expected[path.name],f'Hash inválido: {path}'\n    print(path.name,f'{path.stat().st_size/1024**2:.1f} MiB','| hash aprovado')",
    ),
    cell(
        "markdown",
        "## 4. Auditar o protocolo\n\n"
        "Aqui conferimos DR-IFT seed 42/época 56/threshold 0,18, as 294 imagens e os manifests congelados. Nenhum threshold será procurado novamente.",
    ),
    cell(
        "code",
        "CONFIG=PROJECT_ROOT/'configs/medsam_validation.yaml'\ncfg=yaml.safe_load(CONFIG.read_text())\n"
        "manifest=PROJECT_ROOT/cfg['data']['manifest']; fixed=PROJECT_ROOT/cfg['data']['fixed_visualization_manifest']\n"
        "with manifest.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))\n"
        "assert len(rows)==294 and all(r['split']=='validation' for r in rows)\nassert sha(manifest)==cfg['data']['manifest_sha256']\nassert sha(fixed)==cfg['data']['fixed_visualization_manifest_sha256']\n"
        "assert cfg['network']['threshold']==0.18 and cfg['medsam']['prompt_mode']=='box'\n"
        "print('DR-IFT:',cfg['network']['run_id'],'| época',cfg['network']['checkpoint_epoch'],'| threshold',cfg['network']['threshold'])\n"
        "print('Comparação:',cfg['evaluation']['variants'])\nprint('Validação congelada:',len(rows),'| teste/FUSeg: fechados')",
    ),
    cell(
        "markdown",
        "## 5. Executar os testes\n\nOs testes impedem mudança acidental das caixas, variantes, estatística ou partições.",
    ),
    cell(
        "code",
        "env=os.environ.copy(); env['PYTHONPATH']=str(PROJECT_ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')\n"
        "result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sam_refinement.py','tests/test_sam_validation.py','tests/test_medsam_validation.py'],text=True,capture_output=True,env=env)\n"
        "print(result.stdout); print(result.stderr); result.check_returncode()",
    ),
    cell(
        "markdown",
        (
            '## 6. Executar a ablação\n'
            '\n'
            'Cada imagem passa uma vez pela DR-IFT. Depois, as mesmas caixas alimentam SAM'
            ' e MedSAM. O MedSAM usa o fluxo oficial: quadrado 1024, min–max, box em '
            'coordenadas 1024 e saída >0,5. O relatório registra `status: completed` ao '
            'final da execução.'
        ),
    ),
    cell(
        "code",
        "def run_stream(command):\n    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n    for line in process.stdout: print(line,end='')\n    code=process.wait(); print(f'\\nreturncode={code}')\n    if code: raise subprocess.CalledProcessError(code,command)\n\n"
        "run_stream([sys.executable,'scripts/run_medsam_validation.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config','configs/medsam_validation.yaml'])",
    ),
    cell(
        "markdown",
        "## 7. Ler números e decisão\n\n"
        "O desempenho absoluto não basta: a tabela pareada mede a mudança na **mesma imagem**. O IC 95% mostra a incerteza e Holm controla os quatro contrastes SAM-versus-DR-IFT.",
    ),
    cell(
        "code",
        "import pandas as pd\nfrom IPython.display import display\n"
        "runs=sorted((PROJECT_ROOT/'results/day13b/MedSAM_validation').glob('MedSAM_ViTB_DRIFT_seed42_validation_*'))\nassert runs,'Execução ausente'\nRUN=runs[-1]\n"
        "summary=pd.read_csv(RUN/'variant_summary.csv'); effects=pd.read_csv(RUN/'paired_comparisons.csv'); decision=json.loads((RUN/'advance_decision.json').read_text())\n"
        "display(summary[['variant','images','dice_mean','iou_mean','precision_mean','recall_mean','boundary_iou_mean','zero_dice_images','mean_postprocess_seconds']])\n"
        "display(effects[(effects.metric=='dice')][['comparison','mean_difference','ci95_low','ci95_high','improved','equal','worsened','permutation_p_holm']])\nprint(json.dumps(decision,indent=2,ensure_ascii=False))",
    ),
    cell(
        "markdown",
        "## 8. Ver os 12 casos congelados\n\n"
        "O painel completo permite comparar todos os métodos. No mapa de erros: verde = interseção, laranja = falso positivo e vermelho = falso negativo.",
    ),
    cell(
        "code",
        """from PIL import Image

display(Image.open(RUN / 'medsam_validation_summary.png'))
display(Image.open(RUN / 'fixed_validation_panel_all_models.png'))
display(Image.open(RUN / 'fixed_validation_panel_medsam.png'))""",
    ),
    cell(
        "markdown",
        "## 9. Empacotar para auditoria\n\nO ZIP contém protocolo, métricas por imagem, incerteza, auditoria dos prompts e painéis; não duplica Dataset ou checkpoints.",
    ),
    cell(
        "code",
        """export = PROJECT_ROOT / 'exports/day13b'
export.mkdir(parents=True, exist_ok=True)
archive = shutil.make_archive(str(export / f'medsam_validation_{RUN.name}'), 'zip', RUN)
print('Arquivo exportado:', archive)""",
    ),
    cell(
        "markdown",
        "## Interpretação\n\n"
        "A análise avalia se o MedSAM, condicionado à saída da DR-IFT, atende ao critério pré-registrado para avançar às avaliações finais.",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "18_medsam_validation.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = ROOT / "notebooks" / "18_medsam_validation.ipynb"
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(path)
