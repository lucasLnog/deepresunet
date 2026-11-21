#!/usr/bin/env python3
"""Gera o notebook didático dos treinamentos D0-A1 e D0-A2."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def main() -> int:
    cells = [
        markdown(
            "# Treinamento controlado com data augmentation\n\n"
            "Este notebook compara `D0-A1` (geometria) e `D0-A2` (geometria + uma "
            "alteração fotométrica leve) contra o D0 já congelado. A DeepResUNet, os "
            "splits, a seed, a loss e todas as regras de avaliação permanecem iguais."
        ),
        markdown(
            "## O que está sendo testado\n\n"
            "Cada época contém as mesmas 1.373 unidades canônicas, mas o treino recebe "
            "novos sorteios de transformações. A validação nunca recebe augmentation. "
            "A1/A2 começam do zero; nenhum checkpoint anterior é reutilizado."
        ),
        markdown(
            "## Regra de segurança\n\n"
            "O alerta de retenção de 50% não filtra o treino. Somente uma transformação "
            "que torne vazia uma máscara originalmente positiva é repetida. Há no máximo "
            "três novos sorteios; depois disso, usa-se a amostra sem augmentation. Tudo é registrado."
        ),
        markdown("## 1. Montar o Drive e localizar o projeto"),
        code(
            "from pathlib import Path\nimport json, os, shutil, subprocess, sys, yaml\n\n"
            "try:\n    from google.colab import drive\n    drive.mount('/content/drive')\n"
            "except ImportError:\n    print('Execução fora do Colab.')\n\n"
            "candidates=[Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'),Path('/content/drive/MyDrive/Projeto_TCC_Lucas')]\n"
            "PROJECT_ROOT=next((p for p in candidates if p.is_dir()),Path.cwd())\n"
            "SOURCE_ROOT=PROJECT_ROOT/'src'; os.chdir(PROJECT_ROOT)\n"
            "os.environ['PYTHONPATH']=str(SOURCE_ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')\n"
            "if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0,str(SOURCE_ROOT))\n"
            "print('Projeto:',PROJECT_ROOT)"
        ),
        markdown("## 2. Instalar dependências e preparar o cache temporário"),
        code(
            "result=subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements-colab.txt'],text=True,capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()\n"
            "DATA_ROOT=Path('/content/tcc_data') if Path('/content').is_dir() else PROJECT_ROOT\n"
            "if DATA_ROOT!=PROJECT_ROOT: shutil.copytree(PROJECT_ROOT/'Dataset',DATA_ROOT/'Dataset',dirs_exist_ok=True)\n"
            "print('Dados:',DATA_ROOT/'Dataset')"
        ),
        markdown("## 3. Conferir que somente o augmentation difere do D0"),
        code(
            "from IPython.display import display\n"
            "D0=yaml.safe_load((PROJECT_ROOT/'configs/baseline_d0.yaml').read_text(encoding='utf-8'))\n"
            "A1_PATH=PROJECT_ROOT/'configs/augmentation_d0_a1.yaml'; A2_PATH=PROJECT_ROOT/'configs/augmentation_d0_a2.yaml'\n"
            "A1=yaml.safe_load(A1_PATH.read_text(encoding='utf-8')); A2=yaml.safe_load(A2_PATH.read_text(encoding='utf-8'))\n"
            "AUDIT=yaml.safe_load((PROJECT_ROOT/'configs/augmentation_audit.yaml').read_text(encoding='utf-8'))\n"
            "from tcc_segmentation.augmentation_experiment import validate_augmentation_training_config\n"
            "for cfg in (A1,A2): validate_augmentation_training_config(cfg,d0_config=D0,audit_config=AUDIT)\n"
            "display({'D0':D0['data']['augmentation'],'D0-A1':A1['data']['augmentation'],'D0-A2':A2['data']['augmentation'],"
            "'50% filtra treino':A1['augmentation_safety']['retention_threshold_filters_training']})"
        ),
        markdown("## 4. Executar testes técnicos antes de treinar"),
        code(
            "env=os.environ.copy(); env['PYTHONPATH']=str(SOURCE_ROOT)+os.pathsep+env.get('PYTHONPATH','')\n"
            "result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_augmentation.py','tests/test_augmentation_training.py'],cwd=PROJECT_ROOT,env=env,text=True,capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown(
            "## 5. Visualizar o que a rede poderá receber\n\n"
            "A visualização usa quatro amostras de treino e seed da época zero. Ela é um "
            "controle de implementação, não um critério para escolher A1 ou A2."
        ),
        code(
            "import matplotlib.pyplot as plt\n"
            "from tcc_segmentation.data import DevelopmentDataset\n"
            "from tcc_segmentation.augmentation_experiment import make_augmentation_pair_transform\n"
            "fig,axes=plt.subplots(2,4,figsize=(16,8))\n"
            "for row,(label,cfg) in enumerate([('A1',A1),('A2',A2)]):\n"
            "    transform=make_augmentation_pair_transform(cfg); transform.set_epoch(0)\n"
            "    ds=DevelopmentDataset(PROJECT_ROOT/cfg['data']['train_manifest'],DATA_ROOT,'train',transform=transform,mask_threshold=cfg['data']['mask_threshold'])\n"
            "    for col in range(4):\n"
            "        sample=ds[col]; axes[row,col].imshow(sample['image'].transpose(1,2,0)); axes[row,col].contour(sample['mask'][0],levels=[0.5],colors=['lime'],linewidths=1)\n"
            "        axes[row,col].set_title(label+' | '+sample['canonical_id']); axes[row,col].axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        markdown("## 6. Função para acompanhar cada época"),
        code(
            "def run_stream(command):\n"
            "    process=subprocess.Popen(command,cwd=PROJECT_ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)\n"
            "    for line in process.stdout: print(line,end='')\n"
            "    returncode=process.wait(); print('\\nreturncode=',returncode)\n"
            "    if returncode!=0: raise subprocess.CalledProcessError(returncode,command)"
        ),
        markdown(
            "# Parte A — D0-A1\n\nMantenha `A1_RESUME_DIR=None` para uma execução nova. "
            "Após desconexão, informe o diretório já criado para continuar a mesma execução."
        ),
        code(
            "# Exemplo: A1_RESUME_DIR=PROJECT_ROOT/'results/day7/D0A1/D0A1_seed42_...'\n"
            "A1_RESUME_DIR=None\n"
            "command=[sys.executable,'scripts/run_augmentation_experiment.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config',str(A1_PATH)]\n"
            "if A1_RESUME_DIR is not None: command += ['--resume-dir',str(A1_RESUME_DIR)]\n"
            "run_stream(command)"
        ),
        markdown("## 7. Ver resultados e segurança de A1"),
        code(
            "from PIL import Image\nimport pandas as pd\n"
            "A1_OUT=Path(A1_RESUME_DIR) if A1_RESUME_DIR else max((PROJECT_ROOT/A1['runtime']['output_root']).iterdir(),key=lambda p:p.stat().st_mtime)\n"
            "A1_SUM=json.loads((A1_OUT/'final_summary.json').read_text()); A1_SAFE=json.loads((A1_OUT/'augmentation_safety.json').read_text())\n"
            "display(A1_SAFE); display(Image.open(A1_OUT/'first_training_batch.png')); display(Image.open(A1_OUT/'training_curves.png')); display(Image.open(A1_OUT/'fixed_validation_panel.png'))"
        ),
        markdown("# Parte B — D0-A2\n\nA2 começa do zero e não carrega pesos de A1."),
        code(
            "# Exemplo: A2_RESUME_DIR=PROJECT_ROOT/'results/day7/D0A2/D0A2_seed42_...'\n"
            "A2_RESUME_DIR=None\n"
            "command=[sys.executable,'scripts/run_augmentation_experiment.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config',str(A2_PATH)]\n"
            "if A2_RESUME_DIR is not None: command += ['--resume-dir',str(A2_RESUME_DIR)]\n"
            "run_stream(command)"
        ),
        markdown("## 8. Ver resultados e segurança de A2"),
        code(
            "A2_OUT=Path(A2_RESUME_DIR) if A2_RESUME_DIR else max((PROJECT_ROOT/A2['runtime']['output_root']).iterdir(),key=lambda p:p.stat().st_mtime)\n"
            "A2_SUM=json.loads((A2_OUT/'final_summary.json').read_text()); A2_SAFE=json.loads((A2_OUT/'augmentation_safety.json').read_text())\n"
            "display(A2_SAFE); display(Image.open(A2_OUT/'first_training_batch.png')); display(Image.open(A2_OUT/'training_curves.png')); display(Image.open(A2_OUT/'fixed_validation_panel.png'))"
        ),
        markdown(
            "## 9. Tabela provisória somente da validação\n\nEsta tabela organiza os resultados; "
            "a escolha ainda exigirá comparação pareada por imagem e incerteza."
        ),
        code(
            "D0_DIR=PROJECT_ROOT/'results/day5/D0/D0_seed42_20260829T105601Z'; D0_SUM=json.loads((D0_DIR/'final_summary.json').read_text())\n"
            "rows=[]\n"
            "for label,summary in [('A0/D0',D0_SUM),('A1',A1_SUM),('A2',A2_SUM)]:\n"
            "    s=summary['selection']; rows.append({'braço':label,'melhor_época':summary['training']['best_epoch'],'Dice_0.5':s['metrics_at_0_5']['macro']['dice']['mean'],'threshold':s['threshold'],'Dice_selecionado':s['metrics_selected']['macro']['dice']['mean'],'IoU':s['metrics_selected']['macro']['iou']['mean'],'precisão':s['metrics_selected']['macro']['precision']['mean'],'recall':s['metrics_selected']['macro']['recall']['mean']})\n"
            "display(pd.DataFrame(rows))"
        ),
        markdown(
            (
                '## Artefatos da execução\n'
                '\n'
                'Os arquivos produzidos por configuração incluem `final_summary.json`, '
                '`history.csv`, `training_curves.png`, `threshold_curve.csv`, '
                '`threshold_curves.png`, `threshold_selection.json`, '
                '`fixed_validation_panel.png`, `validation_by_area_bin.csv`, '
                '`validation_per_image_selected.csv`, `augmentation_safety.json` e '
                '`first_training_batch.png`.\n'
                '\n'
                'Esta etapa avalia A1 e A2 na partição de validação do DFUC2022.'
            )
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "11_augmentation_training.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path(__file__).resolve().parents[1] / "notebooks/11_augmentation_training.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
