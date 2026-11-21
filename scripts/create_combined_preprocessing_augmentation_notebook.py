#!/usr/bin/env python3
"""Gera o notebook didático da combinação planejada P1+A1."""

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
            "# Combinação controlada P1+A1\n\n"
            "Este notebook investiga se o Shades-of-Gray (`P1`) e o augmentation "
            "geométrico (`A1`) têm efeitos complementares sobre a DeepResUNet. O "
            "modelo começa do zero. Teste interno e FUSeg permanecem fechados."
        ),
        markdown(
            "## O que muda e o que permanece fixo\n\n"
            "No treino: imagem original → P1 → resize/pad → A1 → tensor `[0,1]`. "
            "Na validação: imagem original → P1 → resize/pad → tensor `[0,1]`, sem "
            "augmentation. Arquitetura, split, seed 42, resolução, loss, otimizador, "
            "scheduler, early stopping e regra de threshold são os mesmos do D0."
        ),
        markdown(
            "## Interpretação planejada\n\n"
            "- se P1+A1 superar P1 e A1, há evidência de complementaridade nesta seed;\n"
            "- se ficar próximo do melhor isolado, os efeitos podem ser redundantes;\n"
            "- se piorar, a combinação pode interferir negativamente.\n\n"
            "A comparação será feita somente depois de retornar os artefatos."
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
        markdown("## 3. Validar que a configuração combina exatamente P1 e A1"),
        code(
            "from IPython.display import display\n"
            "CONFIG_PATH=PROJECT_ROOT/'configs/combined_preprocessing_augmentation_p1_a1.yaml'\n"
            "CFG=yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "D0=yaml.safe_load((PROJECT_ROOT/'configs/baseline_d0.yaml').read_text(encoding='utf-8'))\n"
            "P1=yaml.safe_load((PROJECT_ROOT/'configs/preprocessing_dp1_shades_of_gray.yaml').read_text(encoding='utf-8'))\n"
            "A1=yaml.safe_load((PROJECT_ROOT/'configs/augmentation_d0_a1.yaml').read_text(encoding='utf-8'))\n"
            "AUDIT=yaml.safe_load((PROJECT_ROOT/'configs/augmentation_audit.yaml').read_text(encoding='utf-8'))\n"
            "from tcc_segmentation.combined_preprocessing_augmentation import validate_p1_a1_config\n"
            "validate_p1_a1_config(CFG,d0_config=D0,p1_config=P1,a1_config=A1,audit_config=AUDIT)\n"
            "display({'experimento':CFG['experiment_id'],'pré-processamento':CFG['data']['color_preprocessing'],'p':CFG['data']['color_preprocessing_parameters']['minkowski_p'],'augmentation':CFG['data']['augmentation'],'pesos prévios':CFG['model']['pretrained'],'seed':CFG['seed'],'contraste primário':CFG['comparison_plan']['primary_contrast'],'teste permitido':CFG['comparison_plan']['test_access_allowed']})"
        ),
        markdown("## 4. Executar os testes técnicos"),
        code(
            "env=os.environ.copy(); env['PYTHONPATH']=str(SOURCE_ROOT)+os.pathsep+env.get('PYTHONPATH','')\n"
            "result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_combined_preprocessing_augmentation.py','tests/test_augmentation.py','tests/test_preprocessing.py'],cwd=PROJECT_ROOT,env=env,text=True,capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown(
            "## 5. Visualizar a ordem real do pipeline\n\n"
            "A linha superior mostra P1+A1 no treino; a inferior mostra P1 na "
            "validação. A máscara deve permanecer binária e alinhada."
        ),
        code(
            "import matplotlib.pyplot as plt\nimport numpy as np\n"
            "from tcc_segmentation.data import DevelopmentDataset\n"
            "from tcc_segmentation.combined_preprocessing_augmentation import make_p1_a1_train_transform,make_p1_validation_transform\n"
            "train_transform=make_p1_a1_train_transform(CFG); train_transform.set_epoch(0)\n"
            "train_ds=DevelopmentDataset(PROJECT_ROOT/CFG['data']['train_manifest'],DATA_ROOT,'train',transform=train_transform,mask_threshold=CFG['data']['mask_threshold'])\n"
            "val_ds=DevelopmentDataset(PROJECT_ROOT/CFG['data']['validation_manifest'],DATA_ROOT,'validation',transform=make_p1_validation_transform(CFG),mask_threshold=CFG['data']['mask_threshold'])\n"
            "fig,axes=plt.subplots(2,4,figsize=(16,8))\n"
            "for row,(label,ds) in enumerate([('treino: P1+A1',train_ds),('validação: somente P1',val_ds)]):\n"
            "    for col in range(4):\n"
            "        sample=ds[col]; image=sample['image'].transpose(1,2,0); mask=sample['mask'][0]\n"
            "        axes[row,col].imshow(image); axes[row,col].contour(mask,levels=[0.5],colors=['lime'],linewidths=1)\n"
            "        axes[row,col].set_title(label+' | '+sample['canonical_id']); axes[row,col].axis('off')\n"
            "        assert set(np.unique(mask)).issubset({0.0,1.0})\n"
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
            "## 7. Treinar P1+A1\n\n"
            "Mantenha `RESUME_DIR=None` para uma execução nova. Em caso de desconexão, "
            "informe o diretório criado para continuar a mesma execução. Não aponte "
            "para diretórios de D0, P1 ou A1."
        ),
        code(
            "# Exemplo: RESUME_DIR=PROJECT_ROOT/'results/day8/P1A1/P1A1_seed42_...'\n"
            "RESUME_DIR=None\n"
            "command=[sys.executable,'scripts/run_combined_preprocessing_augmentation.py','--project-root',str(PROJECT_ROOT),'--data-root',str(DATA_ROOT),'--config',str(CONFIG_PATH)]\n"
            "if RESUME_DIR is not None: command += ['--resume-dir',str(RESUME_DIR)]\n"
            "run_stream(command)"
        ),
        markdown("## 8. Localizar e visualizar as saídas"),
        code(
            "from PIL import Image\nimport pandas as pd\n"
            "OUT=Path(RESUME_DIR) if RESUME_DIR else max((PROJECT_ROOT/CFG['runtime']['output_root']).iterdir(),key=lambda p:p.stat().st_mtime)\n"
            "SUMMARY=json.loads((OUT/'final_summary.json').read_text(encoding='utf-8'))\n"
            "SAFETY=json.loads((OUT/'augmentation_safety.json').read_text(encoding='utf-8'))\n"
            "display({'run':SUMMARY['run_id'],'melhor época':SUMMARY['training']['best_epoch'],'Dice 0,5':SUMMARY['selection']['metrics_at_0_5']['macro']['dice']['mean'],'threshold':SUMMARY['selection']['threshold'],'Dice selecionado':SUMMARY['selection']['metrics_selected']['macro']['dice']['mean']})\n"
            "display(SAFETY)\n"
            "for name in ('first_training_batch.png','training_curves.png','threshold_curves.png','fixed_validation_panel.png'):\n    display(Image.open(OUT/name))"
        ),
        markdown(
            (
                '## 9. Verificação dos arquivos da execução\n'
                '\n'
                'Esta célula apenas lista os arquivos exigidos; não escolhe o resultado.'
            )
        ),
        code(
            "required=['final_summary.json','history.csv','training_curves.png','threshold_curve.csv','threshold_curves.png','threshold_selection.json','fixed_validation_panel.png','validation_by_area_bin.csv','validation_per_image_selected.csv','validation_per_image_at_0_5.csv','validation_metrics_selected.json','validation_metrics_at_0_5.json','augmentation_safety.json','first_training_batch.png','environment_report.json','config_used.yaml']\n"
            "missing=[name for name in required if not (OUT/name).is_file()]\n"
            "display(pd.DataFrame({'arquivo':required,'presente':[(OUT/name).is_file() for name in required]}))\n"
            "assert not missing, 'Arquivos ausentes: '+str(missing)\nprint('Arquivos da execução verificados:',OUT)"
        ),
        markdown(
            "## Artefatos da execução\n\n"
            "Arquive os resultados listados ou compacte o diretório da execução. Não abra o "
            "teste interno, não execute FUSeg e não inicie transfer learning antes da análise."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "12_combined_preprocessing_augmentation.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = (
        Path(__file__).resolve().parents[1]
        / "notebooks/12_combined_preprocessing_augmentation.ipynb"
    )
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
