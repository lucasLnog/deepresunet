#!/usr/bin/env python3
"""Gera o notebook didático dos treinamentos D-P1 e D-P2."""

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
            "# Treinamentos controlados D-P1 e D-P2\n\n"
            "Este notebook mede o efeito isolado de dois pré-processamentos sobre a "
            "DeepResUNet central. D-P1 usa Shades-of-Gray e D-P2 usa CLAHE em L\*. "
            "Arquitetura, split, seed, resolução, loss, treinamento e avaliação são "
            "iguais ao D0. Teste interno e FUSeg permanecem fechados."
        ),
        markdown(
            "## Contraste científico\n\n"
            "- `D-P1 − D0`: efeito de constância de cor;\n"
            "- `D-P2 − D0`: efeito de contraste local de luminosidade.\n\n"
            "D-P1 e D-P2 começam da mesma seed 42; nenhum checkpoint do D0 é carregado. "
            "Os métodos não são combinados."
        ),
        markdown(
            "## Organização prática\n\n"
            "Cada treino pode levar tempo semelhante ao D0. Execute P1 e P2 em células "
            "separadas. Se o Colab desconectar, use o diretório daquela execução em "
            "`P1_RESUME_DIR` ou `P2_RESUME_DIR`; não crie uma execução nova para continuar."
        ),
        markdown("## 1. Montar Drive, localizar projeto e registrar `src`"),
        code(
            "from pathlib import Path\nimport json, os, shutil, subprocess, sys, yaml\n\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "except ImportError:\n"
            "    print('Execução fora do Colab.')\n\n"
            "candidates = [Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'), "
            "Path('/content/drive/MyDrive/Projeto_TCC_Lucas')]\n"
            "PROJECT_ROOT = next((p for p in candidates if p.is_dir()), Path.cwd())\n"
            "SOURCE_ROOT = PROJECT_ROOT / 'src'\n"
            "os.chdir(PROJECT_ROOT)\n"
            "os.environ['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
            "if str(SOURCE_ROOT) not in sys.path:\n    sys.path.insert(0, str(SOURCE_ROOT))\n"
            "print('Projeto:', PROJECT_ROOT)\nprint('Código-fonte:', SOURCE_ROOT)"
        ),
        markdown("## 2. Instalar dependências registradas"),
        code(
            "result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
            "'-r', 'requirements-colab.txt'], text=True, capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown("## 3. Copiar dados para o cache temporário"),
        code(
            "DATA_ROOT = Path('/content/tcc_data') if Path('/content').is_dir() else PROJECT_ROOT\n"
            "if DATA_ROOT != PROJECT_ROOT:\n"
            "    shutil.copytree(PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
            "print('Dados:', DATA_ROOT / 'Dataset')"
        ),
        markdown("## 4. Carregar e comparar as configurações"),
        code(
            "from IPython.display import display\n"
            "D0_CONFIG = yaml.safe_load((PROJECT_ROOT / 'configs/baseline_d0.yaml').read_text(encoding='utf-8'))\n"
            "P1_CONFIG_PATH = PROJECT_ROOT / 'configs/preprocessing_dp1_shades_of_gray.yaml'\n"
            "P2_CONFIG_PATH = PROJECT_ROOT / 'configs/preprocessing_dp2_clahe_lstar.yaml'\n"
            "P1_CONFIG = yaml.safe_load(P1_CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "P2_CONFIG = yaml.safe_load(P2_CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "display({'D0': D0_CONFIG['data']['color_preprocessing'], "
            "'D-P1': P1_CONFIG['data']['color_preprocessing'], "
            "'D-P2': P2_CONFIG['data']['color_preprocessing']})\n"
            "for candidate in (P1_CONFIG, P2_CONFIG):\n"
            "    assert candidate['model'] == D0_CONFIG['model']\n"
            "    assert candidate['loss'] == D0_CONFIG['loss']\n"
            "    assert candidate['training'] == D0_CONFIG['training']\n"
            "    assert candidate['threshold_selection'] == D0_CONFIG['threshold_selection']\n"
            "    assert candidate['seed'] == D0_CONFIG['seed'] == 42\n"
            "print('Controle aprovado: somente pré-processamento e identificação diferem.')"
        ),
        markdown("## 5. Executar os testes de configuração e transforms"),
        code(
            "test_environment = os.environ.copy()\n"
            "test_environment['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + test_environment.get('PYTHONPATH', '')\n"
            "result = subprocess.run([sys.executable, '-m', 'pytest', '-q', "
            "'tests/test_preprocessing.py', 'tests/test_preprocessing_training_config.py'], "
            "cwd=PROJECT_ROOT, env=test_environment, text=True, capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown(
            "## 6. Visualizar a entrada real da rede\n\n"
            "Esta célula confirma P0/P1/P2 após resize. A máscara deve permanecer idêntica "
            "entre os braços; apenas a imagem muda."
        ),
        code(
            "import matplotlib.pyplot as plt\nimport numpy as np\n"
            "from tcc_segmentation.data import DevelopmentDataset, ResizePad\n"
            "from tcc_segmentation.preprocessing_experiment import make_preprocessing_pair_transform\n"
            "manifest = PROJECT_ROOT / P1_CONFIG['data']['validation_manifest']\n"
            "datasets = {\n"
            "    'P0': DevelopmentDataset(manifest, DATA_ROOT, 'validation', transform=ResizePad(384, 512)),\n"
            "    'P1': DevelopmentDataset(manifest, DATA_ROOT, 'validation', transform=make_preprocessing_pair_transform(P1_CONFIG)),\n"
            "    'P2': DevelopmentDataset(manifest, DATA_ROOT, 'validation', transform=make_preprocessing_pair_transform(P2_CONFIG)),\n"
            "}\n"
            "sample_id = P1_CONFIG['visualization']['fixed_validation_ids'][2]\n"
            "index = next(i for i,row in enumerate(datasets['P0'].rows) if row['canonical_id'] == sample_id)\n"
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4))\n"
            "masks = []\n"
            "for column, key in enumerate(('P0','P1','P2')):\n"
            "    sample = datasets[key][index]; masks.append(sample['mask'])\n"
            "    axes[column].imshow(sample['image'].transpose(1,2,0)); axes[column].set_title(key); axes[column].axis('off')\n"
            "axes[3].imshow(masks[0][0], cmap='gray'); axes[3].set_title('máscara comum'); axes[3].axis('off')\n"
            "assert all(np.array_equal(masks[0], mask) for mask in masks[1:])\n"
            "plt.tight_layout(); plt.show()"
        ),
        markdown(
            "## 7. Função de execução com saída contínua\n\n"
            "Ela imprime cada época e preserva o erro original caso algo falhe."
        ),
        code(
            "def run_stream(command):\n"
            "    process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, "
            "stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "    for line in process.stdout:\n        print(line, end='')\n"
            "    returncode = process.wait()\n    print('\\nreturncode=', returncode)\n"
            "    if returncode != 0:\n        raise subprocess.CalledProcessError(returncode, command)\n"
            "    return returncode"
        ),
        markdown(
            "# Parte A — D-P1: Shades-of-Gray\n\n"
            "Execute esta parte até o fim. Para uma execução nova, mantenha "
            "`P1_RESUME_DIR = None`. Em caso de desconexão, informe o diretório criado."
        ),
        code(
            "# Exemplo: P1_RESUME_DIR = PROJECT_ROOT / 'results/day6/DP1/DP1_seed42_...'\n"
            "P1_RESUME_DIR = None\n"
            "p1_command = [sys.executable, 'scripts/run_preprocessing_experiment.py', "
            "'--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT), "
            "'--config', str(P1_CONFIG_PATH)]\n"
            "if P1_RESUME_DIR is not None:\n    p1_command += ['--resume-dir', str(P1_RESUME_DIR)]\n"
            "run_stream(p1_command)"
        ),
        markdown("## 8. Visualizar o resultado D-P1"),
        code(
            "from PIL import Image\nimport pandas as pd\n"
            "P1_OUTPUT = Path(P1_RESUME_DIR) if P1_RESUME_DIR else max((PROJECT_ROOT / P1_CONFIG['runtime']['output_root']).iterdir(), key=lambda p:p.stat().st_mtime)\n"
            "P1_SUMMARY = json.loads((P1_OUTPUT / 'final_summary.json').read_text())\n"
            "display({'run': P1_SUMMARY['run_id'], 'melhor_época': P1_SUMMARY['training']['best_epoch'], "
            "'Dice_0.5': P1_SUMMARY['selection']['metrics_at_0_5']['macro']['dice']['mean'], "
            "'threshold': P1_SUMMARY['selection']['threshold'], "
            "'Dice_selecionado': P1_SUMMARY['selection']['metrics_selected']['macro']['dice']['mean']})\n"
            "display(Image.open(P1_OUTPUT / 'training_curves.png'))\n"
            "display(Image.open(P1_OUTPUT / 'threshold_curves.png'))\n"
            "display(Image.open(P1_OUTPUT / 'fixed_validation_panel.png'))"
        ),
        markdown(
            "# Parte B — D-P2: CLAHE em L\*\n\n"
            "Execute esta parte separadamente. Ela não carrega pesos nem resultados de P1."
        ),
        code(
            "# Exemplo: P2_RESUME_DIR = PROJECT_ROOT / 'results/day6/DP2/DP2_seed42_...'\n"
            "P2_RESUME_DIR = None\n"
            "p2_command = [sys.executable, 'scripts/run_preprocessing_experiment.py', "
            "'--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT), "
            "'--config', str(P2_CONFIG_PATH)]\n"
            "if P2_RESUME_DIR is not None:\n    p2_command += ['--resume-dir', str(P2_RESUME_DIR)]\n"
            "run_stream(p2_command)"
        ),
        markdown("## 9. Visualizar o resultado D-P2"),
        code(
            "P2_OUTPUT = Path(P2_RESUME_DIR) if P2_RESUME_DIR else max((PROJECT_ROOT / P2_CONFIG['runtime']['output_root']).iterdir(), key=lambda p:p.stat().st_mtime)\n"
            "P2_SUMMARY = json.loads((P2_OUTPUT / 'final_summary.json').read_text())\n"
            "display({'run': P2_SUMMARY['run_id'], 'melhor_época': P2_SUMMARY['training']['best_epoch'], "
            "'Dice_0.5': P2_SUMMARY['selection']['metrics_at_0_5']['macro']['dice']['mean'], "
            "'threshold': P2_SUMMARY['selection']['threshold'], "
            "'Dice_selecionado': P2_SUMMARY['selection']['metrics_selected']['macro']['dice']['mean']})\n"
            "display(Image.open(P2_OUTPUT / 'training_curves.png'))\n"
            "display(Image.open(P2_OUTPUT / 'threshold_curves.png'))\n"
            "display(Image.open(P2_OUTPUT / 'fixed_validation_panel.png'))"
        ),
        markdown(
            "## 10. Comparação provisória — não abrir teste\n\n"
            "A tabela abaixo organiza D0/P1/P2 usando somente validação. Não escolha "
            "manualmente outra época ou threshold depois de vê-la."
        ),
        code(
            "D0_DIR = PROJECT_ROOT / 'results/day5/D0/D0_seed42_20260829T105601Z'\n"
            "D0_SUMMARY = json.loads((D0_DIR / 'final_summary.json').read_text())\n"
            "rows=[]\n"
            "for label, summary in [('D0/P0',D0_SUMMARY),('D-P1',P1_SUMMARY),('D-P2',P2_SUMMARY)]:\n"
            "    sel=summary['selection']; rows.append({\n"
            "        'braço':label, 'melhor_época':summary['training']['best_epoch'],\n"
            "        'Dice_0.5':sel['metrics_at_0_5']['macro']['dice']['mean'],\n"
            "        'threshold':sel['threshold'],\n"
            "        'Dice_selecionado':sel['metrics_selected']['macro']['dice']['mean'],\n"
            "        'IoU_selecionado':sel['metrics_selected']['macro']['iou']['mean'],\n"
            "        'precisão':sel['metrics_selected']['macro']['precision']['mean'],\n"
            "        'recall':sel['metrics_selected']['macro']['recall']['mean']})\n"
            "display(pd.DataFrame(rows))"
        ),
        markdown(
            (
                '## Artefatos da execução\n'
                '\n'
                'Os arquivos produzidos por configuração são `final_summary.json`, '
                '`history.csv`, `training_curves.png`, `threshold_curve.csv`, '
                '`threshold_curves.png`, `threshold_selection.json`, '
                '`fixed_validation_panel.png` e `validation_by_area_bin.csv`.\n'
                '\n'
                'A comparação precisa ser revisada antes do augmentation. Não execute teste '
                'interno, FUSeg, P1+P2 ou qualquer ajuste adicional.'
            )
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "09_preprocessing_training.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path(__file__).resolve().parents[1] / "notebooks/09_preprocessing_training.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
