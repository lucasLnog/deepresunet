#!/usr/bin/env python3
"""Gera o notebook Colab para treino e threshold dos baselines de desenvolvimento."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
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
            "# 03 — Baselines completos em treino e validação\n\n"
            "Este notebook executa um baseline por vez. O teste interno e o FUSeg "
            "permanecem bloqueados. Comece pela U-Net/ResNet-34 aleatória; somente "
            "depois de revisar seu resultado execute a DeepResUNet histórica."
        ),
        code(
            "from pathlib import Path\n"
            "import os\n"
            "from google.colab import drive\n\n"
            "drive.mount('/content/drive')\n"
            "candidates = [\n"
            "    Path('/content/drive/MyDrive/Projeto_TCC_Lucas'),\n"
            "    Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'),\n"
            "]\n"
            "PROJECT_ROOT = next((path for path in candidates if path.is_dir()), None)\n"
            "if PROJECT_ROOT is None:\n"
            "    raise FileNotFoundError(candidates)\n"
            "os.chdir(PROJECT_ROOT)\n"
            "print(f'Projeto: {PROJECT_ROOT}')"
        ),
        markdown("## Dependências e saída contínua"),
        code(
            "import subprocess\n"
            "import sys\n\n"
            "def run_stream(command):\n"
            "    process = subprocess.Popen(\n"
            "        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
            "        text=True, bufsize=1,\n"
            "    )\n"
            "    for line in process.stdout:\n"
            "        print(line, end='')\n"
            "    returncode = process.wait()\n"
            "    print(f'\\nreturncode={returncode}')\n"
            "    if returncode != 0:\n"
            "        raise subprocess.CalledProcessError(returncode, command)\n"
            "    return returncode\n\n"
            "run_stream([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'])"
        ),
        markdown("## Cache temporário do dataset"),
        code(
            "import shutil\n\n"
            "DATA_ROOT = Path('/content/tcc_data')\n"
            "shutil.copytree(PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
            "print(f'Dados temporários: {DATA_ROOT / \"Dataset\"}')"
        ),
        markdown(
            "## Escolher o baseline\n\n"
            "Mantenha a primeira opção na execução inicial. Para o baseline histórico, "
            "troque pela segunda linha e execute novamente a partir desta célula."
        ),
        code(
            "CONFIG_RELATIVE = Path('configs/baselines/unet_resnet34_random.yaml')\n"
            "# CONFIG_RELATIVE = Path('configs/baselines/deep_res_unet_historical.yaml')\n"
            "CONFIG_PATH = PROJECT_ROOT / CONFIG_RELATIVE\n"
            "print(f'Configuração selecionada: {CONFIG_PATH}')"
        ),
        markdown("## Testes antes de usar a GPU"),
        code(
            "run_stream([\n"
            "    sys.executable, '-m', 'pytest', '-q',\n"
            "    'tests/test_baseline_config.py',\n"
            "    'tests/test_training_components.py',\n"
            "    'tests/test_deep_res_unet.py',\n"
            "])"
        ),
        markdown(
            "## Treinar\n\n"
            "A execução salva `last.pt` a cada época e encerra por no máximo 50 épocas "
            "ou early stopping. A seleção usa somente o Dice médio de validação em 0,5."
        ),
        code(
            "run_stream([\n"
            "    sys.executable, 'scripts/run_development_baseline.py',\n"
            "    '--project-root', str(PROJECT_ROOT),\n"
            "    '--data-root', str(DATA_ROOT),\n"
            "    '--config', str(CONFIG_RELATIVE),\n"
            "])"
        ),
        markdown("## Localizar a execução criada"),
        code(
            "import yaml\n"
            "config = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))\n"
            "experiment_id = config['experiment_id']\n"
            "run_root = PROJECT_ROOT / config['output']['root']\n"
            "runs = list(run_root.glob(f'{experiment_id}_seed*'))\n"
            "if not runs:\n"
            "    raise FileNotFoundError(f'Execução não encontrada para {experiment_id}')\n"
            "RUN_DIR = max(runs, key=lambda path: path.stat().st_mtime)\n"
            "print(f'Execução: {RUN_DIR}')"
        ),
        markdown(
            "## Selecionar threshold somente na validação\n\n"
            "O melhor checkpoint é recarregado, inferido uma vez por batch e avaliado "
            "entre 0,05 e 0,95. Nenhum threshold por imagem é permitido."
        ),
        code(
            "run_stream([\n"
            "    sys.executable, 'scripts/analyze_validation_thresholds.py',\n"
            "    '--project-root', str(PROJECT_ROOT),\n"
            "    '--data-root', str(DATA_ROOT),\n"
            "    '--run-dir', str(RUN_DIR),\n"
            "])"
        ),
        markdown("## Relatórios e curvas"),
        code(
            "import csv\n"
            "import json\n"
            "import matplotlib.pyplot as plt\n\n"
            "training_summary = json.loads((RUN_DIR / 'training_summary.json').read_text())\n"
            "threshold_summary = json.loads((RUN_DIR / 'threshold_selection.json').read_text())\n"
            "display(training_summary)\n"
            "display(threshold_summary)\n\n"
            "with (RUN_DIR / 'history.csv').open(encoding='utf-8') as stream:\n"
            "    history = list(csv.DictReader(stream))\n"
            "epochs = [int(row['epoch']) for row in history]\n"
            "train_loss = [float(row['train_loss_total']) for row in history]\n"
            "val_loss = [float(row['validation_loss_total']) for row in history]\n"
            "val_dice = [float(row['validation_dice_mean']) for row in history]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "axes[0].plot(epochs, train_loss, label='treino')\n"
            "axes[0].plot(epochs, val_loss, label='validação')\n"
            "axes[0].set(xlabel='Época', ylabel='Loss', title='Loss')\n"
            "axes[0].legend()\n"
            "axes[1].plot(epochs, val_dice)\n"
            "axes[1].set(xlabel='Época', ylabel='Dice médio', title='Validação em threshold 0,5')\n"
            "fig.tight_layout()\n"
            "fig.savefig(RUN_DIR / 'training_curves.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()\n\n"
            "with (RUN_DIR / 'threshold_curve.csv').open(encoding='utf-8') as stream:\n"
            "    curve = list(csv.DictReader(stream))\n"
            "thresholds = [float(row['threshold']) for row in curve]\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "for metric in ('dice', 'iou', 'precision', 'recall'):\n"
            "    ax.plot(thresholds, [float(row[f'{metric}_mean']) for row in curve], label=metric)\n"
            "ax.axvline(threshold_summary['selected_threshold'], color='black', linestyle='--', label='selecionado')\n"
            "ax.set(xlabel='Threshold', ylabel='Média por imagem', title='Sensibilidade ao threshold — validação')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "fig.savefig(RUN_DIR / 'threshold_curves.png', dpi=160, bbox_inches='tight')\n"
            "plt.show()\n"
            "print(f'Artefatos: {RUN_DIR}')"
        ),
        markdown(
            "## Retomada após interrupção\n\n"
            "Se a sessão cair, execute as células iniciais, selecione a mesma configuração, "
            "defina `RESUME = RUN_DIR / 'checkpoints/last.pt'` e use:\n\n"
            "```python\n"
            "run_stream([sys.executable, 'scripts/run_development_baseline.py', "
            "'--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT), "
            "'--config', str(CONFIG_RELATIVE), '--resume', str(RESUME)])\n"
            "```"
        ),
        markdown(
            (
                '## Ordem das execuções\n'
                '\n'
                '1. Execute e guarde os artefatos da U-Net/ResNet-34 aleatória.\n'
                '2. Após a revisão, selecione a DeepResUNet histórica e repita treino, '
                'threshold e curvas.\n'
                '3. Registre os resultados das referências antes da comparação com pesos '
                'ImageNet.'
            )
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "03_development_baselines.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path("notebooks/03_development_baselines.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
