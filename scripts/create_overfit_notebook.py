#!/usr/bin/env python3
"""Gera o notebook Colab do gate de overfit em poucas imagens."""

from __future__ import annotations

import argparse
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


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            "# 01 — Gate de treinamento: overfit em quatro imagens\n\n"
            "Este notebook testa o pipeline completo exclusivamente em quatro amostras "
            "de treino. Não é resultado científico, não usa validação, teste ou FUSeg."
        ),
        code(
            "from pathlib import Path\n"
            "import os\n\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive')\n"
            "PROJECT_ROOT = Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas')\n"
            "if not PROJECT_ROOT.is_dir():\n"
            "    raise FileNotFoundError(PROJECT_ROOT)\n"
            "os.chdir(PROJECT_ROOT)\n"
            "print(f'Projeto: {PROJECT_ROOT}')"
        ),
        markdown("## Dependências e saída contínua"),
        code(
            "import subprocess\n"
            "import sys\n\n"
            "def run_stream(command, allowed_returncodes=(0,)):\n"
            "    process = subprocess.Popen(\n"
            "        command,\n"
            "        stdout=subprocess.PIPE,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        text=True,\n"
            "        bufsize=1,\n"
            "    )\n"
            "    for line in process.stdout:\n"
            "        print(line, end='')\n"
            "    returncode = process.wait()\n"
            "    print(f'\\nreturncode={returncode}')\n"
            "    if returncode not in allowed_returncodes:\n"
            "        raise subprocess.CalledProcessError(returncode, command)\n"
            "    return returncode\n\n"
            "run_stream([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'])"
        ),
        markdown("## Ambiente e imports"),
        code("run_stream([sys.executable, 'scripts/check_environment.py'])"),
        markdown("## Cache temporário dos dados"),
        code(
            "import shutil\n\n"
            "DATA_ROOT = Path('/content/tcc_data')\n"
            "shutil.copytree(\n"
            "    PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True\n"
            ")\n"
            "print(DATA_ROOT / 'Dataset')"
        ),
        markdown(
            "## Testes dos componentes\n\n"
            "Loss, gradiente, métricas por imagem, checkpoint, U-Net e proteção da "
            "configuração são testados antes do uso da GPU."
        ),
        code(
            "run_stream([\n"
            "    sys.executable, '-m', 'pytest', '-q',\n"
            "    'tests/test_training_components.py',\n"
            "    'tests/test_overfit_config.py',\n"
            "])"
        ),
        markdown(
            "## Executar o overfit\n\n"
            "O processo imprime uma linha por época e pode encerrar antes de 200 épocas "
            "se Dice ≥ 0,97 for mantido por cinco épocas."
        ),
        code(
            "run_stream([\n"
            "    sys.executable, 'scripts/run_overfit_smoke.py',\n"
            "    '--project-root', str(PROJECT_ROOT),\n"
            "    '--data-root', str(DATA_ROOT),\n"
            "    '--config', 'configs/overfit_smoke.yaml',\n"
            "])"
        ),
        markdown("## Mostrar o resultado mais recente"),
        code(
            "import json\n"
            "from IPython.display import Image as DisplayImage, display\n\n"
            "runs = sorted((PROJECT_ROOT / 'results/smoke/overfit').glob('overfit_smoke_*'))\n"
            "if not runs:\n"
            "    raise FileNotFoundError('Nenhuma execução de overfit encontrada')\n"
            "LATEST_RUN = runs[-1]\n"
            "summary = json.loads((LATEST_RUN / 'final_summary.json').read_text())\n"
            "display(summary)\n"
            "display(DisplayImage(filename=str(LATEST_RUN / 'overfit_panel.png')))\n"
            "print(f'Resultados: {LATEST_RUN}')"
        ),
        markdown(
            "## Estabilização formal após o primeiro resultado\n\n"
            "Use esta célula quando o primeiro treino tiver alcançado Dice ≥ 0,97 apenas "
            "pontualmente. Ela carrega o melhor checkpoint, reduz o learning rate para "
            "0,0001 e exige cinco épocas consecutivas acima da meta. Uma nova pasta é "
            "criada; a execução original não é sobrescrita. O código de saída 2 apenas "
            "indica que o gate ainda não foi atingido e não apaga os artefatos."
        ),
        code(
            "source_runs = [\n"
            "    path for path in (PROJECT_ROOT / 'results/smoke/overfit').glob('overfit_smoke_*')\n"
            "    if (path / 'checkpoints/best.pt').is_file()\n"
            "]\n"
            "if not source_runs:\n"
            "    raise FileNotFoundError('Nenhum checkpoints/best.pt do overfit foi encontrado')\n"
            "SOURCE_RUN = max(source_runs, key=lambda path: path.stat().st_mtime)\n"
            "SOURCE_CHECKPOINT = SOURCE_RUN / 'checkpoints/best.pt'\n"
            "print(f'Checkpoint de origem: {SOURCE_CHECKPOINT}')\n\n"
            "run_stream([\n"
            "    sys.executable, 'scripts/run_overfit_smoke.py',\n"
            "    '--project-root', str(PROJECT_ROOT),\n"
            "    '--data-root', str(DATA_ROOT),\n"
            "    '--config', 'configs/overfit_smoke.yaml',\n"
            "    '--stabilize-from', str(SOURCE_CHECKPOINT),\n"
            "    '--stabilization-epochs', '30',\n"
            "    '--stabilization-learning-rate', '0.0001',\n"
            "], allowed_returncodes=(0, 2))"
        ),
        markdown("## Mostrar o resultado da estabilização"),
        code(
            "stabilization_runs = list(\n"
            "    (PROJECT_ROOT / 'results/smoke/overfit').glob('overfit_stabilization_*')\n"
            ")\n"
            "if not stabilization_runs:\n"
            "    raise FileNotFoundError('Nenhuma execução de estabilização encontrada')\n"
            "STABILIZATION_RUN = max(stabilization_runs, key=lambda path: path.stat().st_mtime)\n"
            "stabilization_summary = json.loads(\n"
            "    (STABILIZATION_RUN / 'final_summary.json').read_text()\n"
            ")\n"
            "display(stabilization_summary)\n"
            "display(DisplayImage(filename=str(STABILIZATION_RUN / 'overfit_panel.png')))\n"
            "print(f'Resultados: {STABILIZATION_RUN}')"
        ),
        markdown(
            "## Retomada após interrupção\n\n"
            "Se o Colab for interrompido, execute novamente as células iniciais e use:\n\n"
            "```python\n"
            "RESUME = LATEST_RUN / 'checkpoints/last.pt'\n"
            "run_stream([sys.executable, 'scripts/run_overfit_smoke.py', "
            "'--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT), "
            "'--config', 'configs/overfit_smoke.yaml', '--resume', str(RESUME)])\n"
            "```"
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "01_training_overfit_smoke.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/01_training_overfit_smoke.ipynb"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_notebook(), indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
