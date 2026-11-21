#!/usr/bin/env python3
"""Gera notebook Colab do gate DeepResUNet/ResNet-34."""

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
            "# 04 — Gate DeepResUNet com encoder ResNet-34\n\n"
            "Valida a substituição do encoder com tensores sintéticos. Comprova shapes, "
            "logits, igualdade inicial do decoder, pesos distintos do encoder, "
            "congelamento, BatchNorm, gradientes, fine-tuning progressivo e memória. "
            "Nenhum dataset é aberto."
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
        markdown("## Dependências"),
        code(
            "import subprocess\n"
            "import sys\n\n"
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'],\n"
            "    check=True, text=True, capture_output=True,\n"
            ")\n"
            "print(result.stdout)\n"
            "print(result.stderr)"
        ),
        markdown("## Testes unitários da variante"),
        code(
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', '-q', "
            "'tests/test_deep_res_unet_resnet34.py'],\n"
            "    check=True, text=True, capture_output=True,\n"
            ")\n"
            "print(result.stdout)\n"
            "print(result.stderr)"
        ),
        markdown("## Gate completo na GPU L4"),
        code(
            "result = subprocess.run(\n"
            "    [sys.executable, 'scripts/check_deep_res_unet_resnet34.py', "
            "'--project-root', str(PROJECT_ROOT), '--batch-size', '4'],\n"
            "    check=True, text=True, capture_output=True,\n"
            ")\n"
            "print(result.stdout)\n"
            "print(result.stderr)"
        ),
        markdown("## Relatório"),
        code(
            "import json\n"
            "report_path = PROJECT_ROOT / "
            "'results/architecture/deep_res_unet_resnet34_gate_report.json'\n"
            "report = json.loads(report_path.read_text(encoding='utf-8'))\n"
            "display(report)\n"
            "print(f'Relatório: {report_path}')"
        ),
        markdown(
            (
                '## Gate de decisão\n'
                '\n'
                'A verificação gera um relatório JSON. O status `passed` é requisito para os '
                'treinamentos com o encoder ResNet-34. A verificação usa entradas sintéticas.'
            )
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "04_deep_res_unet_resnet34_check.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output = Path("notebooks/04_deep_res_unet_resnet34_check.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
