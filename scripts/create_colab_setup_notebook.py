#!/usr/bin/env python3
"""Gera o notebook fino de configuracao e smoke test do Google Colab."""

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
            "# 00 — Ambiente Colab e pipeline de dados\n\n"
            "Objetivo: montar o Drive, registrar o ambiente e validar o carregamento "
            "de **treino e validação somente**. Este notebook não treina modelos, "
            "não abre o teste interno e não acessa o FUSeg.\n\n"
            "Runtime de referência: **Colab 2026.07**, Python 3.12, GPU. Se esse "
            "runtime não estiver disponível, use o mais recente e preserve o relatório "
            "gerado pela célula de diagnóstico."
        ),
        code(
            "from pathlib import Path\n"
            "import os\n\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "except ImportError:\n"
            "    print('Execução fora do Colab: montagem do Drive ignorada.')\n\n"
            "# Ajuste somente se a pasta estiver em outro local do seu Google Drive.\n"
            "PROJECT_ROOT = Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas')\n"
            "if not PROJECT_ROOT.is_dir():\n"
            "    raise FileNotFoundError(f'Raiz do projeto não encontrada: {PROJECT_ROOT}')\n"
            "os.chdir(PROJECT_ROOT)\n"
            "print(f'Projeto: {PROJECT_ROOT}')"
        ),
        markdown(
            "## Dependências\n\n"
            "O PyTorch do Colab não é reinstalado, porque sua distribuição já é "
            "compatível com o CUDA do runtime. As bibliotecas adicionais possuem "
            "versões fixadas em `requirements-colab.txt`. Após esta célula, reinicie "
            "o runtime apenas se o Colab solicitar."
        ),
        code(
            "import subprocess\n"
            "import sys\n\n"
            "def run_and_show(command):\n"
            "    result = subprocess.run(\n"
            "        command,\n"
            "        stdout=subprocess.PIPE,\n"
            "        stderr=subprocess.STDOUT,\n"
            "        text=True,\n"
            "    )\n"
            "    print(result.stdout, end='')\n"
            "    print(f'\\nreturncode={result.returncode}')\n"
            "    result.check_returncode()\n"
            "    return result\n\n"
            "_ = run_and_show(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'],\n"
            ")"
        ),
        markdown("## Registro do ambiente real"),
        code("_ = run_and_show(\n" "    [sys.executable, 'scripts/check_environment.py'],\n" ")"),
        markdown(
            "## Cache temporário dos dados\n\n"
            "O dataset ocupa aproximadamente 303 MB. A cópia para o disco temporário "
            "do runtime reduz leituras repetidas no Google Drive. O código e todos os "
            "resultados permanentes continuam no Drive."
        ),
        code(
            "import shutil\n\n"
            "DATA_ROOT = Path('/content/tcc_data')\n"
            "shutil.copytree(\n"
            "    PROJECT_ROOT / 'Dataset',\n"
            "    DATA_ROOT / 'Dataset',\n"
            "    dirs_exist_ok=True,\n"
            ")\n"
            "print(f'Dados temporários: {DATA_ROOT / \"Dataset\"}')"
        ),
        markdown(
            "## Testes do carregamento\n\n"
            "Os testes abaixo usam imagens reais apenas de treino/validação. Também "
            "confirmam que a proteção rejeita uma solicitação de partição reservada "
            "antes de abrir seu manifesto."
        ),
        code(
            "_ = run_and_show(\n"
            "    [sys.executable, '-m', 'pytest', '-q', 'tests/test_development_data.py'],\n"
            ")"
        ),
        markdown("## Smoke test com PyTorch DataLoader"),
        code(
            "for split in ('train', 'validation'):\n"
            "    _ = run_and_show(\n"
            "        [\n"
            "            sys.executable,\n"
            "            'scripts/run_data_pipeline_smoke.py',\n"
            "            '--project-root', str(PROJECT_ROOT),\n"
            "            '--data-root', str(DATA_ROOT),\n"
            "            '--split', split,\n"
            "            '--samples', '4',\n"
            "            '--with-torch-loader',\n"
            "        ],\n"
            "    )"
        ),
        markdown(
            (
                '## Critério de conclusão\n'
                '\n'
                'A etapa está pronta quando: o relatório mostra GPU disponível; os testes '
                'terminam sem erro; os batches possuem formato `[B, 3, 384, 512]` para imagem '
                'e `[B, 1, 384, 512]` para máscara; e as máscaras contêm apenas 0 e 1. A saída'
                ' do notebook pode ser consultada no Google Drive.'
            )
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "00_colab_setup.ipynb"},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/00_colab_setup.ipynb"),
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
