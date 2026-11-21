#!/usr/bin/env python3
"""Gera notebook Colab para validar a migração da DeepResUNet."""

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
            "# 02 — Validação da migração da DeepResUNet\n\n"
            "Este notebook verifica arquitetura, parâmetros, logits, loss e gradientes "
            "com tensores sintéticos. Nenhum dataset é aberto e nenhum resultado "
            "científico é produzido."
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
            "subprocess.run(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'],\n"
            "    check=True,\n"
            ")"
        ),
        markdown("## Testes específicos da arquitetura"),
        code(
            "subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', '-q', 'tests/test_deep_res_unet.py'],\n"
            "    check=True,\n"
            ")"
        ),
        markdown("## Relatório reproduzível da migração"),
        code(
            "subprocess.run(\n"
            "    [\n"
            "        sys.executable, 'scripts/check_deep_res_unet_migration.py',\n"
            "        '--project-root', str(PROJECT_ROOT),\n"
            "    ],\n"
            "    check=True,\n"
            ")"
        ),
        markdown("## Mostrar e localizar a saída"),
        code(
            "import json\n"
            "report_path = (\n"
            "    PROJECT_ROOT / 'results/architecture/deep_res_unet_migration_report.json'\n"
            ")\n"
            "report = json.loads(report_path.read_text(encoding='utf-8'))\n"
            "display(report)\n"
            "print(f'Relatório: {report_path}')"
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "02_deep_res_unet_migration.ipynb"},
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
    output = Path("notebooks/02_deep_res_unet_migration.ipynb")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
