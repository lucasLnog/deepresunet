#!/usr/bin/env python3
"""Gera o notebook didático da auditoria visual de augmentation da etapa de aumento de dados."""

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
            "# Auditoria visual de data augmentation\n\n"
            "Este notebook verifica se A1 e A2 são **plausíveis e corretamente pareados** antes "
            "de qualquer treinamento. Ele não escolhe o melhor método e abre somente oito imagens "
            "fixas do treino. Validação, teste e FUSeg permanecem fechados."
        ),
        markdown(
            (
                '## Critérios da auditoria\n'
                '\n'
                'A auditoria inspeciona a plausibilidade das transformações, o alinhamento '
                'entre imagem e máscara e a retenção do primeiro plano.'
            )
        ),
        markdown(
            "## Braços pré-registrados\n\n"
            "- **A0:** sem augmentation.\n"
            "- **A1:** flips e affine moderado (rotação, escala e translação), sempre aplicado de "
            "forma idêntica à imagem e à máscara.\n"
            "- **A2:** A1 mais, com probabilidade 0,5, exatamente uma alteração fotométrica leve.\n\n"
            "Alterações fotométricas atingem somente a imagem; a máscara não muda."
        ),
        markdown("## 1. Montar o Drive e localizar o projeto"),
        code(
            "from pathlib import Path\nimport os, shutil, subprocess, sys\n\n"
            "try:\n    from google.colab import drive\n    drive.mount('/content/drive')\n"
            "except ImportError:\n    print('Execução fora do Colab.')\n\n"
            "candidates = [Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'), "
            "Path('/content/drive/MyDrive/Projeto_TCC_Lucas')]\n"
            "PROJECT_ROOT = next((p for p in candidates if p.is_dir()), Path.cwd())\n"
            "CONFIG_RELATIVE = Path('configs/augmentation_audit.yaml')\n"
            "if not (PROJECT_ROOT / CONFIG_RELATIVE).is_file():\n"
            "    raise FileNotFoundError('Configuração da auditoria de aumento de dados não encontrada.')\n"
            "os.chdir(PROJECT_ROOT)\nSOURCE_ROOT = PROJECT_ROOT / 'src'\n"
            "os.environ['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
            "if str(SOURCE_ROOT) not in sys.path: sys.path.insert(0, str(SOURCE_ROOT))\n"
            "print('Projeto:', PROJECT_ROOT)"
        ),
        markdown("## 2. Instalar o ambiente registrado"),
        code(
            "result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', "
            "'requirements-colab.txt'], text=True, capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown("## 3. Copiar os dados para o disco temporário do Colab"),
        code(
            "DATA_ROOT = Path('/content/tcc_data') if Path('/content').is_dir() else PROJECT_ROOT\n"
            "if DATA_ROOT != PROJECT_ROOT:\n"
            "    shutil.copytree(PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
            "print('Dados:', DATA_ROOT / 'Dataset')"
        ),
        markdown(
            "## 4. Conferir o protocolo antes de ver as imagens\n\n"
            "Esta célula confirma que as faixas já estavam registradas e que a auditoria só pode ler treino."
        ),
        code(
            "import yaml\nfrom IPython.display import display\n"
            "CONFIG = yaml.safe_load((PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding='utf-8'))\n"
            "display(CONFIG)\n"
            "assert CONFIG['runtime']['allowed_split'] == 'train'\n"
            "assert CONFIG['runtime']['model_training_allowed'] is False\n"
            "assert CONFIG['pipelines']['A1']['affine']['shear_degrees'] == [0.0, 0.0]"
        ),
        markdown(
            "## 5. Testar as funções isoladamente\n\n"
            "Os testes confirmam reprodutibilidade, máscara binária, mesma dimensão entre imagem e "
            "máscara, proibição de shear e no máximo uma transformação fotométrica em A2."
        ),
        code(
            "test_env = os.environ.copy()\n"
            "test_env['PYTHONPATH'] = str(SOURCE_ROOT) + os.pathsep + test_env.get('PYTHONPATH', '')\n"
            "result = subprocess.run([sys.executable, '-m', 'pytest', '-q', "
            "'tests/test_augmentation.py'], cwd=PROJECT_ROOT, env=test_env, text=True, capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown(
            "## 6. Gerar os painéis e diagnósticos\n\n"
            "Cada linha contém um caso fixo. Há uma imagem original, três sorteios de A1 e três de "
            "A2. O verde mostra onde a máscara ficou depois da transformação pareada."
        ),
        code(
            "command = [sys.executable, 'scripts/audit_augmentation.py', "
            "'--project-root', str(PROJECT_ROOT), '--data-root', str(DATA_ROOT), "
            "'--config', str(CONFIG_RELATIVE)]\n"
            "process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, "
            "text=True, cwd=PROJECT_ROOT, env=test_env)\n"
            "for line in process.stdout: print(line, end='')\n"
            "returncode = process.wait(); print('returncode=', returncode)\n"
            "if returncode != 0: raise subprocess.CalledProcessError(returncode, command)"
        ),
        markdown("## 7. Visualizar e interpretar as saídas"),
        code(
            "from PIL import Image\nimport pandas as pd\n"
            "AUDIT_ROOT = PROJECT_ROOT / CONFIG['runtime']['output_root']\n"
            "OUTPUT_DIR = max((p for p in AUDIT_ROOT.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)\n"
            "print('Saídas:', OUTPUT_DIR)\n"
            "display(Image.open(OUTPUT_DIR / 'augmentation_panel.png'))\n"
            "display(Image.open(OUTPUT_DIR / 'augmentation_mask_panel.png'))\n"
            "diagnostics = pd.read_csv(OUTPUT_DIR / 'augmentation_diagnostics.csv')\n"
            "display(diagnostics.groupby('pipeline').mean(numeric_only=True))\n"
            "display(diagnostics[diagnostics['requires_review'] == True])"
        ),
        markdown(
            "## O que observar\n\n"
            "1. A máscara verde deve acompanhar exatamente a ferida após flips/affine.\n"
            "2. Nenhuma máscara deve virar cinza: seus valores continuam binários.\n"
            "3. A ferida não deve ser apagada ou cortada de modo severo.\n"
            "4. A2 pode mudar cor/luz levemente, sem criar cores clinicamente absurdas.\n"
            "5. Bordas pretas pequenas podem surgir pela rotação; valores extensos merecem revisão.\n\n"
            "Uma variante visualmente bonita não é evidência de melhor segmentação."
        ),
        markdown(
            "## Artefatos da execução\n\n"
            "Arquive `augmentation_audit_summary.json`, `augmentation_diagnostics.csv`, "
            "`augmentation_panel.png` e `augmentation_mask_panel.png`. Só depois da revisão "
            "conjunta liberaremos o treinamento A0/A1/A2."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "10_augmentation_audit.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path(__file__).resolve().parents[1] / "notebooks/10_augmentation_audit.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
