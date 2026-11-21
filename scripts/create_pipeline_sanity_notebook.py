#!/usr/bin/env python3
"""Gera o notebook didático e sequencial do gate técnico da verificação do pipeline."""

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
            "# Pipeline e testes de sanidade da DeepResUNet\n\n"
            "**Objetivo:** verificar visualmente e numericamente o pipeline inteiro antes "
            "de qualquer baseline completo.\n\n"
            "Os resultados deste notebook são **diagnósticos técnicos**, não desempenho "
            "científico. Somente treino e validação são permitidos; teste e FUSeg ficam fechados."
        ),
        markdown(
            "## Mapa da execução\n\n"
            "`dados → DeepResUNet → logits → BCE+Dice → atualização`\n\n"
            "`logits → sigmoid → probabilidade → threshold 0,5 → máscara → métricas`\n\n"
            "Primeiro a rede tentará memorizar quatro imagens. Depois faremos uma época "
            "curta em 16 imagens de treino e 8 de validação e testaremos o checkpoint."
        ),
        markdown("## 1. Montar o Drive e localizar o projeto"),
        code(
            "from pathlib import Path\nimport os\n\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "except ImportError:\n"
            "    print('Execução fora do Colab.')\n\n"
            "candidates = [\n"
            "    Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'),\n"
            "    Path('/content/drive/MyDrive/Projeto_TCC_Lucas'),\n"
            "]\n"
            "PROJECT_ROOT = next((path for path in candidates if path.is_dir()), Path.cwd())\n"
            "if not (PROJECT_ROOT / 'configs/pipeline_sanity.yaml').is_file():\n"
            "    raise FileNotFoundError('Projeto ou configuração da verificação do pipeline não encontrados.')\n"
            "os.chdir(PROJECT_ROOT)\nprint('Projeto:', PROJECT_ROOT)"
        ),
        markdown("## 2. Instalar as dependências registradas"),
        code(
            "import subprocess, sys\n"
            "result = subprocess.run(\n"
            "    [sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'],\n"
            "    text=True, capture_output=True\n"
            ")\n"
            "print(result.stdout)\nprint(result.stderr)\nresult.check_returncode()"
        ),
        markdown(
            "## 3. Copiar os dados para o disco temporário\n\n"
            "O Drive guarda os arquivos permanentes. A cópia temporária acelera a leitura "
            "sem alterar os dados originais."
        ),
        code(
            "import shutil\n"
            "DATA_ROOT = Path('/content/tcc_data') if Path('/content').is_dir() else PROJECT_ROOT\n"
            "if DATA_ROOT != PROJECT_ROOT:\n"
            "    shutil.copytree(PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
            "print('Raiz usada para leitura:', DATA_ROOT)"
        ),
        markdown("## 4. Carregar configuração, bibliotecas e fixar a seed"),
        code(
            "import inspect, json, time, yaml\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "import torch\n"
            "from PIL import Image\n"
            "from IPython.display import display\n\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
            "from tcc_segmentation.data import DevelopmentDataset, ResizePad, create_development_dataloader\n"
            "from tcc_segmentation.pipeline_sanity import (\n"
            "    batch_diagnostics, collect_predictions, deterministic_prefix_indices,\n"
            "    indices_for_canonical_ids, prediction_max_abs_difference,\n"
            "    run_overfit_sanity, run_smoke_sanity, save_before_after_panel,\n"
            "    save_history_csv, write_json,\n"
            ")\n"
            "from tcc_segmentation.models import DeepResUNet, build_segmentation_model, count_parameters\n"
            "from tcc_segmentation.training import (\n"
            "    BCEDiceLoss, load_training_checkpoint, save_training_checkpoint, seed_everything,\n"
            ")\n\n"
            "CONFIG = yaml.safe_load((PROJECT_ROOT / 'configs/pipeline_sanity.yaml').read_text(encoding='utf-8'))\n"
            "seed_everything(int(CONFIG['seed']))\n"
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "RUN_ID = time.strftime('day4_%Y%m%dT%H%M%SZ', time.gmtime())\n"
            "OUTPUT_DIR = PROJECT_ROOT / CONFIG['runtime']['output_root'] / RUN_ID\n"
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=False)\n"
            "print('Device:', DEVICE)\nprint('Saídas:', OUTPUT_DIR)\n"
            "display(CONFIG)"
        ),
        markdown(
            "## 5. Criar datasets permitidos\n\n"
            "A classe abaixo rejeita manifestos protegidos e confere o hash dos manifestos "
            "de treino e validação. Nenhum caminho de teste aparece nesta configuração."
        ),
        code(
            "transform = ResizePad(height=CONFIG['data']['height'], width=CONFIG['data']['width'])\n"
            "train_dataset = DevelopmentDataset(\n"
            "    PROJECT_ROOT / CONFIG['data']['train_manifest'], DATA_ROOT, 'train',\n"
            "    transform=transform, mask_threshold=CONFIG['data']['mask_threshold'],\n"
            ")\n"
            "validation_dataset = DevelopmentDataset(\n"
            "    PROJECT_ROOT / CONFIG['data']['validation_manifest'], DATA_ROOT, 'validation',\n"
            "    transform=transform, mask_threshold=CONFIG['data']['mask_threshold'],\n"
            ")\n"
            "print({'treino': len(train_dataset), 'validação': len(validation_dataset),\n"
            "       'teste_acessado': False, 'FUSeg_acessado': False})"
        ),
        markdown("## 6. Visualizar entradas e referências antes do treinamento"),
        code(
            "ids = CONFIG['overfit']['canonical_ids']\n"
            "indices = indices_for_canonical_ids(train_dataset, ids)\n"
            "overfit_dataset = torch.utils.data.Subset(train_dataset, indices)\n"
            "overfit_loader = create_development_dataloader(\n"
            "    overfit_dataset, batch_size=CONFIG['overfit']['batch_size'], shuffle=False,\n"
            "    seed=CONFIG['seed'], num_workers=0,\n"
            ")\n"
            "batch = next(iter(overfit_loader))\n"
            "diagnostics = batch_diagnostics(batch)\n"
            "display(diagnostics)\n\n"
            "fig, axes = plt.subplots(4, 2, figsize=(10, 14))\n"
            "for row, sample_index in enumerate(range(4)):\n"
            "    sample = overfit_dataset[sample_index]\n"
            "    axes[row, 0].imshow(sample['image'].transpose(1, 2, 0))\n"
            "    axes[row, 0].set_title(sample['canonical_id'] + ' — imagem')\n"
            "    axes[row, 1].imshow(sample['mask'][0], cmap='gray', vmin=0, vmax=1)\n"
            "    axes[row, 1].set_title('máscara canônica')\n"
            "    axes[row, 0].axis('off'); axes[row, 1].axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        markdown(
            "## 7. Conferir a arquitetura central\n\n"
            "A sigmoid foi retirada apenas da saída final. Os attention gates continuam "
            "usando sigmoid internamente para calcular seus coeficientes de atenção."
        ),
        code(
            "print(inspect.getsource(DeepResUNet))\n"
            "model = build_segmentation_model(CONFIG['model']).to(DEVICE)\n"
            "print('Parâmetros:', count_parameters(model))"
        ),
        markdown("## 8. Fazer uma passagem e observar logits, probabilidades e loss"),
        code(
            "images = batch['image'].to(DEVICE)\nmasks = batch['mask'].to(DEVICE)\n"
            "model.eval()\n"
            "with torch.inference_mode():\n"
            "    logits = model(images).float()\n"
            "    probabilities = torch.sigmoid(logits)\n"
            "criterion = BCEDiceLoss(\n"
            "    CONFIG['loss']['bce_weight'], CONFIG['loss']['dice_weight'], CONFIG['loss']['smooth']\n"
            ")\n"
            "components = criterion.components(logits, masks)\n"
            "display({\n"
            "    'shape_logits': list(logits.shape), 'shape_máscaras': list(masks.shape),\n"
            "    'logit_min': float(logits.min()), 'logit_max': float(logits.max()),\n"
            "    'probabilidade_min': float(probabilities.min()),\n"
            "    'probabilidade_max': float(probabilities.max()),\n"
            "    'loss_total': float(components['total']), 'bce': float(components['bce']),\n"
            "    'dice_loss': float(components['dice_loss']), 'tudo_finito': bool(torch.isfinite(logits).all()),\n"
            "})"
        ),
        markdown(
            "## 9. Registrar as predições antes e executar o overfit pequeno\n\n"
            "A execução pode parar antes de 300 épocas se a loss cair pelo menos 80% e "
            "o Dice técnico médio atingir 0,90. Isso é um gate de depuração, não seleção de modelo."
        ),
        code(
            "seed_everything(CONFIG['seed'])\n"
            "model = build_segmentation_model(CONFIG['model']).to(DEVICE)\n"
            "optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['overfit']['learning_rate'])\n"
            "before = collect_predictions(model, overfit_loader, DEVICE)\n"
            "history, overfit_accepted = run_overfit_sanity(\n"
            "    model=model, loader=overfit_loader, criterion=criterion, optimizer=optimizer,\n"
            "    device=DEVICE, epochs=CONFIG['overfit']['epochs'], threshold=0.5,\n"
            "    min_loss_reduction_fraction=CONFIG['overfit']['early_acceptance']['min_loss_reduction_fraction'],\n"
            "    min_macro_dice=CONFIG['overfit']['early_acceptance']['min_macro_dice'],\n"
            ")\n"
            "after = collect_predictions(model, overfit_loader, DEVICE)\n"
            "history_df = pd.DataFrame(history)\n"
            "display(history_df.tail())\nprint('Gate do overfit aprovado:', overfit_accepted)"
        ),
        markdown("## 10. Curvas e comparação visual antes/depois"),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "axes[0].plot(history_df['epoch'], history_df['loss'])\n"
            "axes[0].set(title='Loss — overfit pequeno', xlabel='Época', ylabel='Loss')\n"
            "axes[1].plot(history_df['epoch'], history_df['macro_dice_at_0_5'])\n"
            "axes[1].set(title='Dice técnico em threshold 0,5', xlabel='Época', ylabel='Dice')\n"
            "plt.tight_layout(); plt.show()\n"
            "panel_path = save_before_after_panel(before, after, OUTPUT_DIR / 'overfit_before_after.png', 0.5)\n"
            "display(Image.open(panel_path))\n"
            "save_history_csv(history, OUTPUT_DIR / 'overfit_history.csv')"
        ),
        markdown(
            "## 11. Smoke test curto com modelo reinicializado\n\n"
            "O modelo é recriado para que a memorização anterior não contamine este teste. "
            "As métricas abaixo apenas confirmam que treino e validação executam."
        ),
        code(
            "smoke = CONFIG['smoke']\n"
            "smoke_train = torch.utils.data.Subset(\n"
            "    train_dataset, deterministic_prefix_indices(len(train_dataset), smoke['train_samples']))\n"
            "smoke_validation = torch.utils.data.Subset(\n"
            "    validation_dataset, deterministic_prefix_indices(len(validation_dataset), smoke['validation_samples']))\n"
            "smoke_train_loader = create_development_dataloader(\n"
            "    smoke_train, smoke['batch_size'], True, CONFIG['seed'], 0)\n"
            "smoke_validation_loader = create_development_dataloader(\n"
            "    smoke_validation, smoke['batch_size'], False, CONFIG['seed'], 0)\n"
            "seed_everything(CONFIG['seed'])\n"
            "smoke_model = build_segmentation_model(CONFIG['model']).to(DEVICE)\n"
            "smoke_optimizer = torch.optim.Adam(smoke_model.parameters(), lr=smoke['learning_rate'])\n"
            "smoke_result = run_smoke_sanity(\n"
            "    model=smoke_model, train_loader=smoke_train_loader,\n"
            "    validation_loader=smoke_validation_loader, criterion=criterion,\n"
            "    optimizer=smoke_optimizer, device=DEVICE, threshold=0.5)\n"
            "display({\n"
            "    'resultado_científico': False,\n"
            "    'loss_treino': smoke_result['train']['loss'],\n"
            "    'loss_validação': smoke_result['validation']['loss'],\n"
            "    'métricas_técnicas_validação': {k: v['mean'] for k, v in smoke_result['validation']['metrics'].items() if isinstance(v, dict)},\n"
            "})"
        ),
        markdown("## 12. Salvar, recarregar e comparar o checkpoint"),
        code(
            "predictions_before_reload = collect_predictions(smoke_model, smoke_validation_loader, DEVICE)\n"
            "split_summary = json.loads((PROJECT_ROOT / 'manifests/splits/split_summary.json').read_text())\n"
            "checkpoint_path = OUTPUT_DIR / 'smoke_checkpoint.pt'\n"
            "save_training_checkpoint(\n"
            "    checkpoint_path, kind='last', epoch=1,\n"
            "    best_score=smoke_result['validation']['metrics']['dice']['mean'],\n"
            "    model=smoke_model, optimizer=smoke_optimizer, config=CONFIG, run_id=RUN_ID,\n"
            "    manifest_hashes={k: split_summary['manifest_hashes'][k] for k in ('train', 'validation')},\n"
            "    extra={'scientific_result': False, 'protected_partitions_accessed': False},\n"
            ")\n"
            "reloaded_model = build_segmentation_model(CONFIG['model']).to(DEVICE)\n"
            "reloaded_optimizer = torch.optim.Adam(reloaded_model.parameters(), lr=smoke['learning_rate'])\n"
            "load_training_checkpoint(\n"
            "    checkpoint_path, model=reloaded_model, optimizer=reloaded_optimizer,\n"
            "    map_location=DEVICE, restore_rng=False)\n"
            "predictions_after_reload = collect_predictions(reloaded_model, smoke_validation_loader, DEVICE)\n"
            "roundtrip_error = prediction_max_abs_difference(predictions_before_reload, predictions_after_reload)\n"
            "print('Maior diferença absoluta entre logits:', roundtrip_error)\n"
            "print('Checkpoint reproduz exatamente:', roundtrip_error <= 1e-6)"
        ),
        markdown("## 13. Gerar o relatório técnico final"),
        code(
            "report = {\n"
            "    'status': 'passed' if overfit_accepted and roundtrip_error <= 1e-6 else 'review',\n"
            "    'scientific_result': False, 'run_id': RUN_ID, 'device': str(DEVICE),\n"
            "    'model_parameters': count_parameters(smoke_model),\n"
            "    'overfit': {'canonical_ids': ids, 'epochs_executed': len(history),\n"
            "                'accepted': overfit_accepted, 'first': history[0], 'last': history[-1]},\n"
            "    'smoke': {'train_samples': len(smoke_train), 'validation_samples': len(smoke_validation),\n"
            "              'checkpoint_roundtrip_max_abs_logit_error': roundtrip_error},\n"
            "    'partitions_accessed': ['train', 'validation'],\n"
            "    'protected_partitions_accessed': False,\n"
            "}\n"
            "write_json(diagnostics, OUTPUT_DIR / 'batch_diagnostics.json')\n"
            "write_json(smoke_result, OUTPUT_DIR / 'smoke_summary.json')\n"
            "write_json(report, OUTPUT_DIR / 'day4_sanity_report.json')\n"
            "display(report)\nprint('Arquivos salvos em:', OUTPUT_DIR)"
        ),
        markdown(
            (
                '## Como interpretar\n'
                '\n'
                '- `passed`: o pipeline está tecnicamente apto a avançar para o baseline.\n'
                '- `review`: a verificação requer inspeção das curvas, do painel, dos '
                'gradientes ou do checkpoint.\n'
                '\n'
                'Em nenhum caso as métricas deste notebook representam desempenho final da '
                'rede. Arquive `day4_sanity_report.json`, `overfit_history.csv`, '
                '`overfit_before_after.png` e `smoke_summary.json` para a auditoria da '
                'execução.'
            )
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "06_pipeline_sanity.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path(__file__).resolve().parents[1] / "notebooks/06_pipeline_sanity.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
