#!/usr/bin/env python3
"""Gera o notebook didático do baseline D0 para Google Colab."""

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
            "# D0: baseline central da DeepResUNet\n\n"
            "Este notebook treina a **DeepResUNet sem melhorias**. D0 será o ponto zero "
            "para pré-processamento, augmentation, transfer learning, fine-tuning e SAM.\n\n"
            "Somente treino e validação são acessados. Teste interno e FUSeg permanecem fechados."
        ),
        markdown(
            "## Mapa da etapa\n\n"
            "1. conferir configuração e casos visuais fixos;\n"
            "2. fazer preflight de memória e gradientes;\n"
            "3. treinar com checkpoint e retomada;\n"
            "4. congelar a melhor época pelo Dice macro em threshold 0,5;\n"
            "5. calibrar threshold usando somente a validação;\n"
            "6. gerar curvas, métricas por imagem e painel fixo."
        ),
        markdown("## 1. Montar o Drive e localizar o projeto"),
        code(
            "from pathlib import Path\nimport os\n\n"
            "try:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n"
            "except ImportError:\n"
            "    print('Execução fora do Colab.')\n\n"
            "candidates = [Path('/content/drive/MyDrive/TCC/Projeto_TCC_Lucas'), "
            "Path('/content/drive/MyDrive/Projeto_TCC_Lucas')]\n"
            "PROJECT_ROOT = next((p for p in candidates if p.is_dir()), Path.cwd())\n"
            "if not (PROJECT_ROOT / 'configs/baseline_d0.yaml').is_file():\n"
            "    raise FileNotFoundError('Configuração do D0 não encontrada.')\n"
            "os.chdir(PROJECT_ROOT)\nprint('Projeto:', PROJECT_ROOT)"
        ),
        markdown("## 2. Instalar dependências registradas"),
        code(
            "import subprocess, sys\n"
            "result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', "
            "'-r', 'requirements-colab.txt'], text=True, capture_output=True)\n"
            "print(result.stdout); print(result.stderr); result.check_returncode()"
        ),
        markdown(
            "## 3. Usar cache temporário dos dados\n\n"
            "Os resultados continuam no Drive. Apenas a leitura das imagens é feita no disco "
            "temporário para reduzir latência."
        ),
        code(
            "import shutil\n"
            "DATA_ROOT = Path('/content/tcc_data') if Path('/content').is_dir() else PROJECT_ROOT\n"
            "if DATA_ROOT != PROJECT_ROOT:\n"
            "    shutil.copytree(PROJECT_ROOT / 'Dataset', DATA_ROOT / 'Dataset', dirs_exist_ok=True)\n"
            "print('Dados:', DATA_ROOT / 'Dataset')"
        ),
        markdown("## 4. Importar funções e exibir a configuração congelada"),
        code(
            "import csv, inspect, json, time, yaml\n"
            "import matplotlib.pyplot as plt\nimport pandas as pd\nimport torch\n"
            "from PIL import Image\nfrom IPython.display import display\n\n"
            "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
            "from tcc_segmentation.data import DevelopmentDataset, ResizePad, create_development_dataloader\n"
            "from tcc_segmentation.baseline_training import (\n"
            "    aggregate_rows, fixed_subset, make_model, preflight_batch_size,\n"
            "    save_fixed_validation_panel, subgroup_summary, sweep_validation_thresholds,\n"
            "    thresholds_from_config, train_d0, write_csv, write_json,\n"
            ")\n"
            "from tcc_segmentation.models import DeepResUNet, count_parameters\n"
            "from tcc_segmentation.training import load_training_checkpoint, seed_everything\n\n"
            "CONFIG = yaml.safe_load((PROJECT_ROOT / 'configs/baseline_d0.yaml').read_text(encoding='utf-8'))\n"
            "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
            "display(CONFIG)\nprint('Device:', DEVICE)"
        ),
        markdown(
            "## 5. Criar treino e validação protegidos\n\n"
            "Os manifestos têm hashes aprovados. A classe rejeita nomes de teste, avaliação "
            "externa e datasets diferentes do DFUC2022."
        ),
        code(
            "transform = ResizePad(CONFIG['data']['height'], CONFIG['data']['width'])\n"
            "train_dataset = DevelopmentDataset(\n"
            "    PROJECT_ROOT / CONFIG['data']['train_manifest'], DATA_ROOT, 'train',\n"
            "    transform=transform, mask_threshold=CONFIG['data']['mask_threshold'])\n"
            "validation_dataset = DevelopmentDataset(\n"
            "    PROJECT_ROOT / CONFIG['data']['validation_manifest'], DATA_ROOT, 'validation',\n"
            "    transform=transform, mask_threshold=CONFIG['data']['mask_threshold'])\n"
            "split_summary = json.loads((PROJECT_ROOT / 'manifests/splits/split_summary.json').read_text())\n"
            "MANIFEST_HASHES = {k: split_summary['manifest_hashes'][k] for k in ('train', 'validation')}\n"
            "display({'treino': len(train_dataset), 'validação': len(validation_dataset),\n"
            "         'teste_acessado': False, 'FUSeg_acessado': False})"
        ),
        markdown(
            "## 6. Visualizar os casos fixados antes do modelo\n\n"
            "Eles foram selecionados por tamanho da ferida e número de componentes. A "
            "seleção não conhece nenhuma predição."
        ),
        code(
            "visual_dataset = fixed_subset(validation_dataset, CONFIG['visualization']['fixed_validation_ids'])\n"
            "fig, axes = plt.subplots(len(visual_dataset), 2, figsize=(10, 3 * len(visual_dataset)))\n"
            "for row in range(len(visual_dataset)):\n"
            "    sample = visual_dataset[row]\n"
            "    axes[row, 0].imshow(sample['image'].transpose(1, 2, 0))\n"
            "    axes[row, 0].set_title(sample['canonical_id'] + ' — imagem')\n"
            "    axes[row, 1].imshow(sample['mask'][0], cmap='gray', vmin=0, vmax=1)\n"
            "    axes[row, 1].set_title('referência')\n"
            "    axes[row, 0].axis('off'); axes[row, 1].axis('off')\n"
            "plt.tight_layout(); plt.show()"
        ),
        markdown(
            "## 7. Conferir a rede central e preparar a execução\n\n"
            "Para retomar após uma desconexão, substitua `RESUME_DIR = None` pelo caminho "
            "da execução já criada e execute novamente a partir desta célula."
        ),
        code(
            "model_preview = make_model(CONFIG)\n"
            "print(model_preview)\nprint('Parâmetros:', count_parameters(model_preview))\n"
            "assert isinstance(model_preview, DeepResUNet)\ndel model_preview\n\n"
            "# Exemplo: RESUME_DIR = PROJECT_ROOT / 'results/day5/D0/D0_seed42_...'\n"
            "RESUME_DIR = None\n"
            "if RESUME_DIR is None:\n"
            "    RUN_ID = time.strftime('D0_seed42_%Y%m%dT%H%M%SZ', time.gmtime())\n"
            "    OUTPUT_DIR = PROJECT_ROOT / CONFIG['runtime']['output_root'] / RUN_ID\n"
            "    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)\n"
            "    shutil.copy2(PROJECT_ROOT / 'configs/baseline_d0.yaml', OUTPUT_DIR / 'config_used.yaml')\n"
            "    RESUME = False\n"
            "else:\n"
            "    OUTPUT_DIR = Path(RESUME_DIR)\n    RUN_ID = OUTPUT_DIR.name\n    RESUME = True\n"
            "print('Execução:', RUN_ID)\nprint('Saídas:', OUTPUT_DIR)"
        ),
        markdown(
            "## 8. Preflight de memória, logits, loss e gradientes\n\n"
            "O modelo deste teste será descartado. Se batch 4 causar OOM, batch 2 será "
            "selecionado antes da época 1. Nenhuma métrica decide o batch."
        ),
        code(
            "if not RESUME:\n"
            "    SELECTED_BATCH, PREFLIGHT = preflight_batch_size(\n"
            "        train_dataset=train_dataset, config=CONFIG, device=DEVICE)\n"
            "    write_json(OUTPUT_DIR / 'preflight.json', PREFLIGHT)\n"
            "    display(PREFLIGHT)\n"
            "else:\n"
            "    PREFLIGHT = json.loads((OUTPUT_DIR / 'preflight.json').read_text())\n"
            "    SELECTED_BATCH = PREFLIGHT['selected_batch_size']\n"
            "    display({'retomada': True, 'batch': SELECTED_BATCH})\n"
            "write_json(OUTPUT_DIR / 'environment_report.json', {\n"
            "    'device': str(DEVICE), 'torch_version': torch.__version__,\n"
            "    'cuda_available': torch.cuda.is_available(),\n"
            "    'cuda_device_name': torch.cuda.get_device_name(DEVICE) if DEVICE.type == 'cuda' else None,\n"
            "    'train_records': len(train_dataset), 'validation_records': len(validation_dataset),\n"
            "    'protected_partitions_accessed': False,\n"
            "})"
        ),
        markdown(
            "## 9. Treinar D0\n\n"
            "Cada linha impressa corresponde a uma época. `best_checkpoint.pt` guarda o "
            "maior Dice macro em 0,5; `last_checkpoint.pt` permite retomada."
        ),
        code(
            "TRAINING_SUMMARY = train_d0(\n"
            "    train_dataset=train_dataset, validation_dataset=validation_dataset,\n"
            "    config=CONFIG, output_dir=OUTPUT_DIR, run_id=RUN_ID, device=DEVICE,\n"
            "    manifest_hashes=MANIFEST_HASHES, resume=RESUME,\n"
            "    preflight_result=None if RESUME else (SELECTED_BATCH, PREFLIGHT),\n"
            ")\n"
            "display(TRAINING_SUMMARY)"
        ),
        markdown("## 10. Curvas de treinamento e melhor época"),
        code(
            "history_df = pd.read_csv(OUTPUT_DIR / 'history.csv')\n"
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n"
            "axes[0].plot(history_df.epoch, history_df.train_loss, label='treino')\n"
            "axes[0].plot(history_df.epoch, history_df.validation_loss, label='validação')\n"
            "axes[0].set(title='Loss', xlabel='Época', ylabel='Loss'); axes[0].legend()\n"
            "axes[1].plot(history_df.epoch, history_df.validation_dice_at_0_5)\n"
            "axes[1].axvline(TRAINING_SUMMARY['best_epoch'], color='black', linestyle='--', label='melhor')\n"
            "axes[1].set(title='Dice macro de validação em 0,5', xlabel='Época', ylabel='Dice'); axes[1].legend()\n"
            "axes[2].plot(history_df.epoch, history_df.learning_rate)\n"
            "axes[2].set(title='Learning rate', xlabel='Época', ylabel='LR')\n"
            "plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'training_curves.png', dpi=170); plt.show()"
        ),
        markdown(
            (
                '## 11. Recarregar a melhor época e calibrar o threshold\n'
                '\n'
                'Com o checkpoint fixado, a busca em grade usa os mesmos mapas de '
                'probabilidade da validação para selecionar o limiar com maior Dice macro.'
            )
        ),
        code(
            "seed_everything(CONFIG['seed'])\nmodel = make_model(CONFIG).to(DEVICE)\n"
            "best_payload = load_training_checkpoint(\n"
            "    OUTPUT_DIR / 'best_checkpoint.pt', model=model, map_location=DEVICE, restore_rng=False)\n"
            "validation_loader = create_development_dataloader(\n"
            "    validation_dataset, TRAINING_SUMMARY['selected_batch_size'], False, CONFIG['seed'],\n"
            "    CONFIG['training']['num_workers'])\n"
            "metadata_by_id = {row['canonical_id']: row for row in validation_dataset.rows}\n"
            "sweep = sweep_validation_thresholds(\n"
            "    model=model, loader=validation_loader, device=DEVICE,\n"
            "    thresholds=thresholds_from_config(CONFIG), metadata_by_id=metadata_by_id)\n"
            "write_csv(OUTPUT_DIR / 'threshold_curve.csv', sweep['curve'])\n"
            "write_csv(OUTPUT_DIR / 'validation_per_image_at_0_5.csv', sweep['per_image_at_0_5'])\n"
            "write_csv(OUTPUT_DIR / 'validation_per_image_selected.csv', sweep['per_image_selected'])\n"
            "metrics_05 = aggregate_rows(sweep['per_image_at_0_5'])\n"
            "metrics_selected = aggregate_rows(sweep['per_image_selected'])\n"
            "write_json(OUTPUT_DIR / 'validation_metrics_at_0_5.json', metrics_05)\n"
            "write_json(OUTPUT_DIR / 'validation_metrics_selected.json', metrics_selected)\n"
            "write_csv(OUTPUT_DIR / 'validation_by_area_bin.csv', subgroup_summary(sweep['per_image_selected']))\n"
            "selection = {\n"
            "    'checkpoint_epoch': int(best_payload['epoch']),\n"
            "    'checkpoint_rule': 'highest_validation_macro_dice_at_threshold_0.5',\n"
            "    'threshold': sweep['selected_threshold'],\n"
            "    'threshold_rule': 'highest_validation_macro_dice_then_closest_to_0.5',\n"
            "    'grid': CONFIG['threshold_selection'],\n"
            "    'metrics_at_0_5': metrics_05, 'metrics_selected': metrics_selected,\n"
            "    'protected_partitions_accessed': False,\n"
            "}\nwrite_json(OUTPUT_DIR / 'threshold_selection.json', selection)\n"
            "display({'melhor_época': best_payload['epoch'], 'threshold': sweep['selected_threshold'],\n"
            "         'Dice_0.5': metrics_05['macro']['dice']['mean'],\n"
            "         'Dice_selecionado': metrics_selected['macro']['dice']['mean']})"
        ),
        markdown("## 12. Curvas de threshold"),
        code(
            "curve_df = pd.DataFrame(sweep['curve'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n"
            "curve_df.plot(x='threshold', y=['dice', 'iou'], ax=axes[0])\n"
            "curve_df.plot(x='threshold', y=['precision', 'recall'], ax=axes[1])\n"
            "for axis in axes:\n"
            "    axis.axvline(sweep['selected_threshold'], color='black', linestyle='--', label='selecionado')\n"
            "    axis.legend()\n"
            "axes[0].set_title('Sobreposição'); axes[1].set_title('Precisão e recall')\n"
            "plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'threshold_curves.png', dpi=170); plt.show()"
        ),
        markdown(
            "## 13. Painel fixo e mapa de erros\n\n"
            "No mapa de erros: verde = verdadeiro positivo, laranja = falso positivo e "
            "vermelho = falso negativo."
        ),
        code(
            "panel_path = save_fixed_validation_panel(\n"
            "    model=model, dataset=visual_dataset, device=DEVICE,\n"
            "    threshold=sweep['selected_threshold'], output_path=OUTPUT_DIR / 'fixed_validation_panel.png')\n"
            "display(Image.open(panel_path))"
        ),
        markdown("## 14. Resumo final e arquivos para análise conjunta"),
        code(
            "FINAL_SUMMARY = {\n"
            "    'status': 'completed', 'scientific_role': 'central_baseline_development',\n"
            "    'run_id': RUN_ID, 'seed': CONFIG['seed'],\n"
            "    'model_parameters': count_parameters(model),\n"
            "    'training': TRAINING_SUMMARY, 'selection': selection,\n"
            "    'partitions_accessed': ['train', 'validation'],\n"
            "    'protected_partitions_accessed': False,\n"
            "}\n"
            "write_json(OUTPUT_DIR / 'final_summary.json', FINAL_SUMMARY)\n"
            "display(FINAL_SUMMARY)\nprint('Arquivos:', OUTPUT_DIR)"
        ),
        markdown(
            "## Não avance automaticamente\n\n"
            "Arquive `final_summary.json`, `history.csv`, `training_curves.png`, "
            "`threshold_curve.csv`, `threshold_curves.png`, `threshold_selection.json`, "
            "`fixed_validation_panel.png` e `validation_by_area_bin.csv`.\n\n"
            "Esses resultados devem ser interpretados antes de iniciar qualquer ablação. "
            "Teste interno e FUSeg continuam fechados."
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": "07_baseline_d0_training.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = Path(__file__).resolve().parents[1] / "notebooks/07_baseline_d0_training.ipynb"
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
