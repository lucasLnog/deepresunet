#!/usr/bin/env python3
"""Treina D-P1 ou D-P2 isoladamente usando somente treino e validação."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import matplotlib.pyplot as plt
    import torch
    import yaml

    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else root
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root / "src"))

    from tcc_segmentation.data import (  # noqa: PLC0415
        DevelopmentDataset,
        create_development_dataloader,
    )
    from tcc_segmentation.baseline_training import (  # noqa: PLC0415
        aggregate_rows,
        fixed_subset,
        make_model,
        save_fixed_validation_panel,
        subgroup_summary,
        sweep_validation_thresholds,
        thresholds_from_config,
        train_d0,
        write_csv,
        write_json,
    )
    from tcc_segmentation.preprocessing_experiment import (  # noqa: PLC0415
        make_preprocessing_pair_transform,
        validate_preprocessing_training_config,
    )
    from tcc_segmentation.models import count_parameters  # noqa: PLC0415
    from tcc_segmentation.training import (  # noqa: PLC0415
        load_training_checkpoint,
        seed_everything,
    )

    validate_preprocessing_training_config(config)
    run_prefix = str(config["runtime"]["run_prefix"])
    if args.resume_dir:
        output_dir = args.resume_dir.expanduser().resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        if not output_dir.name.startswith(run_prefix + "_"):
            raise ValueError(
                f"Diretório de retomada não pertence a {run_prefix}: {output_dir.name}"
            )
        stored_config_path = output_dir / "config_used.yaml"
        if not stored_config_path.is_file():
            raise FileNotFoundError(stored_config_path)
        stored_config = yaml.safe_load(stored_config_path.read_text(encoding="utf-8"))
        if stored_config != config:
            raise ValueError("Configuração atual difere da configuração da execução")
        run_id = output_dir.name
        resume = True
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{run_prefix}_{timestamp}"
        output_dir = root / config["runtime"]["output_root"] / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, output_dir / "config_used.yaml")
        resume = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = make_preprocessing_pair_transform(config)
    train_dataset = DevelopmentDataset(
        root / config["data"]["train_manifest"],
        data_root,
        expected_split="train",
        transform=transform,
        mask_threshold=int(config["data"]["mask_threshold"]),
    )
    validation_dataset = DevelopmentDataset(
        root / config["data"]["validation_manifest"],
        data_root,
        expected_split="validation",
        transform=transform,
        mask_threshold=int(config["data"]["mask_threshold"]),
    )
    split_summary = json.loads(
        (root / "manifests/splits/split_summary.json").read_text(encoding="utf-8")
    )
    manifest_hashes = {
        key: split_summary["manifest_hashes"][key] for key in ("train", "validation")
    }
    environment = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "experiment_id": config["experiment_id"],
        "color_preprocessing": config["data"]["color_preprocessing"],
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "protected_partitions_accessed": False,
    }
    write_json(output_dir / "environment_report.json", environment)

    training_summary = train_d0(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
        output_dir=output_dir,
        run_id=run_id,
        device=device,
        manifest_hashes=manifest_hashes,
        resume=resume,
    )

    seed_everything(int(config["seed"]))
    model = make_model(config).to(device)
    best_payload = load_training_checkpoint(
        output_dir / "best_checkpoint.pt",
        model=model,
        map_location=device,
        restore_rng=False,
    )
    validation_loader = create_development_dataloader(
        validation_dataset,
        batch_size=int(training_summary["selected_batch_size"]),
        shuffle=False,
        seed=int(config["seed"]),
        num_workers=int(config["training"]["num_workers"]),
    )
    metadata_by_id = {str(row["canonical_id"]): row for row in validation_dataset.rows}
    sweep = sweep_validation_thresholds(
        model=model,
        loader=validation_loader,
        device=device,
        thresholds=thresholds_from_config(config),
        metadata_by_id=metadata_by_id,
    )
    write_csv(output_dir / "threshold_curve.csv", sweep["curve"])
    write_csv(output_dir / "validation_per_image_at_0_5.csv", sweep["per_image_at_0_5"])
    write_csv(output_dir / "validation_per_image_selected.csv", sweep["per_image_selected"])
    metrics_at_0_5 = aggregate_rows(sweep["per_image_at_0_5"])
    metrics_selected = aggregate_rows(sweep["per_image_selected"])
    write_json(output_dir / "validation_metrics_at_0_5.json", metrics_at_0_5)
    write_json(output_dir / "validation_metrics_selected.json", metrics_selected)
    write_csv(
        output_dir / "validation_by_area_bin.csv",
        subgroup_summary(sweep["per_image_selected"]),
    )
    selection = {
        "experiment_id": config["experiment_id"],
        "color_preprocessing": config["data"]["color_preprocessing"],
        "checkpoint_epoch": int(best_payload["epoch"]),
        "checkpoint_rule": "highest_validation_macro_dice_at_threshold_0.5",
        "threshold": float(sweep["selected_threshold"]),
        "threshold_rule": "highest_validation_macro_dice_then_closest_to_0.5",
        "grid": config["threshold_selection"],
        "metrics_at_0_5": metrics_at_0_5,
        "metrics_selected": metrics_selected,
        "protected_partitions_accessed": False,
    }
    write_json(output_dir / "threshold_selection.json", selection)

    visual_dataset = fixed_subset(
        validation_dataset, config["visualization"]["fixed_validation_ids"]
    )
    panel_path = save_fixed_validation_panel(
        model=model,
        dataset=visual_dataset,
        device=device,
        threshold=float(sweep["selected_threshold"]),
        output_path=output_dir / "fixed_validation_panel.png",
    )

    history = list(csv.DictReader((output_dir / "history.csv").open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history], label="treino")
    axes[0].plot(epochs, [float(row["validation_loss"]) for row in history], label="validação")
    axes[0].set(title="Loss", xlabel="Época", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["validation_dice_at_0_5"]) for row in history])
    axes[1].axvline(int(best_payload["epoch"]), color="black", linestyle="--", label="melhor época")
    axes[1].set(title="Dice macro de validação em 0,5", xlabel="Época", ylabel="Dice")
    axes[1].legend()
    axes[2].plot(epochs, [float(row["learning_rate"]) for row in history])
    axes[2].set(title="Learning rate", xlabel="Época", ylabel="LR")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    threshold_values = [float(row["threshold"]) for row in sweep["curve"]]
    for metric in ("dice", "iou"):
        axes[0].plot(threshold_values, [float(row[metric]) for row in sweep["curve"]], label=metric)
    for metric in ("precision", "recall"):
        axes[1].plot(threshold_values, [float(row[metric]) for row in sweep["curve"]], label=metric)
    for axis in axes:
        axis.axvline(
            float(sweep["selected_threshold"]), color="black", linestyle="--", label="selecionado"
        )
        axis.set(xlabel="Threshold", ylabel="Média macro")
        axis.legend()
    axes[0].set_title("Sobreposição")
    axes[1].set_title("Precisão e recall")
    figure.tight_layout()
    figure.savefig(output_dir / "threshold_curves.png", dpi=170)
    plt.close(figure)

    final_summary = {
        "status": "completed",
        "scientific_role": str(config["scientific_role"]),
        "experiment_id": str(config["experiment_id"]),
        "color_preprocessing": str(config["data"]["color_preprocessing"]),
        "run_id": run_id,
        "seed": int(config["seed"]),
        "model_parameters": count_parameters(model),
        "training": training_summary,
        "selection": selection,
        "artifacts": {
            "training_curves": str(output_dir / "training_curves.png"),
            "threshold_curves": str(output_dir / "threshold_curves.png"),
            "fixed_validation_panel": str(panel_path),
        },
        "partitions_accessed": ["train", "validation"],
        "protected_partitions_accessed": False,
    }
    write_json(output_dir / "final_summary.json", final_summary)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
