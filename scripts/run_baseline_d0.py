#!/usr/bin/env python3
"""Treina e avalia um braço controlado baseado no D0, somente em treino/validação."""

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
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_d0.yaml"))
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

    def merge(left, right):
        result = dict(left)
        for key, value in right.items():
            result[key] = (
                merge(result[key], value)
                if key in result and isinstance(result[key], dict) and isinstance(value, dict)
                else value
            )
        return result

    def load_config(path: Path, seen=()):
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"Ciclo em base_config: {resolved}")
        current = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        base_name = current.pop("base_config", None)
        if base_name is None:
            return current
        base = load_config(root / str(base_name), (*seen, resolved))
        return merge(base, current)

    config = load_config(config_path)
    sys.path.insert(0, str(root / "src"))

    from tcc_segmentation.data import (  # noqa: PLC0415
        DevelopmentDataset,
        ResizePad,
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
    from tcc_segmentation.models import count_parameters  # noqa: PLC0415
    from tcc_segmentation.training import (  # noqa: PLC0415
        load_training_checkpoint,
        seed_everything,
    )

    run_prefix = str(config.get("runtime", {}).get("run_prefix", "D0_seed42"))
    if args.resume_dir:
        output_dir = args.resume_dir.expanduser().resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        if not output_dir.name.startswith(run_prefix + "_"):
            raise ValueError(f"Diretório de retomada não pertence a {run_prefix}")
        stored_config_path = output_dir / "config_used.yaml"
        if not stored_config_path.is_file():
            raise FileNotFoundError(stored_config_path)
        stored_config = yaml.safe_load(stored_config_path.read_text(encoding="utf-8"))
        if stored_config != config:
            raise ValueError("Configuração atual difere da configuração da execução")
        run_id = output_dir.name
        resume = True
    else:
        run_id = f"{run_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        output_dir = root / config["runtime"]["output_root"] / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config_used.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        resume = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmentation_name = str(config["data"].get("augmentation", "none"))
    is_augmentation_arm = augmentation_name in {"A1", "A2"}
    is_p1_a1_arm = str(config.get("experiment_id")) == "P1-A1"
    is_transfer_arm = str(config.get("experiment_id")) in {"DR-RT", "DR-IT"}
    is_fine_tuning_arm = str(config.get("experiment_id")) in {"DR-IF", "DR-IFT", "DR-IFT-B"}
    normalization = config["data"].get("normalization")
    if is_fine_tuning_arm:
        from tcc_segmentation.fine_tuning import (
            make_fine_tuning_transform,
            train_fine_tuning,
            validate_fine_tuning_config,
            validate_fine_tuning_pair,
        )
        from tcc_segmentation.transfer_learning import save_normalized_batch_panel

        a1_config = yaml.safe_load((root / "configs/augmentation_d0_a1.yaml").read_text(encoding="utf-8"))
        audit_config = yaml.safe_load(
            (root / "configs/augmentation_audit.yaml").read_text(encoding="utf-8")
        )
        validate_fine_tuning_config(config, a1_config=a1_config, audit_config=audit_config)
        if config["experiment_id"] != "DR-IFT-B":
            peer_name = (
                "fine_tuning_progressive.yaml" if config["experiment_id"] == "DR-IF" else "fine_tuning_frozen_encoder.yaml"
            )
            peer_raw = yaml.safe_load((root / "configs" / peer_name).read_text(encoding="utf-8"))
            peer_raw.pop("base_config", None)
            peer_config = merge(a1_config, peer_raw)
            # Replicações multisseed usam o contrato canônico do braço par, mas devem
            # compartilhar seed, agenda de aleatoriedade e estágio com o braço atual.
            peer_config["seed"] = config["seed"]
            peer_config["stage"] = config["stage"]
            peer_config["augmentation_randomness"] = config["augmentation_randomness"]
            validate_fine_tuning_pair(config, peer_config) if config[
                "experiment_id"
            ] == "DR-IF" else validate_fine_tuning_pair(peer_config, config)
        else:
            from tcc_segmentation.boundary_loss_ablation import validate_boundary_ablation_config

            validate_boundary_ablation_config(config, baseline_config=a1_config)
        transform = make_fine_tuning_transform(config)
        validation_transform = ResizePad(
            int(config["data"]["height"]), int(config["data"]["width"])
        )
    elif is_transfer_arm:
        from tcc_segmentation.transfer_learning import (
            make_transfer_train_transform,
            validate_transfer_config,
            validate_transfer_pair,
            save_normalized_batch_panel,
        )

        a1_config = yaml.safe_load((root / "configs/augmentation_d0_a1.yaml").read_text(encoding="utf-8"))
        audit_config = yaml.safe_load(
            (root / "configs/augmentation_audit.yaml").read_text(encoding="utf-8")
        )
        peer_name = "transfer_learning_imagenet_encoder.yaml" if config["experiment_id"] == "DR-RT" else "transfer_learning_random_encoder.yaml"
        peer_raw = yaml.safe_load((root / "configs" / peer_name).read_text(encoding="utf-8"))
        peer_raw.pop("base_config", None)
        peer_config = merge(a1_config, peer_raw)
        validate_transfer_config(config, a1_config=a1_config, audit_config=audit_config)
        validate_transfer_pair(config, peer_config) if config[
            "experiment_id"
        ] == "DR-RT" else validate_transfer_pair(peer_config, config)
        transform = make_transfer_train_transform(config)
        validation_transform = ResizePad(
            int(config["data"]["height"]), int(config["data"]["width"])
        )
    elif is_p1_a1_arm:
        from tcc_segmentation.combined_preprocessing_augmentation import (  # noqa: PLC0415
            make_p1_a1_train_transform,
            make_p1_validation_transform,
            validate_p1_a1_config,
        )
        from tcc_segmentation.augmentation_experiment import (  # noqa: PLC0415
            save_first_training_batch_panel,
        )

        d0_config = yaml.safe_load((root / "configs/baseline_d0.yaml").read_text(encoding="utf-8"))
        p1_config = yaml.safe_load(
            (root / "configs/preprocessing_dp1_shades_of_gray.yaml").read_text(encoding="utf-8")
        )
        a1_config = yaml.safe_load((root / "configs/augmentation_d0_a1.yaml").read_text(encoding="utf-8"))
        audit_config = yaml.safe_load(
            (root / "configs/augmentation_audit.yaml").read_text(encoding="utf-8")
        )
        validate_p1_a1_config(
            config,
            d0_config=d0_config,
            p1_config=p1_config,
            a1_config=a1_config,
            audit_config=audit_config,
        )
        transform = make_p1_a1_train_transform(config)
        validation_transform = make_p1_validation_transform(config)
    elif is_augmentation_arm:
        from tcc_segmentation.augmentation_experiment import (  # noqa: PLC0415
            make_augmentation_pair_transform,
            save_first_training_batch_panel,
            validate_augmentation_training_config,
        )

        d0_config = yaml.safe_load((root / "configs/baseline_d0.yaml").read_text(encoding="utf-8"))
        audit_config = yaml.safe_load(
            (root / "configs/augmentation_audit.yaml").read_text(encoding="utf-8")
        )
        validate_augmentation_training_config(
            config, d0_config=d0_config, audit_config=audit_config
        )
        transform = make_augmentation_pair_transform(config)
        validation_transform = ResizePad(
            height=int(config["data"]["height"]), width=int(config["data"]["width"])
        )
    else:
        transform = ResizePad(
            height=int(config["data"]["height"]), width=int(config["data"]["width"])
        )
        validation_transform = ResizePad(
            height=int(config["data"]["height"]), width=int(config["data"]["width"])
        )
    train_dataset = DevelopmentDataset(
        root / config["data"]["train_manifest"],
        data_root,
        expected_split="train",
        transform=transform,
        mask_threshold=int(config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"] if normalization else None,
        normalization_std=normalization["std"] if normalization else None,
    )
    validation_dataset = DevelopmentDataset(
        root / config["data"]["validation_manifest"],
        data_root,
        expected_split="validation",
        transform=validation_transform,
        mask_threshold=int(config["data"]["mask_threshold"]),
        normalization_mean=normalization["mean"] if normalization else None,
        normalization_std=normalization["std"] if normalization else None,
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
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "experiment_id": str(config.get("experiment_id", "D0")),
        "augmentation": augmentation_name,
        "color_preprocessing": str(config["data"].get("color_preprocessing", "none")),
        "normalization": normalization,
        "model_architecture": config["model"]["architecture"],
        "encoder_weights": config["model"].get("encoder_weights"),
        "fine_tuning_strategy": config.get("fine_tuning", {}).get("strategy"),
        "protected_partitions_accessed": False,
    }
    write_json(output_dir / "environment_report.json", environment)

    if is_augmentation_arm and not resume:
        transform.set_epoch(0)
        (
            save_normalized_batch_panel
            if (is_transfer_arm or is_fine_tuning_arm)
            else save_first_training_batch_panel
        )(
            train_dataset,
            output_dir / "first_training_batch.png",
            samples=int(config["training"]["batch_size"]),
        )

    training_function = train_fine_tuning if is_fine_tuning_arm else train_d0
    training_summary = training_function(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        config=config,
        output_dir=output_dir,
        run_id=run_id,
        device=device,
        manifest_hashes=manifest_hashes,
        resume=resume,
    )

    if is_augmentation_arm:
        history_rows = list(csv.DictReader((output_dir / "history.csv").open(encoding="utf-8")))
        safety = {
            "rejection_rule": config["augmentation_safety"]["rejection_rule"],
            "retention_review_threshold": config["augmentation_safety"][
                "retention_review_threshold"
            ],
            "retention_threshold_filters_training": False,
            "epochs_recorded": len(history_rows),
            "samples_seen": sum(
                int(row.get("augmentation_samples_seen", 0)) for row in history_rows
            ),
            "empty_mask_rejections": sum(
                int(row.get("augmentation_empty_mask_rejections", 0)) for row in history_rows
            ),
            "resample_attempts": sum(
                int(row.get("augmentation_resample_attempts", 0)) for row in history_rows
            ),
            "identity_fallbacks": sum(
                int(row.get("augmentation_identity_fallbacks", 0)) for row in history_rows
            ),
            "protected_partitions_accessed": False,
        }
        write_json(output_dir / "augmentation_safety.json", safety)

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
        "scientific_role": str(config.get("scientific_role", "central_baseline_development")),
        "experiment_id": str(config.get("experiment_id", "D0")),
        "augmentation": augmentation_name,
        "color_preprocessing": str(config["data"].get("color_preprocessing", "none")),
        "normalization": normalization,
        "model_architecture": config["model"]["architecture"],
        "encoder_weights": config["model"].get("encoder_weights"),
        "fine_tuning_strategy": config.get("fine_tuning", {}).get("strategy"),
        "run_id": run_id,
        "seed": int(config["seed"]),
        "model_parameters": count_parameters(model),
        "training": training_summary,
        "selection": selection,
        **({"augmentation_safety": safety} if is_augmentation_arm else {}),
        "artifacts": {
            "training_curves": str(output_dir / "training_curves.png"),
            "threshold_curves": str(output_dir / "threshold_curves.png"),
            "fixed_validation_panel": str(panel_path),
            **(
                {"first_training_batch": str(output_dir / "first_training_batch.png")}
                if is_augmentation_arm
                else {}
            ),
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
