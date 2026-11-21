"""Validação do protocolo e treinamento com encoder fixo ou progressivo."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

from .augmentation_experiment import EpochSeededSafeAugmentation

ARM_STRATEGIES = {
    "DR-IF": "fixed",
    "DR-IFT": "progressive",
    "DR-IFT-B": "progressive",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def validate_fine_tuning_config(
    config: Mapping[str, Any], *, a1_config=None, audit_config=None
) -> None:
    """Confere o braço e as condições que devem ser iguais às de A1."""
    arm = str(config.get("experiment_id"))
    strategy = str(config.get("fine_tuning", {}).get("strategy"))
    if arm not in ARM_STRATEGIES or strategy != ARM_STRATEGIES.get(arm):
        raise ValueError("braço/estratégia de fine-tuning inválido")

    model = config["model"]
    data = config["data"]
    fine_tuning = config["fine_tuning"]
    if (
        model.get("architecture") != "deep_res_unet_resnet34"
        or model.get("encoder_weights") != "imagenet"
    ):
        raise ValueError("fine-tuning exige DeepResUNet/ResNet-34 ImageNet")
    if not bool(model.get("freeze_encoder")):
        raise ValueError("o estado inicial deve ter encoder congelado")
    if data.get("augmentation") != "A1" or data.get("color_preprocessing") != "none":
        raise ValueError("A comparação de ajuste fino exige P0+A1")

    normalization = data.get("normalization", {})
    if (
        tuple(normalization.get("mean", ())) != IMAGENET_MEAN
        or tuple(normalization.get("std", ())) != IMAGENET_STD
    ):
        raise ValueError("normalização ImageNet inválida")
    if int(fine_tuning.get("minimum_epochs_before_stopping", -1)) != 31:
        raise ValueError("early stopping só pode atuar da época 31 em diante")
    if (
        float(fine_tuning.get("decoder_learning_rate", -1)) != 1e-3
        or float(fine_tuning.get("encoder_learning_rate", -1)) != 1e-4
    ):
        raise ValueError("learning rates diferem do protocolo")

    expected_phases = []
    if arm != "DR-IF":
        expected_phases = [
            {"start_epoch": 1, "trainable_encoder_stages": []},
            {"start_epoch": 16, "trainable_encoder_stages": ["layer4"]},
            {"start_epoch": 31, "trainable_encoder_stages": ["layer3", "layer4"]},
        ]
    if fine_tuning.get("phases") != expected_phases:
        raise ValueError("fases diferem do protocolo congelado")

    if a1_config is not None:
        comparison_keys = ("threshold_selection", "confirmation")
        # A ablação DR-IFT-B altera somente a função de perda neste contraste.
        if arm != "DR-IFT-B":
            comparison_keys = ("loss", *comparison_keys)
        for key in comparison_keys:
            if config[key] != a1_config[key]:
                raise ValueError(f"{key} difere do A1")
        for key in (
            "train_manifest",
            "validation_manifest",
            "height",
            "width",
            "mask_threshold",
            "color_preprocessing",
            "augmentation",
            "protected_partitions",
        ):
            if data[key] != a1_config["data"][key]:
                raise ValueError(f"data.{key} difere do A1")
    if audit_config is not None and config["augmentation_pipelines"] != audit_config["pipelines"]:
        raise ValueError("A1 difere da auditoria")


def validate_fine_tuning_pair(fixed, progressive) -> None:
    """Isola a estratégia do encoder na comparação entre DR-IF e DR-IFT."""
    validate_fine_tuning_config(fixed)
    validate_fine_tuning_config(progressive)
    allowed_differences = {"experiment_id", "scientific_role", "runtime", "fine_tuning"}
    for key in set(fixed) - allowed_differences:
        if fixed[key] != progressive[key]:
            raise ValueError(f"braços diferem indevidamente em {key}")


def make_fine_tuning_transform(config):
    """Monta a transformação A1 com a política de sementes do experimento."""
    validate_fine_tuning_config(config)
    return EpochSeededSafeAugmentation(
        name="A1",
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        pipelines=config["augmentation_pipelines"],
        augmentation_seed_base=int(config["augmentation_randomness"]["seed_base"]),
        max_resamples=int(config["augmentation_safety"]["max_resamples"]),
    )


def stage_for_epoch(config, epoch: int) -> tuple[str, tuple[str, ...]]:
    """Retorna a fase e os blocos treináveis do encoder naquela época."""
    if config["fine_tuning"]["strategy"] == "fixed":
        return "encoder_fixed", ()
    if epoch < 16:
        return "decoder_warmup", ()
    if epoch < 31:
        return "layer4_released", ("layer4",)
    return "layer3_layer4_released", ("layer3", "layer4")


def _optimizer(model, config):
    import torch

    fine_tuning = config["fine_tuning"]
    decoder_lr = float(fine_tuning["decoder_learning_rate"])
    encoder_lr = float(fine_tuning["encoder_learning_rate"])
    return torch.optim.Adam(
        [
            {"params": list(model.decoder_parameters()), "lr": decoder_lr, "name": "decoder"},
            {"params": list(model.layer4.parameters()), "lr": encoder_lr, "name": "layer4"},
            {"params": list(model.layer3.parameters()), "lr": encoder_lr, "name": "layer3"},
        ]
    )


def _scheduler(optimizer, config):
    import torch

    settings = config["training"]["scheduler"]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=settings["mode"],
        factor=float(settings["factor"]),
        patience=int(settings["patience"]),
        min_lr=float(settings["min_learning_rate"]),
    )


def train_fine_tuning(
    *,
    train_dataset,
    validation_dataset,
    config,
    output_dir,
    run_id,
    device,
    manifest_hashes,
    resume=False,
    preflight_result=None,
):
    """Treina ou retoma um braço e seleciona o checkpoint pelo Dice de validação."""
    from .baseline_training import (
        _history_row,
        make_criterion,
        make_model,
        preflight_batch_size,
        read_csv,
        write_csv,
        write_json,
    )
    from .data import create_development_dataloader
    from .models import count_parameters
    from .training import (
        evaluate_epoch,
        load_training_checkpoint,
        save_training_checkpoint,
        seed_everything,
        train_epoch,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "training_summary.json"
    if resume and summary_path.is_file():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    history_path = output / "history.csv"
    last_path = output / "last_checkpoint.pt"
    best_path = output / "best_checkpoint.pt"
    training = config["training"]
    fine_tuning = config["fine_tuning"]
    resuming_checkpoint = resume and last_path.is_file()
    if resuming_checkpoint:
        history = read_csv(history_path)
        batch_size = int(history[0]["batch_size"])
    else:
        if preflight_result is None:
            batch_size, preflight = preflight_batch_size(
                train_dataset=train_dataset, config=config, device=device
            )
        else:
            batch_size, preflight = int(preflight_result[0]), dict(preflight_result[1])
        write_json(output / "preflight.json", preflight)
        history = []

    seed = int(config["seed"])
    seed_everything(seed)
    model = make_model(config).to(device)
    criterion = make_criterion(config)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)

    start_epoch = 1
    best_eligible_dice = -math.inf
    best_display_dice = -math.inf
    material_reference = -math.inf
    epochs_without_improvement = 0
    previous_stage = None
    if resuming_checkpoint:
        checkpoint = load_training_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            restore_rng=True,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_eligible_dice = float(checkpoint["best_score"])
        extra = checkpoint.get("extra", {})
        best_display_dice = float(extra.get("display_best", best_eligible_dice))
        material_reference = float(extra.get("material_reference", best_eligible_dice))
        epochs_without_improvement = int(extra.get("epochs_without_material_improvement", 0))
        previous_stage = extra.get("fine_tuning_stage")

    validation_loader = create_development_dataloader(
        validation_dataset, batch_size=batch_size, shuffle=False, seed=seed, num_workers=0
    )
    first_eligible_epoch = int(fine_tuning["minimum_epochs_before_stopping"])
    stopped_early = False
    for epoch in range(start_epoch, int(training["max_epochs"]) + 1):
        stage, trainable_stages = stage_for_epoch(config, epoch)
        model.set_trainable_encoder_stages(trainable_stages)
        if stage != previous_stage or epoch == first_eligible_epoch:
            # Uma nova fase inicia outra contagem de épocas sem melhora.
            epochs_without_improvement = 0
            material_reference = -math.inf
            for group in optimizer.param_groups:
                if group["name"] in trainable_stages:
                    group["lr"] = float(fine_tuning["encoder_learning_rate"])
            previous_stage = stage

        started = time.perf_counter()
        train_dataset.transform.set_epoch(epoch)
        train_loader = create_development_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed + epoch, num_workers=0
        )
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=None,
            amp=False,
            threshold=0.5,
        )
        validation_metrics = evaluate_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp=False,
            threshold=0.5,
        )
        validation_dice = float(validation_metrics["metrics"]["dice"]["mean"])
        current_lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
        scheduler.step(validation_dice)

        best_display_dice = max(best_display_dice, validation_dice)
        checkpoint_eligible = epoch >= first_eligible_epoch
        if checkpoint_eligible and validation_dice > best_eligible_dice:
            best_eligible_dice = validation_dice
            save_training_checkpoint(
                best_path,
                kind="best",
                epoch=epoch,
                best_score=best_eligible_dice,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                run_id=run_id,
                manifest_hashes=manifest_hashes,
                extra={"selected_batch_size": batch_size, "fine_tuning_stage": stage},
            )

        min_delta = float(training["early_stopping"]["min_delta"])
        if material_reference == -math.inf or validation_dice >= material_reference + min_delta:
            material_reference = validation_dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = _history_row(
            epoch,
            train_metrics,
            validation_metrics,
            current_lrs["decoder"],
            time.perf_counter() - started,
            best_display_dice,
            epochs_without_improvement,
        )
        row.update(
            {
                "batch_size": batch_size,
                "fine_tuning_stage": stage,
                "checkpoint_eligible": checkpoint_eligible,
                "trainable_encoder_stages": "|".join(trainable_stages),
                "decoder_learning_rate": current_lrs["decoder"],
                "layer4_learning_rate": current_lrs["layer4"],
                "layer3_learning_rate": current_lrs["layer3"],
                "trainable_parameters": count_parameters(model)["trainable"],
            }
        )
        for name, value in train_dataset.transform.get_epoch_stats().items():
            row[f"augmentation_{name}"] = value
        history.append(row)
        write_csv(history_path, history)
        save_training_checkpoint(
            last_path,
            kind="last",
            epoch=epoch,
            best_score=best_eligible_dice,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            run_id=run_id,
            manifest_hashes=manifest_hashes,
            extra={
                "selected_batch_size": batch_size,
                "material_reference": material_reference,
                "epochs_without_material_improvement": epochs_without_improvement,
                "fine_tuning_stage": stage,
                "display_best": best_display_dice,
            },
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if checkpoint_eligible and epochs_without_improvement >= int(
            training["early_stopping"]["patience"]
        ):
            stopped_early = True
            break

    best_checkpoint = load_training_checkpoint(
        best_path, model=model, map_location=device, restore_rng=False
    )
    summary = {
        "run_id": run_id,
        "experiment_id": config["experiment_id"],
        "scientific_role": config["scientific_role"],
        "fine_tuning_strategy": fine_tuning["strategy"],
        "epochs_executed": int(history[-1]["epoch"]),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_dice_at_0_5": float(best_checkpoint["best_score"]),
        "stopped_early": stopped_early,
        "selected_batch_size": batch_size,
        "model_parameters": count_parameters(model)["trainable"],
        "device": str(device),
        "protected_partitions_accessed": False,
    }
    write_json(summary_path, summary)
    return summary
