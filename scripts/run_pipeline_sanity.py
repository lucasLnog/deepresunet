#!/usr/bin/env python3
"""Executa o gate técnico aprovado da verificação do pipeline sem acessar teste ou FUSeg."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline_sanity.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
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
    run_id = datetime.now(timezone.utc).strftime("day4_%Y%m%dT%H%M%SZ")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else root / config["runtime"]["output_root"] / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config_used.yaml")
    sys.path.insert(0, str(root / "src"))

    from tcc_segmentation.data import (  # noqa: PLC0415
        DevelopmentDataset,
        ResizePad,
        create_development_dataloader,
    )
    from tcc_segmentation.pipeline_sanity import (  # noqa: PLC0415
        batch_diagnostics,
        collect_predictions,
        deterministic_prefix_indices,
        indices_for_canonical_ids,
        prediction_max_abs_difference,
        run_overfit_sanity,
        run_smoke_sanity,
        save_before_after_panel,
        save_history_csv,
        write_json,
    )
    from tcc_segmentation.models import (  # noqa: PLC0415
        build_segmentation_model,
        count_parameters,
    )
    from tcc_segmentation.training import (  # noqa: PLC0415
        BCEDiceLoss,
        load_training_checkpoint,
        save_training_checkpoint,
        seed_everything,
    )

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = ResizePad(height=int(config["data"]["height"]), width=int(config["data"]["width"]))
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

    overfit_cfg = config["overfit"]
    overfit_indices = indices_for_canonical_ids(train_dataset, overfit_cfg["canonical_ids"])
    overfit_dataset = torch.utils.data.Subset(train_dataset, overfit_indices)
    overfit_loader = create_development_dataloader(
        overfit_dataset,
        batch_size=int(overfit_cfg["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(config["runtime"]["num_workers"]),
    )
    diagnostics = batch_diagnostics(next(iter(overfit_loader)))
    write_json(diagnostics, output_dir / "batch_diagnostics.json")

    criterion = BCEDiceLoss(
        bce_weight=float(config["loss"]["bce_weight"]),
        dice_weight=float(config["loss"]["dice_weight"]),
        smooth=float(config["loss"]["smooth"]),
    )
    model = build_segmentation_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(overfit_cfg["learning_rate"]))
    before = collect_predictions(model, overfit_loader, device)
    history, overfit_accepted = run_overfit_sanity(
        model=model,
        loader=overfit_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=int(overfit_cfg["epochs"]),
        threshold=float(overfit_cfg["threshold"]),
        min_loss_reduction_fraction=float(
            overfit_cfg["early_acceptance"]["min_loss_reduction_fraction"]
        ),
        min_macro_dice=float(overfit_cfg["early_acceptance"]["min_macro_dice"]),
    )
    after = collect_predictions(model, overfit_loader, device)
    save_history_csv(history, output_dir / "overfit_history.csv")
    save_before_after_panel(
        before,
        after,
        output_dir / "overfit_before_after.png",
        threshold=float(overfit_cfg["threshold"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot([row["epoch"] for row in history], [row["loss"] for row in history])
    axes[0].set(title="Loss — overfit pequeno", xlabel="Época", ylabel="Loss")
    axes[1].plot(
        [row["epoch"] for row in history],
        [row["macro_dice_at_0_5"] for row in history],
    )
    axes[1].set(title="Dice técnico em threshold 0,5", xlabel="Época", ylabel="Dice")
    figure.tight_layout()
    figure.savefig(output_dir / "overfit_curves.png", dpi=160)
    plt.close(figure)

    smoke_cfg = config["smoke"]
    smoke_train_dataset = torch.utils.data.Subset(
        train_dataset,
        deterministic_prefix_indices(len(train_dataset), int(smoke_cfg["train_samples"])),
    )
    smoke_validation_dataset = torch.utils.data.Subset(
        validation_dataset,
        deterministic_prefix_indices(len(validation_dataset), int(smoke_cfg["validation_samples"])),
    )
    train_loader = create_development_dataloader(
        smoke_train_dataset,
        batch_size=int(smoke_cfg["batch_size"]),
        shuffle=True,
        seed=seed,
        num_workers=int(config["runtime"]["num_workers"]),
    )
    validation_loader = create_development_dataloader(
        smoke_validation_dataset,
        batch_size=int(smoke_cfg["batch_size"]),
        shuffle=False,
        seed=seed,
        num_workers=int(config["runtime"]["num_workers"]),
    )
    seed_everything(seed)
    smoke_model = build_segmentation_model(config["model"]).to(device)
    smoke_optimizer = torch.optim.Adam(
        smoke_model.parameters(), lr=float(smoke_cfg["learning_rate"])
    )
    smoke_result = run_smoke_sanity(
        model=smoke_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=criterion,
        optimizer=smoke_optimizer,
        device=device,
        threshold=float(smoke_cfg["threshold"]),
    )
    predictions_before_reload = collect_predictions(smoke_model, validation_loader, device)
    summary = json.loads((root / "manifests/splits/split_summary.json").read_text(encoding="utf-8"))
    checkpoint_path = output_dir / "smoke_checkpoint.pt"
    save_training_checkpoint(
        checkpoint_path,
        kind="last",
        epoch=1,
        best_score=float(smoke_result["validation"]["metrics"]["dice"]["mean"]),
        model=smoke_model,
        optimizer=smoke_optimizer,
        config=config,
        run_id=run_id,
        manifest_hashes={
            "train": summary["manifest_hashes"]["train"],
            "validation": summary["manifest_hashes"]["validation"],
        },
        extra={"scientific_result": False, "protected_partitions_accessed": False},
    )
    reloaded_model = build_segmentation_model(config["model"]).to(device)
    reloaded_optimizer = torch.optim.Adam(
        reloaded_model.parameters(), lr=float(smoke_cfg["learning_rate"])
    )
    load_training_checkpoint(
        checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        map_location=device,
        restore_rng=False,
    )
    predictions_after_reload = collect_predictions(reloaded_model, validation_loader, device)
    roundtrip_error = prediction_max_abs_difference(
        predictions_before_reload, predictions_after_reload
    )
    write_json(smoke_result, output_dir / "smoke_summary.json")

    report = {
        "status": "passed" if overfit_accepted and roundtrip_error <= 1e-6 else "review",
        "scientific_result": False,
        "run_id": run_id,
        "device": str(device),
        "model_parameters": count_parameters(smoke_model),
        "overfit": {
            "canonical_ids": list(overfit_cfg["canonical_ids"]),
            "epochs_executed": len(history),
            "accepted": overfit_accepted,
            "first": history[0],
            "last": history[-1],
        },
        "smoke": {
            "train_samples": len(smoke_train_dataset),
            "validation_samples": len(smoke_validation_dataset),
            "checkpoint": str(checkpoint_path),
            "checkpoint_roundtrip_max_abs_logit_error": roundtrip_error,
        },
        "partitions_accessed": ["train", "validation"],
        "protected_partitions_accessed": False,
    }
    write_json(report, output_dir / "day4_sanity_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaídas: {output_dir}")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
