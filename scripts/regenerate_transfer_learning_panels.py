#!/usr/bin/env python3
"""Regenera o painel de validação da comparação de inicialização do encoder com RGB desnormalizado."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser()
    run_dir = run_dir.resolve() if run_dir.is_absolute() else (root / run_dir).resolve()
    sys.path.insert(0, str(root / "src"))
    import torch, yaml
    from tcc_segmentation.data import DevelopmentDataset, ResizePad
    from tcc_segmentation.baseline_training import (
        fixed_subset,
        make_model,
        save_fixed_validation_panel,
    )
    from tcc_segmentation.training import load_training_checkpoint

    config = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    if config.get("experiment_id") not in {"DR-RT", "DR-IT"}:
        raise ValueError("run não pertence à comparação de inicialização do encoder")
    selection = json.loads((run_dir / "threshold_selection.json").read_text(encoding="utf-8"))
    norm = config["data"]["normalization"]
    dataset = DevelopmentDataset(
        root / config["data"]["validation_manifest"],
        data_root,
        "validation",
        transform=ResizePad(int(config["data"]["height"]), int(config["data"]["width"])),
        mask_threshold=int(config["data"]["mask_threshold"]),
        normalization_mean=norm["mean"],
        normalization_std=norm["std"],
    )
    visual = fixed_subset(dataset, config["visualization"]["fixed_validation_ids"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device)
    load_training_checkpoint(
        run_dir / "best_checkpoint.pt", model=model, map_location=device, restore_rng=False
    )
    destination = save_fixed_validation_panel(
        model=model,
        dataset=visual,
        device=device,
        threshold=float(selection["threshold"]),
        output_path=run_dir / "fixed_validation_panel_corrected.png",
    )
    report = {
        "status": "completed",
        "experiment_id": config["experiment_id"],
        "partition": "validation",
        "checkpoint": str(run_dir / "best_checkpoint.pt"),
        "threshold": float(selection["threshold"]),
        "output": str(destination),
        "original_panel_overwritten": False,
        "protected_partitions_accessed": False,
    }
    (run_dir / "panel_regeneration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
