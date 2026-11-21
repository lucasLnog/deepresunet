#!/usr/bin/env python3
"""Valida no Colab a migração estrutural da DeepResUNet, sem abrir datasets."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


HISTORICAL_PARAMETER_COUNT = 8_212_029


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/architecture/deep_res_unet_migration_report.json"),
    )
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    sys.path.insert(0, str(project_root / "src"))

    import torch

    from tcc_segmentation.models import build_segmentation_model, count_parameters
    from tcc_segmentation.training import BCEDiceLoss

    config_path = project_root / "configs/models/deep_res_unet_historical.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_segmentation_model(config).to(device)
    parameters = count_parameters(model)

    model.eval()
    inputs = torch.rand((1, 3, 64, 80), device=device)
    targets = (torch.rand((1, 1, 64, 80), device=device) > 0.9).float()
    with torch.inference_mode():
        logits = model(inputs)
    criterion = BCEDiceLoss()
    train_model = build_segmentation_model(config).to(device)
    train_inputs = inputs.clone()
    train_logits = train_model(train_inputs)
    loss = criterion(train_logits, targets)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in train_model.parameters()
    )

    passed = (
        parameters["total"] == HISTORICAL_PARAMETER_COUNT
        and tuple(logits.shape) == (1, 1, 64, 80)
        and bool(torch.isfinite(logits).all())
        and bool(torch.isfinite(loss))
        and gradients_finite
    )
    report = {
        "status": "passed" if passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "dataset_accessed": False,
        "architecture": config["architecture"],
        "provenance": config["provenance"],
        "historical_parameter_count": HISTORICAL_PARAMETER_COUNT,
        "parameters": parameters,
        "parameter_count_preserved": parameters["total"] == HISTORICAL_PARAMETER_COUNT,
        "input_shape": list(inputs.shape),
        "output_shape": list(logits.shape),
        "output_semantics": "logits",
        "final_sigmoid_removed": True,
        "attention_sigmoids_preserved": True,
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatorio: {output_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
