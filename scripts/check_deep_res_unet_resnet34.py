#!/usr/bin/env python3
"""Valida a DeepResUNet/ResNet-34 com tensores sinteticos no Colab."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def tensor_dict_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/architecture/deep_res_unet_resnet34_gate_report.json"),
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size deve ser positivo")

    project_root = args.project_root.expanduser().resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path
    sys.path.insert(0, str(project_root / "src"))

    import torch

    from tcc_segmentation.models import build_segmentation_model, count_parameters
    from tcc_segmentation.training import BCEDiceLoss

    if not torch.cuda.is_available():
        raise RuntimeError("O gate completo requer GPU CUDA no Google Colab")
    device = torch.device("cuda")
    model_dir = project_root / "configs" / "models"
    random_config = yaml.safe_load(
        (model_dir / "deep_res_unet_resnet34_random_frozen.yaml").read_text(encoding="utf-8")
    )
    imagenet_config = yaml.safe_load(
        (model_dir / "deep_res_unet_resnet34_imagenet_frozen.yaml").read_text(encoding="utf-8")
    )

    torch.manual_seed(42)
    random_model = build_segmentation_model(random_config)
    random_decoder_hash = tensor_dict_sha256(random_model.decoder_state_dict())
    random_encoder_hash = tensor_dict_sha256({"stem_conv.weight": random_model.stem_conv.weight})
    random_parameters = count_parameters(random_model)

    torch.manual_seed(42)
    imagenet_model = build_segmentation_model(imagenet_config)
    imagenet_decoder_hash = tensor_dict_sha256(imagenet_model.decoder_state_dict())
    imagenet_encoder_hash = tensor_dict_sha256(
        {"stem_conv.weight": imagenet_model.stem_conv.weight}
    )
    imagenet_parameters = count_parameters(imagenet_model)

    decoder_hash_equal = random_decoder_hash == imagenet_decoder_hash
    encoder_hash_different = random_encoder_hash != imagenet_encoder_hash
    random_model.eval().to(device)
    small_inputs = torch.rand((1, 3, 64, 80), device=device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=True):
        random_logits = random_model(small_inputs)
    random_shape_valid = tuple(random_logits.shape) == (1, 1, 64, 80)
    del random_model, random_logits, small_inputs
    torch.cuda.empty_cache()

    imagenet_model.to(device)
    imagenet_model.train()
    frozen_encoder = imagenet_model.encoder_is_frozen()
    frozen_batchnorm_eval = all(
        not module.training
        for stage in imagenet_model._encoder_stages().values()
        for root in stage
        for module in root.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )

    torch.cuda.reset_peak_memory_stats(device)
    inputs = torch.rand((args.batch_size, 3, 384, 512), device=device, dtype=torch.float32)
    targets = (torch.rand((args.batch_size, 1, 384, 512), device=device) > 0.97).float()
    criterion = BCEDiceLoss()
    with torch.amp.autocast("cuda", enabled=True):
        logits = imagenet_model(inputs)
        loss = criterion(logits, targets)
    loss.backward()

    encoder_gradients_absent = all(
        parameter.grad is None for parameter in imagenet_model.encoder_parameters()
    )
    decoder_gradients = [
        parameter.grad
        for parameter in imagenet_model.decoder_parameters()
        if parameter.requires_grad
    ]
    decoder_gradients_present = bool(decoder_gradients) and all(
        gradient is not None for gradient in decoder_gradients
    )
    decoder_gradients_finite = decoder_gradients_present and all(
        bool(torch.isfinite(gradient).all()) for gradient in decoder_gradients
    )
    full_shape_valid = tuple(logits.shape) == (
        args.batch_size,
        1,
        384,
        512,
    )
    logits_finite = bool(torch.isfinite(logits).all())
    loss_finite = bool(torch.isfinite(loss))
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    imagenet_model.set_trainable_encoder_stages(("layer4",))
    imagenet_model.train()
    progressive_stage_valid = (
        any(parameter.requires_grad for parameter in imagenet_model.layer4.parameters())
        and not any(parameter.requires_grad for parameter in imagenet_model.layer3.parameters())
        and any(
            module.training
            for module in imagenet_model.layer4.modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        )
        and not imagenet_model.stem_bn.training
    )

    passed = all(
        (
            decoder_hash_equal,
            encoder_hash_different,
            random_shape_valid,
            frozen_encoder,
            frozen_batchnorm_eval,
            encoder_gradients_absent,
            decoder_gradients_present,
            decoder_gradients_finite,
            full_shape_valid,
            logits_finite,
            loss_finite,
            progressive_stage_valid,
        )
    )
    report = {
        "status": "passed" if passed else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_result": False,
        "dataset_accessed": False,
        "protected_partitions_accessed": False,
        "architecture": "deep_res_unet_resnet34",
        "decoder": "DeepResUNet historica preservada",
        "encoder": "ResNet-34 com conv1 stride 1 e max-pooling preservado",
        "random_parameters": random_parameters,
        "imagenet_parameters": imagenet_parameters,
        "random_decoder_sha256": random_decoder_hash,
        "imagenet_decoder_sha256": imagenet_decoder_hash,
        "decoder_initialization_identical": decoder_hash_equal,
        "encoder_initialization_different": encoder_hash_different,
        "random_output_shape_valid": random_shape_valid,
        "full_batch_input_shape": list(inputs.shape),
        "full_batch_output_shape": list(logits.shape),
        "full_batch_shape_valid": full_shape_valid,
        "output_semantics": "logits",
        "logits_finite": logits_finite,
        "loss_finite": loss_finite,
        "encoder_frozen": frozen_encoder,
        "frozen_encoder_batchnorm_eval": frozen_batchnorm_eval,
        "encoder_gradients_absent_when_frozen": encoder_gradients_absent,
        "decoder_gradients_present": decoder_gradients_present,
        "decoder_gradients_finite": decoder_gradients_finite,
        "progressive_layer4_gate_passed": progressive_stage_valid,
        "cuda_peak_memory_mb": peak_memory_mb,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Relatorio: {output_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
