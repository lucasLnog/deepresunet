#!/usr/bin/env python3
"""Registra o ambiente real da execucao local ou Colab em JSON."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


PACKAGES = (
    "torch",
    "torchvision",
    "albumentations",
    "opencv-python-headless",
    "segmentation-models-pytorch",
    "numpy",
    "Pillow",
    "PyYAML",
)

IMPORT_PROBES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "albumentations": "albumentations",
    "opencv": "cv2",
    "segmentation_models_pytorch": "segmentation_models_pytorch",
    "yaml": "yaml",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_info() -> dict[str, object]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    cuda_available = torch.cuda.is_available()
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }


def import_probes() -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for label, module_name in IMPORT_PROBES.items():
        try:
            module = importlib.import_module(module_name)
            results[label] = {
                "ok": True,
                "module": module_name,
                "version": getattr(module, "__version__", None),
            }
        except Exception as error:  # diagnostico deve registrar, nao ocultar a falha
            results[label] = {
                "ok": False,
                "module": module_name,
                "error": f"{type(error).__name__}: {error}",
            }
    return results


def build_report() -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "is_google_colab": "COLAB_RELEASE_TAG" in os.environ,
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "packages": {name: package_version(name) for name in PACKAGES},
        "import_probes": import_probes(),
        "torch": torch_info(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/environment/environment_report.json"),
    )
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
