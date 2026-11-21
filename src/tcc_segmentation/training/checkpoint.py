"""Checkpoints atomicos para selecao e retomada de treinamento."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    kind: str,
    epoch: int,
    best_score: float,
    model,
    optimizer,
    scaler=None,
    scheduler=None,
    config: Mapping[str, Any],
    run_id: str,
    manifest_hashes: Mapping[str, str],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    if kind not in {"last", "best"}:
        raise ValueError("kind deve ser 'last' ou 'best'")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "kind": kind,
        "epoch": int(epoch),
        "best_score": float(best_score),
        "run_id": str(run_id),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": dict(config),
        "manifest_hashes": dict(manifest_hashes),
        "rng_state": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    # O arquivo e produzido pelo proprio projeto. weights_only=False e necessario
    # porque estados de Python/NumPy tambem fazem parte da retomada reproduzivel.
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Versao de checkpoint nao suportada")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
