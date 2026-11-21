"""Loops de uma epoca com logits, AMP e metricas por imagem."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from .metrics import (
    aggregate_metric_rows,
    binary_metrics_from_logits,
    metrics_to_rows,
)


def _autocast_context(device: torch.device, enabled: bool):
    use_amp = bool(enabled and device.type == "cuda")
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _move_batch(batch: dict[str, Any], device: torch.device):
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def _gradient_diagnostics(model) -> dict[str, Any]:
    trainable = 0
    with_gradient = 0
    nonfinite_tensors = 0
    squared_norm = 0.0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        trainable += 1
        if parameter.grad is None:
            continue
        with_gradient += 1
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            nonfinite_tensors += 1
        squared_norm += float(torch.sum(gradient.float() ** 2).item())
    return {
        "trainable_parameter_tensors": trainable,
        "parameter_tensors_with_gradient": with_gradient,
        "parameter_tensors_without_gradient": trainable - with_gradient,
        "nonfinite_gradient_tensors": nonfinite_tensors,
        "gradient_l2_norm": squared_norm**0.5,
    }


def train_epoch(
    *,
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler=None,
    amp: bool = True,
    threshold: float = 0.5,
    gradient_clip_norm: float | None = None,
) -> dict[str, Any]:
    model.train()
    component_sums: dict[str, float] = {}
    samples = 0
    rows: list[dict[str, Any]] = []
    last_gradient_diagnostics: dict[str, Any] | None = None
    optimizer_steps = 0
    amp_skipped_optimizer_steps = 0

    for batch in loader:
        images, masks = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp):
            logits = model(images)
            components = criterion.components(logits, masks)
            loss = components["total"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss nao finita: {loss.detach().cpu().item()}")

        scaler_enabled = scaler is not None and scaler.is_enabled()
        if scaler_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        last_gradient_diagnostics = _gradient_diagnostics(model)
        if last_gradient_diagnostics["parameter_tensors_with_gradient"] == 0:
            raise RuntimeError("Nenhum parametro recebeu gradiente")
        gradients_nonfinite = bool(last_gradient_diagnostics["nonfinite_gradient_tensors"])
        if gradients_nonfinite and not scaler_enabled:
            raise FloatingPointError("Gradiente nao finito detectado sem AMP")
        if gradients_nonfinite:
            # GradScaler.step inspeciona os gradientes, pula optimizer.step quando há
            # inf/NaN e update reduz dinamicamente a escala para o próximo batch.
            scaler.step(optimizer)
            scaler.update()
            amp_skipped_optimizer_steps += 1
        else:
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_steps += 1

        batch_size = int(images.shape[0])
        samples += batch_size
        if not component_sums:
            component_sums = {name: 0.0 for name in components}
        if set(components) != set(component_sums):
            raise RuntimeError("Componentes da loss mudaram dentro da época")
        for name in component_sums:
            component_sums[name] += float(components[name].detach().cpu().item()) * batch_size
        metrics = binary_metrics_from_logits(logits.detach(), masks, threshold=threshold)
        rows.extend(metrics_to_rows(metrics, list(batch["canonical_id"])))

    if samples == 0:
        raise RuntimeError("DataLoader de treino vazio")
    if optimizer_steps == 0:
        raise FloatingPointError(
            "Todas as atualizacoes do optimizer foram rejeitadas por gradientes nao finitos"
        )
    last_gradient_diagnostics = dict(last_gradient_diagnostics or {})
    last_gradient_diagnostics.update(
        {
            "optimizer_steps": optimizer_steps,
            "amp_skipped_optimizer_steps": amp_skipped_optimizer_steps,
        }
    )
    return {
        "loss": {name: value / samples for name, value in component_sums.items()},
        "metrics": aggregate_metric_rows(rows),
        "per_image": rows,
        "gradient_diagnostics": last_gradient_diagnostics,
    }


@torch.inference_mode()
def evaluate_epoch(
    *,
    model,
    loader,
    criterion,
    device: torch.device,
    amp: bool = True,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    component_sums: dict[str, float] = {}
    samples = 0
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images, masks = _move_batch(batch, device)
        with _autocast_context(device, amp):
            logits = model(images)
            components = criterion.components(logits, masks)
        batch_size = int(images.shape[0])
        samples += batch_size
        if not component_sums:
            component_sums = {name: 0.0 for name in components}
        if set(components) != set(component_sums):
            raise RuntimeError("Componentes da loss mudaram dentro da época")
        for name in component_sums:
            component_sums[name] += float(components[name].detach().cpu().item()) * batch_size
        metrics = binary_metrics_from_logits(logits, masks, threshold=threshold)
        rows.extend(metrics_to_rows(metrics, list(batch["canonical_id"])))
    if samples == 0:
        raise RuntimeError("DataLoader de avaliacao vazio")
    return {
        "loss": {name: value / samples for name, value in component_sums.items()},
        "metrics": aggregate_metric_rows(rows),
        "per_image": rows,
    }
