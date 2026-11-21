from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.tcc_segmentation.models import build_segmentation_model
from src.tcc_segmentation.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.tcc_segmentation.training.engine import evaluate_epoch, train_epoch
from src.tcc_segmentation.training.losses import BCEDiceLoss, SoftDiceLoss
from src.tcc_segmentation.training.metrics import (
    aggregate_metric_rows,
    binary_metrics_from_logits,
    metrics_to_rows,
)


def test_bce_dice_loss_is_finite_and_differentiable() -> None:
    logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, :, 2:6, 2:6] = 1.0
    criterion = BCEDiceLoss(0.5, 0.5, smooth=1.0)
    components = criterion.components(logits, targets)
    assert all(torch.isfinite(value) for value in components.values())
    components["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0


def test_soft_dice_prefers_perfect_logits() -> None:
    targets = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    perfect = torch.where(targets > 0, 12.0, -12.0)
    inverted = -perfect
    criterion = SoftDiceLoss(smooth=1.0)
    assert criterion(perfect, targets) < criterion(inverted, targets)


def test_metrics_are_per_image_and_perfect() -> None:
    targets = torch.tensor(
        [
            [[[0.0, 1.0], [1.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ]
    )
    logits = torch.where(targets > 0, 12.0, -12.0)
    metrics = binary_metrics_from_logits(logits, targets, threshold=0.5)
    assert metrics["dice"].shape == (2,)
    assert torch.allclose(metrics["dice"], torch.ones(2))
    assert torch.allclose(metrics["iou"], torch.ones(2))
    rows = metrics_to_rows(metrics, ["a", "b"])
    summary = aggregate_metric_rows(rows)
    assert summary["images"] == 2
    assert summary["dice"]["mean"] == pytest.approx(1.0)


def test_both_empty_prediction_and_reference_are_explicitly_perfect() -> None:
    logits = torch.full((1, 1, 4, 4), -12.0)
    targets = torch.zeros_like(logits)
    metrics = binary_metrics_from_logits(logits, targets)
    for name in ("dice", "iou", "precision", "recall", "specificity", "accuracy"):
        assert metrics[name].item() == pytest.approx(1.0)


def test_empty_prediction_against_positive_reference_has_zero_precision() -> None:
    logits = torch.full((1, 1, 4, 4), -12.0)
    targets = torch.zeros_like(logits)
    targets[:, :, 1:3, 1:3] = 1.0
    metrics = binary_metrics_from_logits(logits, targets)
    assert metrics["precision"].item() == pytest.approx(0.0)
    assert metrics["recall"].item() == pytest.approx(0.0)
    assert metrics["dice"].item() == pytest.approx(0.0)


class _TinyDataset:
    def __init__(self) -> None:
        self.images = torch.rand((2, 3, 8, 8))
        self.masks = (self.images[:, :1] > 0.5).float()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {
            "image": self.images[index],
            "mask": self.masks[index],
            "canonical_id": f"tiny_{index}",
        }


def test_train_and_evaluate_epoch_produce_per_image_rows() -> None:
    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = BCEDiceLoss()
    train_result = train_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        scaler=None,
        amp=False,
    )
    eval_result = evaluate_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
        amp=False,
    )
    assert len(train_result["per_image"]) == 2
    assert len(eval_result["per_image"]) == 2
    assert train_result["gradient_diagnostics"]["nonfinite_gradient_tensors"] == 0
    assert train_result["gradient_diagnostics"]["optimizer_steps"] == 1
    assert train_result["gradient_diagnostics"]["amp_skipped_optimizer_steps"] == 0


class _FakeGradScaler:
    def is_enabled(self) -> bool:
        return True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        gradients = [
            parameter.grad
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if all(torch.isfinite(gradient).all() for gradient in gradients):
            optimizer.step()

    def update(self) -> None:
        return None


def test_amp_transient_nonfinite_gradient_skips_only_affected_step() -> None:
    loader = torch.utils.data.DataLoader(_TinyDataset(), batch_size=1, shuffle=False)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    calls = {"count": 0}

    def inject_once(gradient):
        calls["count"] += 1
        return torch.full_like(gradient, float("inf")) if calls["count"] == 1 else gradient

    model.weight.register_hook(inject_once)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    result = train_epoch(
        model=model,
        loader=loader,
        criterion=BCEDiceLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        scaler=_FakeGradScaler(),
        amp=False,
    )
    diagnostics = result["gradient_diagnostics"]
    assert diagnostics["amp_skipped_optimizer_steps"] == 1
    assert diagnostics["optimizer_steps"] == 1


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "last.pt"
    save_training_checkpoint(
        path,
        kind="last",
        epoch=3,
        best_score=0.75,
        model=model,
        optimizer=optimizer,
        config={"experiment_type": "test"},
        run_id="run_test",
        manifest_hashes={"train": "abc"},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    payload = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        restore_rng=False,
    )
    assert payload["epoch"] == 3
    assert payload["kind"] == "last"
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])


def test_unet_returns_logits_with_expected_shape() -> None:
    pytest.importorskip("segmentation_models_pytorch")
    model = build_segmentation_model(
        {
            "architecture": "unet",
            "encoder": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 1,
            "activation": None,
        }
    )
    model.eval()
    final_convolution = model.segmentation_head[0]
    with torch.no_grad():
        final_convolution.weight.zero_()
        final_convolution.bias.fill_(2.0)
    with torch.inference_mode():
        output = model(torch.rand((1, 3, 64, 64)))
    assert output.shape == (1, 1, 64, 64)
    # Se houvesse sigmoid na cabeca, o valor seria sigmoid(2) ~= 0,88.
    assert torch.allclose(output, torch.full_like(output, 2.0))
