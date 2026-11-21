"""Dataset de desenvolvimento compatível com PyTorch DataLoader."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .manifest import load_development_manifest, resolve_dataset_path
from .masks import load_canonical_mask


PairTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class DevelopmentDataset:
    """Fornece arrays CHW para treino/validacao sem conhecer o teste."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        expected_split: str,
        transform: PairTransform | None = None,
        mask_threshold: int = 128,
        normalization_mean: tuple[float, float, float] | list[float] | None = None,
        normalization_std: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.rows = load_development_manifest(manifest_path, expected_split)
        self.expected_split = expected_split
        self.transform = transform
        self.mask_threshold = mask_threshold
        if (normalization_mean is None) != (normalization_std is None):
            raise ValueError("mean e std devem ser informados juntos")
        self.normalization_mean = None
        self.normalization_std = None
        if normalization_mean is not None:
            if len(normalization_mean) != 3 or len(normalization_std) != 3:
                raise ValueError("normalização RGB exige três médias e três desvios")
            if any(float(value) <= 0 for value in normalization_std):
                raise ValueError("desvios da normalização devem ser positivos")
            self.normalization_mean = np.asarray(normalization_mean, dtype=np.float32)[
                :, None, None
            ]
            self.normalization_std = np.asarray(normalization_std, dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = resolve_dataset_path(self.project_root, row["image_path"])
        mask_paths = [
            resolve_dataset_path(self.project_root, path) for path in row["mask_paths"].split("|")
        ]
        with Image.open(image_path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
        mask = load_canonical_mask(
            mask_paths, policy=row["mask_policy"], threshold=self.mask_threshold
        )
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Dimensoes incompatíveis em {row['canonical_id']}")
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image_chw = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        image_chw /= 255.0
        if self.normalization_mean is not None:
            image_chw = np.ascontiguousarray(
                (image_chw - self.normalization_mean) / self.normalization_std,
                dtype=np.float32,
            )
        mask_chw = np.ascontiguousarray(mask[None, ...], dtype=np.float32)
        return {
            "image": image_chw,
            "mask": mask_chw,
            "canonical_id": row["canonical_id"],
            "split": row["split"],
            "original_size": np.asarray([int(row["height"]), int(row["width"])], dtype=np.int64),
        }


def _seed_worker(worker_id: int) -> None:
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_development_dataloader(
    dataset: DevelopmentDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """Cria DataLoader com gerador e workers semeados de modo reproduzivel."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch nao esta instalado. Execute este carregador no ambiente Colab."
        ) from error
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
