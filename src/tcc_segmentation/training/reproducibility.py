"""Controle explicito das fontes usuais de aleatoriedade."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    if seed < 0:
        raise ValueError("seed deve ser nao negativa")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic_algorithms)
    torch.use_deterministic_algorithms(bool(deterministic_algorithms), warn_only=True)
