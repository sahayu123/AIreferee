from __future__ import annotations

import random

import numpy as np
import torch


def get_device(requested: str = "auto") -> str:
    requested = requested.lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return "cuda"
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Apple MPS was requested but is unavailable.")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

