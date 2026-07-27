"""Offline goalkeeper-glove crop classification.

The classifier expects small BGR ``uint8`` hand crops, converts them to RGB,
and returns probabilities for ``bare_hand`` and ``goalkeeper_glove``.  Model
construction always uses ``weights=None``: inference never downloads
torchvision weights and only uses the explicitly supplied local checkpoint.

Checkpoint format (schema version 1)::

    {
        "schema": "ai_referee.glove_classifier",
        "schema_version": 1,
        "architecture": "mobilenet_v3_small",
        "class_names": ["bare_hand", "goalkeeper_glove"],
        "input_size": [128, 128],
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "state_dict": model.state_dict(),
    }

Only tensor/primitive checkpoints are supported so they can be loaded with
``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small

CHECKPOINT_SCHEMA = "ai_referee.glove_classifier"
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE = "mobilenet_v3_small"
CLASS_NAMES = ("bare_hand", "goalkeeper_glove")
DEFAULT_INPUT_SIZE = (128, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_glove_model() -> nn.Module:
    """Build the binary MobileNetV3-Small without downloading any weights."""

    model = mobilenet_v3_small(weights=None)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise RuntimeError("Unexpected torchvision MobileNetV3-Small classifier layout")
    model.classifier[-1] = nn.Linear(final_layer.in_features, len(CLASS_NAMES))
    return model


def _validated_input_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Glove checkpoint input_size must be [height, width]")
    height, width = value
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height < 16
        or width < 16
    ):
        raise ValueError(
            "Glove checkpoint input_size values must be integers of at least 16"
        )
    return height, width


def _validated_normalization(value: Any, name: str, *, positive: bool) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Glove checkpoint {name} must contain three numbers")
    normalized: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"Glove checkpoint {name} must contain three numbers")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0):
            requirement = "positive finite" if positive else "finite"
            raise ValueError(
                f"Glove checkpoint {name} values must be {requirement} numbers"
            )
        normalized.append(number)
    return tuple(normalized)


def create_glove_checkpoint(
    model: nn.Module,
    *,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN,
    normalization_std: tuple[float, float, float] = IMAGENET_STD,
) -> dict[str, Any]:
    """Create a safe, versioned checkpoint payload from a trained model."""

    validated_size = _validated_input_size(input_size)
    validated_mean = _validated_normalization(
        normalization_mean, "normalization_mean", positive=False
    )
    validated_std = _validated_normalization(
        normalization_std, "normalization_std", positive=True
    )
    state_dict = model.state_dict()
    if not state_dict or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Glove model state_dict must contain named tensors")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "class_names": list(CLASS_NAMES),
        "input_size": list(validated_size),
        "normalization_mean": list(validated_mean),
        "normalization_std": list(validated_std),
        "state_dict": {
            key: value.detach().cpu() for key, value in state_dict.items()
        },
    }


def save_glove_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN,
    normalization_std: tuple[float, float, float] = IMAGENET_STD,
) -> Path:
    """Save a trained model in the supported inference checkpoint format."""

    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        create_glove_checkpoint(
            model,
            input_size=input_size,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
        ),
        checkpoint_path,
    )
    return checkpoint_path


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Glove classifier checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Could not safely load glove classifier checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Glove classifier checkpoint must contain a dictionary")
    return payload


def _validate_checkpoint(
    payload: Mapping[str, Any],
) -> tuple[
    Mapping[str, torch.Tensor],
    tuple[int, int],
    tuple[float, ...],
    tuple[float, ...],
]:
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported glove checkpoint schema: {payload.get('schema')!r}"
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported glove checkpoint schema_version: "
            f"{payload.get('schema_version')!r}; expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError(
            f"Unsupported glove checkpoint architecture: {payload.get('architecture')!r}"
        )
    class_names = payload.get("class_names")
    if not isinstance(class_names, (list, tuple)) or tuple(class_names) != CLASS_NAMES:
        raise ValueError(
            "Glove checkpoint class_names must be "
            f"{list(CLASS_NAMES)!r} in that order"
        )
    input_size = _validated_input_size(payload.get("input_size"))
    normalization_mean = _validated_normalization(
        payload.get("normalization_mean"),
        "normalization_mean",
        positive=False,
    )
    normalization_std = _validated_normalization(
        payload.get("normalization_std"),
        "normalization_std",
        positive=True,
    )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("Glove checkpoint state_dict must be a non-empty dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Glove checkpoint state_dict must contain named tensors")
    return state_dict, input_size, normalization_mean, normalization_std


def preprocess_bgr_crops(
    crops: Sequence[np.ndarray],
    input_size: tuple[int, int],
    normalization_mean: tuple[float, ...] = IMAGENET_MEAN,
    normalization_std: tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """Validate and preprocess BGR ``uint8`` crops into an NCHW RGB tensor."""

    if isinstance(crops, np.ndarray) or not isinstance(crops, Sequence):
        raise ValueError(
            "crops must be a sequence of BGR images; wrap one image in a list"
        )
    validated_size = _validated_input_size(input_size)
    mean_values = _validated_normalization(
        normalization_mean, "normalization_mean", positive=False
    )
    std_values = _validated_normalization(
        normalization_std, "normalization_std", positive=True
    )
    tensors: list[torch.Tensor] = []
    for index, crop in enumerate(crops):
        if not isinstance(crop, np.ndarray):
            raise ValueError(f"crops[{index}] must be a NumPy array")
        if crop.dtype != np.uint8:
            raise ValueError(f"crops[{index}] must have dtype uint8")
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"crops[{index}] must have shape [height, width, 3]")
        if crop.shape[0] < 1 or crop.shape[1] < 1:
            raise ValueError(f"crops[{index}] must not be empty")
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=validated_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        tensors.append(tensor)

    if not tensors:
        return torch.empty((0, 3, *validated_size), dtype=torch.float32)
    batch = torch.stack(tensors)
    mean = torch.tensor(mean_values, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_values, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


class GloveClassifier:
    """CPU-friendly local MobileNetV3-Small glove classifier."""

    class_names = CLASS_NAMES
    glove_class_index = CLASS_NAMES.index("goalkeeper_glove")

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        torch_threads: int | None = None,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Glove classifier batch_size must be a positive integer")
        if torch_threads is not None and (
            isinstance(torch_threads, bool)
            or not isinstance(torch_threads, int)
            or torch_threads < 1
        ):
            raise ValueError("torch_threads must be a positive integer or None")
        try:
            resolved_device = torch.device(device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"Invalid glove classifier device {device!r}") from exc
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {device}")
        if torch_threads is not None:
            torch.set_num_threads(torch_threads)

        checkpoint_path = Path(checkpoint).expanduser()
        payload = _load_checkpoint(checkpoint_path)
        state_dict, input_size, mean, std = _validate_checkpoint(payload)
        model = build_glove_model()
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Glove checkpoint state_dict is incompatible with "
                f"{ARCHITECTURE}: {exc}"
            ) from exc
        model.eval()
        self.checkpoint = checkpoint_path
        self.device = resolved_device
        self.batch_size = batch_size
        self.input_size = input_size
        self.normalization_mean = mean
        self.normalization_std = std
        self.model = model.to(resolved_device)

    def predict_proba(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return ``[bare_hand, goalkeeper_glove]`` probabilities per crop."""

        inputs = preprocess_bgr_crops(
            crops,
            self.input_size,
            self.normalization_mean,
            self.normalization_std,
        )
        if inputs.shape[0] == 0:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)

        probabilities: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, inputs.shape[0], self.batch_size):
                batch = inputs[start : start + self.batch_size].to(self.device)
                logits = self.model(batch)
                if logits.ndim != 2 or logits.shape != (batch.shape[0], len(CLASS_NAMES)):
                    raise ValueError(
                        "Glove classifier produced an invalid output shape: "
                        f"{tuple(logits.shape)}"
                    )
                batch_probabilities = torch.softmax(logits, dim=1)
                if not torch.isfinite(batch_probabilities).all():
                    raise ValueError("Glove classifier produced non-finite probabilities")
                probabilities.append(batch_probabilities.cpu())
        return torch.cat(probabilities).numpy().astype(np.float32, copy=False)

    def predict_glove_probability(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return the goalkeeper-glove probability for each BGR crop."""

        return self.predict_proba(crops)[:, self.glove_class_index]
