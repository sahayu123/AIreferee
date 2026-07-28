"""Supervised full-player goalkeeper image classification.

Inference is deliberately offline: a versioned local checkpoint is required
and model construction never downloads weights.  Training may initialize the
same architecture with torchvision ImageNet weights.
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
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

CHECKPOINT_SCHEMA = "ai_referee.goalkeeper_classifier"
CHECKPOINT_SCHEMA_VERSION = 1
ARCHITECTURE = "mobilenet_v3_small"
CLASS_NAMES = ("not_goalkeeper", "goalkeeper")
DEFAULT_INPUT_SIZE = (224, 224)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_goalkeeper_model(*, pretrained: bool = False) -> nn.Module:
    """Build the supported two-class MobileNetV3-Small."""

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    final_layer = model.classifier[-1]
    if not isinstance(final_layer, nn.Linear):
        raise RuntimeError(
            "Unexpected torchvision MobileNetV3-Small classifier layout"
        )
    model.classifier[-1] = nn.Linear(
        final_layer.in_features, len(CLASS_NAMES)
    )
    return model


def _validate_input_size(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Goalkeeper checkpoint input_size must be [height, width]")
    height, width = value
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or height < 32
        or width < 32
    ):
        raise ValueError(
            "Goalkeeper checkpoint input dimensions must be integers >= 32"
        )
    return height, width


def _validate_triplet(
    value: Any, name: str, *, positive: bool
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Goalkeeper checkpoint {name} must contain 3 numbers")
    numbers = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"Goalkeeper checkpoint {name} must be finite")
    if positive and not all(item > 0 for item in numbers):
        raise ValueError(f"Goalkeeper checkpoint {name} must be positive")
    return numbers  # type: ignore[return-value]


def create_checkpoint(
    model: nn.Module,
    *,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN,
    normalization_std: tuple[float, float, float] = IMAGENET_STD,
    decision_thresholds: tuple[float, float] = (0.25, 0.75),
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    low, high = (float(item) for item in decision_thresholds)
    if not 0 <= low < high <= 1:
        raise ValueError("Decision thresholds must satisfy 0 <= low < high <= 1")
    state_dict = model.state_dict()
    if not state_dict:
        raise ValueError("Goalkeeper model state_dict cannot be empty")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "class_names": list(CLASS_NAMES),
        "input_size": list(_validate_input_size(input_size)),
        "normalization_mean": list(
            _validate_triplet(
                normalization_mean, "normalization_mean", positive=False
            )
        ),
        "normalization_std": list(
            _validate_triplet(
                normalization_std, "normalization_std", positive=True
            )
        ),
        "decision_thresholds": [low, high],
        "training_metadata": dict(training_metadata or {}),
        "state_dict": {
            key: value.detach().cpu() for key, value in state_dict.items()
        },
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    **kwargs: Any,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".temporary")
    torch.save(create_checkpoint(model, **kwargs), temporary)
    temporary.replace(destination)
    return destination


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Goalkeeper classifier checkpoint not found: {path}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"Could not safely load goalkeeper checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Goalkeeper checkpoint must contain a dictionary")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Unsupported goalkeeper checkpoint schema: {payload.get('schema')!r}"
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported goalkeeper checkpoint schema version")
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported goalkeeper checkpoint architecture")
    if tuple(payload.get("class_names", ())) != CLASS_NAMES:
        raise ValueError(
            f"Goalkeeper checkpoint classes must be {list(CLASS_NAMES)!r}"
        )
    return payload


def preprocess_bgr_crops(
    crops: Sequence[np.ndarray],
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    if isinstance(crops, np.ndarray) or not isinstance(crops, Sequence):
        raise ValueError("crops must be a sequence of BGR uint8 images")
    height, width = _validate_input_size(input_size)
    mean_values = _validate_triplet(mean, "normalization_mean", positive=False)
    std_values = _validate_triplet(std, "normalization_std", positive=True)
    tensors: list[torch.Tensor] = []
    for index, crop in enumerate(crops):
        if (
            not isinstance(crop, np.ndarray)
            or crop.dtype != np.uint8
            or crop.ndim != 3
            or crop.shape[2] != 3
            or min(crop.shape[:2]) < 1
        ):
            raise ValueError(
                f"crops[{index}] must be a non-empty BGR uint8 HxWx3 image"
            )
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255)
        tensor = F.interpolate(
            tensor[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0]
        tensors.append(tensor)
    if not tensors:
        return torch.empty((0, 3, height, width), dtype=torch.float32)
    batch = torch.stack(tensors)
    mean_tensor = torch.tensor(mean_values).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std_values).view(1, 3, 1, 1)
    return (batch - mean_tensor) / std_tensor


def aggregate_track_probabilities(
    probabilities: Sequence[float],
    *,
    minimum_crops: int = 3,
    not_goalkeeper_threshold: float = 0.25,
    goalkeeper_threshold: float = 0.75,
    minimum_agreement: float = 0.60,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Track probabilities must be a finite 1-D sequence")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Track probabilities must be between 0 and 1")
    if minimum_crops < 1:
        raise ValueError("minimum_crops must be positive")
    if not 0 <= not_goalkeeper_threshold < goalkeeper_threshold <= 1:
        raise ValueError("Invalid goalkeeper decision thresholds")
    if not 0.5 <= minimum_agreement <= 1:
        raise ValueError("minimum_agreement must be between 0.5 and 1")
    if len(values) < minimum_crops:
        return {
            "status": "unknown",
            "is_goalkeeper": None,
            "probability": float(np.median(values)) if len(values) else None,
            "valid_crops": int(len(values)),
            "agreement": None,
            "reason": "insufficient_valid_player_crops",
        }
    median = float(np.median(values))
    goalkeeper_agreement = float(np.mean(values >= goalkeeper_threshold))
    field_agreement = float(np.mean(values <= not_goalkeeper_threshold))
    if (
        median >= goalkeeper_threshold
        and goalkeeper_agreement >= minimum_agreement
    ):
        status, is_goalkeeper, agreement, reason = (
            "goalkeeper",
            True,
            goalkeeper_agreement,
            "median_and_frame_votes_support_goalkeeper",
        )
    elif (
        median <= not_goalkeeper_threshold
        and field_agreement >= minimum_agreement
    ):
        status, is_goalkeeper, agreement, reason = (
            "not_goalkeeper",
            False,
            field_agreement,
            "median_and_frame_votes_support_field_player",
        )
    else:
        status, is_goalkeeper = "unknown", None
        agreement = max(goalkeeper_agreement, field_agreement)
        reason = "track_probabilities_are_ambiguous"
    return {
        "status": status,
        "is_goalkeeper": is_goalkeeper,
        "probability": median,
        "valid_crops": int(len(values)),
        "agreement": agreement,
        "reason": reason,
    }


class GoalkeeperClassifier:
    goalkeeper_class_index = CLASS_NAMES.index("goalkeeper")

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device requested but unavailable: {device}")
        payload = _load_checkpoint(Path(checkpoint).expanduser())
        self.input_size = _validate_input_size(payload.get("input_size"))
        self.mean = _validate_triplet(
            payload.get("normalization_mean"),
            "normalization_mean",
            positive=False,
        )
        self.std = _validate_triplet(
            payload.get("normalization_std"),
            "normalization_std",
            positive=True,
        )
        thresholds = payload.get("decision_thresholds")
        if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 2:
            raise ValueError("Goalkeeper checkpoint thresholds are invalid")
        self.not_goalkeeper_threshold = float(thresholds[0])
        self.goalkeeper_threshold = float(thresholds[1])
        model = build_goalkeeper_model(pretrained=False)
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
        except (KeyError, RuntimeError) as exc:
            raise ValueError(
                f"Incompatible goalkeeper checkpoint state_dict: {exc}"
            ) from exc
        self.model = model.eval().to(resolved_device)
        self.device = resolved_device
        self.batch_size = batch_size

    def predict_proba(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        inputs = preprocess_bgr_crops(
            crops, self.input_size, self.mean, self.std
        )
        if len(inputs) == 0:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(inputs), self.batch_size):
                logits = self.model(
                    inputs[start : start + self.batch_size].to(self.device)
                )
                if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
                    raise RuntimeError("Goalkeeper model returned invalid logits")
                outputs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(outputs).numpy().astype(np.float32, copy=False)

    def predict_goalkeeper_probability(
        self, crops: Sequence[np.ndarray]
    ) -> np.ndarray:
        return self.predict_proba(crops)[:, self.goalkeeper_class_index]
