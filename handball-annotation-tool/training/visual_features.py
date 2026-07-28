"""Extract frozen visual embeddings aligned with the temporal handball features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)
import yaml

from handball_annotator.runtime import get_device

from .config import project_path
from .data import feature_metadata
from .features import FEATURE_NAMES, feature_path
from .logging_utils import configure_logging
from .manifest import sorted_frames

VISUAL_BACKBONE = "mobilenet_v3_small_imagenet_v1"
VISUAL_EMBEDDING_SIZE = 576


@dataclass(frozen=True)
class VisualFeatureConfig:
    manifest: Path
    base_features_dir: Path
    visual_features_dir: Path
    logs_dir: Path
    device: str
    crop_margin: float
    batch_size: int


def load_visual_feature_config(path: str | Path) -> VisualFeatureConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        extraction = raw["visual_extraction"]
        paths = raw["paths"]
        config = VisualFeatureConfig(
            manifest=project_path(paths["manifest"]),
            base_features_dir=project_path(paths["base_features"]),
            visual_features_dir=project_path(paths["visual_features"]),
            logs_dir=project_path(paths["visual_logs"]),
            device=str(extraction.get("device", "auto")),
            crop_margin=float(extraction.get("crop_margin", 0.35)),
            batch_size=int(extraction.get("batch_size", 32)),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid visual feature configuration: {exc}") from exc
    if config.crop_margin < 0 or config.crop_margin > 2:
        raise ValueError("visual_extraction.crop_margin must be between 0 and 2")
    if config.batch_size < 1:
        raise ValueError("visual_extraction.batch_size must be positive")
    return config


class MobileNetVisualEncoder(nn.Module):
    """ImageNet MobileNetV3-Small without its classification head."""

    def __init__(self, *, pretrained: bool = True):
        super().__init__()
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        model = mobilenet_v3_small(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        input_features = model.classifier[0].in_features
        if input_features != VISUAL_EMBEDDING_SIZE:
            raise RuntimeError(
                "Unexpected MobileNetV3-Small embedding size: "
                f"{input_features}"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.features(images)
        return torch.flatten(self.avgpool(encoded), 1)


def visual_feature_path(root: Path, row: pd.Series) -> Path:
    return feature_path(root, row)


def _normalized_box(
    feature_row: np.ndarray,
    indices: Mapping[str, int],
    *,
    prefix: str,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    if float(feature_row[indices[f"{prefix}_valid"]]) <= 0:
        return None
    center_x = float(feature_row[indices[f"{prefix}_x"]]) * frame_width
    center_y = float(feature_row[indices[f"{prefix}_y"]]) * frame_height
    width = float(feature_row[indices[f"{prefix}_w"]]) * frame_width
    height = float(feature_row[indices[f"{prefix}_h"]]) * frame_height
    if width <= 1 or height <= 1:
        return None
    return (
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def context_crop_box(
    feature_row: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    crop_margin: float,
) -> tuple[int, int, int, int]:
    """Return a player-and-ball context crop, falling back to the full frame."""

    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    player = _normalized_box(
        feature_row,
        indices,
        prefix="player",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    ball = _normalized_box(
        feature_row,
        indices,
        prefix="ball",
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if player is None:
        return (0, 0, frame_width, frame_height)
    boxes = [player]
    if ball is not None:
        boxes.append(ball)
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    left -= width * crop_margin
    right += width * crop_margin
    top -= height * crop_margin
    bottom += height * crop_margin
    # Avoid extremely narrow crops when only a partial player was detected.
    minimum_side = min(frame_width, frame_height) * 0.12
    if right - left < minimum_side:
        center = (left + right) / 2
        left, right = center - minimum_side / 2, center + minimum_side / 2
    if bottom - top < minimum_side:
        center = (top + bottom) / 2
        top, bottom = center - minimum_side / 2, center + minimum_side / 2
    return (
        max(0, int(np.floor(left))),
        max(0, int(np.floor(top))),
        min(frame_width, int(np.ceil(right))),
        min(frame_height, int(np.ceil(bottom))),
    )


def load_context_crops(
    row: pd.Series,
    base_features_dir: Path,
    crop_margin: float,
) -> tuple[list[Image.Image], list[int], Path]:
    base_path = feature_path(base_features_dir, row)
    if not base_path.is_file():
        raise FileNotFoundError(f"Base feature artifact not found: {base_path}")
    with np.load(base_path, allow_pickle=False) as loaded:
        features = loaded["features"].astype(np.float32)
    metadata = feature_metadata(base_path)
    names = [str(name) for name in metadata.get("feature_names", [])]
    selected = [
        int(index)
        for index in metadata.get("selected_frame_indices", [])
    ]
    if names != FEATURE_NAMES:
        raise ValueError(f"Base feature schema mismatch in {base_path}")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected base feature shape in {base_path}: {features.shape}"
        )
    if len(selected) != len(features):
        raise ValueError(
            f"Base feature/index mismatch in {base_path}: "
            f"{len(features)} != {len(selected)}"
        )
    frames_dir = project_path(str(row["frames_dir"]))
    frame_paths = sorted_frames(frames_dir)
    crops: list[Image.Image] = []
    for feature_row, frame_index in zip(features, selected):
        if not 0 <= frame_index < len(frame_paths):
            raise ValueError(
                f"Selected frame {frame_index} is unavailable in {frames_dir}"
            )
        with Image.open(frame_paths[frame_index]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        box = context_crop_box(
            feature_row,
            frame_width=image.width,
            frame_height=image.height,
            crop_margin=crop_margin,
        )
        crops.append(image.crop(box))
    return crops, selected, base_path


def encode_crops(
    crops: Sequence[Image.Image],
    model: nn.Module,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if not crops:
        raise ValueError("Cannot encode an empty crop sequence")
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    tensors = [transform(crop) for crop in crops]
    embeddings: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size]).to(device)
            values = model(batch).detach().cpu().numpy().astype(np.float32)
            embeddings.append(values)
    matrix = np.concatenate(embeddings, axis=0)
    if matrix.shape != (len(crops), VISUAL_EMBEDDING_SIZE):
        raise RuntimeError(f"Unexpected visual embedding shape: {matrix.shape}")
    return matrix


def save_visual_artifact(
    destination: Path,
    embeddings: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    if (
        embeddings.ndim != 2
        or embeddings.shape[1] != VISUAL_EMBEDDING_SIZE
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(
            "Visual embeddings must be a finite "
            f"[time, {VISUAL_EMBEDDING_SIZE}] matrix"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.npz")
    np.savez_compressed(
        temporary,
        embeddings=embeddings.astype(np.float32),
        metadata=json.dumps(dict(metadata), sort_keys=True),
    )
    temporary.replace(destination)


def extract_visual_manifest(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
    device_override: str | None = None,
) -> None:
    config = load_visual_feature_config(config_path)
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    manifest = pd.read_csv(config.manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        manifest = manifest.head(limit)
    device = get_device(device_override or config.device)
    model = MobileNetVisualEncoder(pretrained=True).to(device)
    logger = configure_logging(config.logs_dir / "visual_features.log")
    logger.info(
        "backbone=%s device=%s examples=%d",
        VISUAL_BACKBONE,
        device,
        len(manifest),
    )
    for number, (_, row) in enumerate(manifest.iterrows(), start=1):
        destination = visual_feature_path(config.visual_features_dir, row)
        if destination.is_file() and not overwrite:
            logger.info(
                "[%d/%d] cached %s %s",
                number,
                len(manifest),
                row["example_id"],
                row["view_id"],
            )
            continue
        crops, selected, base_path = load_context_crops(
            row,
            config.base_features_dir,
            config.crop_margin,
        )
        embeddings = encode_crops(
            crops,
            model,
            device=device,
            batch_size=config.batch_size,
        )
        metadata = {
            "schema": "ai_referee.visual_features",
            "schema_version": 1,
            "backbone": VISUAL_BACKBONE,
            "embedding_size": VISUAL_EMBEDDING_SIZE,
            "example_id": str(row["example_id"]),
            "view_id": str(row["view_id"]),
            "label": int(row["label"]),
            "domain": str(row["domain"]),
            "source_group": str(row["source_group"]),
            "selected_frame_indices": selected,
            "base_feature_path": str(base_path),
            "crop_margin": config.crop_margin,
        }
        save_visual_artifact(destination, embeddings, metadata)
        logger.info(
            "[%d/%d] extracted %s %s shape=%s",
            number,
            len(manifest),
            row["example_id"],
            row["view_id"],
            tuple(embeddings.shape),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen MobileNetV3 player-and-ball visual embeddings."
        )
    )
    parser.add_argument("--config", default="configs/visual_fusion.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    extract_visual_manifest(
        args.config,
        overwrite=args.overwrite,
        limit=args.limit,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()
