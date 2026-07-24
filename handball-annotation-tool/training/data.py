from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FEATURE_NAMES, feature_path


@dataclass(frozen=True)
class FeatureView:
    example_id: str
    view_id: str
    label: int
    domain: str
    fold: int
    path: Path
    features: np.ndarray


def load_views(manifest_path: Path, features_dir: Path, require_all: bool = True) -> list[FeatureView]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    views: list[FeatureView] = []
    missing: list[Path] = []
    for _, row in manifest.iterrows():
        path = feature_path(features_dir, row)
        if not path.is_file():
            missing.append(path)
            continue
        loaded = np.load(path)
        features = loaded["features"].astype(np.float32)
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Unexpected feature shape in {path}: {features.shape}")
        views.append(FeatureView(
            str(row["example_id"]), str(row["view_id"]), int(row["label"]),
            str(row["domain"]), int(row["fold"]), path, features,
        ))
    if missing and require_all:
        raise FileNotFoundError(
            f"{len(missing)} feature files are missing. Run training.features first. "
            f"First missing file: {missing[0]}"
        )
    return views


def aggregate_sequence(features: np.ndarray) -> np.ndarray:
    """Convert a temporal feature sequence into an interpretable baseline vector."""
    return np.concatenate([
        features.mean(axis=0),
        features.std(axis=0),
        features.min(axis=0),
        features.max(axis=0),
        features[-1] - features[0],
    ]).astype(np.float32)


class RandomViewDataset(Dataset):
    """One independent incident per item, with a random camera view in training."""

    def __init__(
        self,
        views: list[FeatureView],
        mean: np.ndarray,
        std: np.ndarray,
        random_view: bool,
    ):
        self.by_example: dict[str, list[FeatureView]] = {}
        for view in views:
            self.by_example.setdefault(view.example_id, []).append(view)
        self.example_ids = sorted(self.by_example)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)
        self.random_view = random_view
        self.labels = [self.by_example[example_id][0].label for example_id in self.example_ids]

    def __len__(self) -> int:
        return len(self.example_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        example_id = self.example_ids[index]
        candidates = self.by_example[example_id]
        view = random.choice(candidates) if self.random_view else candidates[0]
        normalized = (view.features - self.mean) / self.std
        return torch.from_numpy(normalized), torch.tensor(view.label, dtype=torch.float32), example_id


def normalization(views: list[FeatureView]) -> tuple[np.ndarray, np.ndarray]:
    if not views:
        raise ValueError("Cannot calculate normalization from an empty training set")
    matrix = np.concatenate([view.features for view in views], axis=0)
    return matrix.mean(axis=0), matrix.std(axis=0)


def feature_metadata(path: Path) -> dict[str, object]:
    loaded = np.load(path)
    return json.loads(str(loaded["metadata"]))

