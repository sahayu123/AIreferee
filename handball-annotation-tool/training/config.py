from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class FeatureConfig:
    detector: Path
    mediapipe_model: Path
    tracker: str
    device: str
    confidence: float
    ball_confidence: float
    pose_confidence: float
    pose_presence: float
    pose_input_size: int
    crop_margin: float
    max_nearby_players: int
    sequence_length: int
    manifest: Path
    features_dir: Path
    overlays_dir: Path
    overlay_examples_per_domain: int


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    features_dir: Path
    checkpoints_dir: Path
    reports_dir: Path
    logs_dir: Path
    device: str
    seed: int
    folds: int
    fold: int
    epochs: int
    batch_size: int
    hidden_size: int
    layers: int
    dropout: float
    learning_rate: float
    weight_decay: float
    patience: int


def _read(path: str | Path) -> dict[str, Any]:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid configuration in {config_path}: expected a mapping")
    return raw


def load_feature_config(path: str | Path) -> FeatureConfig:
    raw = _read(path)
    try:
        model, extraction, paths = raw["models"], raw["extraction"], raw["paths"]
        return FeatureConfig(
            detector=project_path(model["detector"]),
            mediapipe_model=project_path(model["mediapipe_pose"]),
            tracker=str(model.get("tracker", "bytetrack.yaml")),
            device=str(model.get("device", "auto")),
            confidence=float(extraction.get("person_confidence", 0.25)),
            ball_confidence=float(extraction.get("ball_confidence", 0.10)),
            pose_confidence=float(extraction.get("pose_detection_confidence", 0.35)),
            pose_presence=float(extraction.get("pose_presence_confidence", 0.35)),
            pose_input_size=int(extraction.get("pose_input_size", 512)),
            crop_margin=float(extraction.get("crop_margin", 0.25)),
            max_nearby_players=int(extraction.get("max_nearby_players", 3)),
            sequence_length=int(extraction.get("sequence_length", 12)),
            manifest=project_path(paths["manifest"]),
            features_dir=project_path(paths["features"]),
            overlays_dir=project_path(paths["overlays"]),
            overlay_examples_per_domain=int(extraction.get("overlay_examples_per_domain", 5)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid feature configuration: {exc}") from exc


def load_train_config(path: str | Path) -> TrainConfig:
    raw = _read(path)
    try:
        training, paths = raw["training"], raw["paths"]
        return TrainConfig(
            manifest=project_path(paths["manifest"]),
            features_dir=project_path(paths["features"]),
            checkpoints_dir=project_path(paths["checkpoints"]),
            reports_dir=project_path(paths["reports"]),
            logs_dir=project_path(paths["logs"]),
            device=str(training.get("device", "auto")),
            seed=int(training.get("seed", 42)),
            folds=int(training.get("folds", 5)),
            fold=int(training.get("fold", 0)),
            epochs=int(training.get("epochs", 60)),
            batch_size=int(training.get("batch_size", 16)),
            hidden_size=int(training.get("hidden_size", 64)),
            layers=int(training.get("layers", 2)),
            dropout=float(training.get("dropout", 0.25)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            patience=int(training.get("patience", 10)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid training configuration: {exc}") from exc
