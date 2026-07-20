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
class AppConfig:
    detector: str
    pose_model: str
    tracker: str
    device: str
    confidence: float
    person_class_id: int
    ball_class_id: int
    arm_distance_threshold: float
    pose_keypoint_confidence: float
    nearby_player_margin: float
    frames_before: int
    frames_after: int
    candidate_cooldown_frames: int
    max_nearby_players: int
    uploads_dir: Path
    candidates_dir: Path
    dataset_dir: Path
    state_dir: Path


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid configuration in {config_path}: expected a mapping")
    try:
        models, mining, paths = raw["models"], raw["mining"], raw["paths"]
        return AppConfig(
            detector=str(models["detector"]), pose_model=str(models["pose"]),
            tracker=str(models["tracker"]), device=str(models.get("device", "auto")),
            confidence=float(mining["confidence"]),
            person_class_id=int(mining["person_class_id"]), ball_class_id=int(mining["ball_class_id"]),
            arm_distance_threshold=float(mining["arm_distance_threshold"]),
            pose_keypoint_confidence=float(mining["pose_keypoint_confidence"]),
            nearby_player_margin=float(mining["nearby_player_margin"]),
            frames_before=int(mining["frames_before"]), frames_after=int(mining["frames_after"]),
            candidate_cooldown_frames=int(mining["candidate_cooldown_frames"]),
            max_nearby_players=int(mining["max_nearby_players"]),
            uploads_dir=project_path(paths["uploads"]), candidates_dir=project_path(paths["candidates"]),
            dataset_dir=project_path(paths["dataset"]), state_dir=project_path(paths["state"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration in {config_path}: {exc}") from exc
