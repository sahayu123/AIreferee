"""Apply a trained player-crop classifier to the associated handball actor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np
import yaml

from .config import project_path
from .goalkeeper_classifier import (
    GoalkeeperClassifier,
    aggregate_track_probabilities,
)
from .goalkeeper_dataset import expanded_crop
from .prtreid_role import (
    PersonTracker,
    PRTReIDConfig,
    TrackObservation,
    associate_handball_actor,
    load_prtreid_config,
    track_all_people,
)

SCHEMA_VERSION = 1


class PlayerCropClassifier(Protocol):
    not_goalkeeper_threshold: float
    goalkeeper_threshold: float

    def predict_goalkeeper_probability(
        self, crops: Sequence[np.ndarray]
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SupervisedGoalkeeperConfig:
    base_config_path: Path
    base_config: PRTReIDConfig
    checkpoint: Path
    device: str
    batch_size: int
    crop_margin: float
    minimum_player_height: int
    minimum_blur_variance: float
    maximum_samples: int
    minimum_samples: int
    not_goalkeeper_threshold: float | None
    goalkeeper_threshold: float | None
    minimum_frame_agreement: float
    roles_dir: Path
    reports_dir: Path


def load_supervised_goalkeeper_config(
    path: str | Path,
) -> SupervisedGoalkeeperConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Supervised goalkeeper config not found: {config_path}"
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        base, classifier, crops, decision, paths = (
            raw["base"],
            raw["classifier"],
            raw["crops"],
            raw["decision"],
            raw["paths"],
        )
        base_path = project_path(base["prtreid_config"])

        def optional_float(value: Any) -> float | None:
            return None if value is None else float(value)

        config = SupervisedGoalkeeperConfig(
            base_config_path=base_path,
            base_config=load_prtreid_config(base_path),
            checkpoint=project_path(classifier["checkpoint"]),
            device=str(classifier.get("device", "cpu")),
            batch_size=int(classifier.get("batch_size", 16)),
            crop_margin=float(crops.get("margin", 0.15)),
            minimum_player_height=int(
                crops.get("minimum_player_height", 100)
            ),
            minimum_blur_variance=float(
                crops.get("minimum_blur_variance", 15.0)
            ),
            maximum_samples=int(crops.get("maximum_samples", 8)),
            minimum_samples=int(crops.get("minimum_samples", 3)),
            not_goalkeeper_threshold=optional_float(
                decision.get("not_goalkeeper_threshold")
            ),
            goalkeeper_threshold=optional_float(
                decision.get("goalkeeper_threshold")
            ),
            minimum_frame_agreement=float(
                decision.get("minimum_frame_agreement", 0.60)
            ),
            roles_dir=project_path(paths["roles"]),
            reports_dir=project_path(paths["reports"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid supervised goalkeeper config: {exc}"
        ) from exc
    if not 0 <= config.crop_margin <= 1:
        raise ValueError("crop margin must be between 0 and 1")
    if config.minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if config.maximum_samples < config.minimum_samples:
        raise ValueError("maximum_samples must be >= minimum_samples")
    return config


def config_fingerprint(config: SupervisedGoalkeeperConfig) -> str:
    checkpoint = (
        {
            "path": str(config.checkpoint),
            "size": config.checkpoint.stat().st_size,
            "mtime_ns": config.checkpoint.stat().st_mtime_ns,
        }
        if config.checkpoint.is_file()
        else {"path": str(config.checkpoint), "size": None, "mtime_ns": None}
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": checkpoint,
        "base_config": str(config.base_config_path),
        "crop": {
            "margin": config.crop_margin,
            "minimum_player_height": config.minimum_player_height,
            "minimum_blur_variance": config.minimum_blur_variance,
            "maximum_samples": config.maximum_samples,
            "minimum_samples": config.minimum_samples,
        },
        "decision": {
            "not_goalkeeper_threshold": config.not_goalkeeper_threshold,
            "goalkeeper_threshold": config.goalkeeper_threshold,
            "minimum_frame_agreement": config.minimum_frame_agreement,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _frame_shapes(frame_paths: Sequence[Path]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {path}")
        shapes.append(frame.shape[:2])
    return shapes


def extract_track_crops(
    frame_paths: Sequence[Path],
    observations: Sequence[TrackObservation],
    config: SupervisedGoalkeeperConfig,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for observation in observations:
        if not 0 <= observation.frame_index < len(frame_paths):
            continue
        x1, y1, x2, y2 = observation.box
        native_height = float(y2 - y1)
        if native_height < config.minimum_player_height:
            continue
        frame = cv2.imread(str(frame_paths[observation.frame_index]))
        if frame is None:
            continue
        try:
            crop, expanded_box = expanded_crop(
                frame, observation.box, margin=config.crop_margin
            )
        except ValueError:
            continue
        blur = float(
            cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()
        )
        if blur < config.minimum_blur_variance:
            continue
        quality = (
            max(0.0, float(observation.detection_confidence))
            * np.sqrt(max(native_height, 1.0))
            * np.log1p(blur)
        )
        candidates.append(
            (
                float(quality),
                crop,
                {
                    "frame_index": int(observation.frame_index),
                    "bbox": list(expanded_box),
                    "native_player_height": native_height,
                    "blur_variance": blur,
                    "detection_confidence": float(
                        observation.detection_confidence
                    ),
                    "quality": float(quality),
                },
            )
        )
    selected = sorted(
        sorted(candidates, key=lambda item: item[0], reverse=True)[
            : config.maximum_samples
        ],
        key=lambda item: item[2]["frame_index"],
    )
    return (
        [item[1] for item in selected],
        [item[2] for item in selected],
    )


def classify_supervised_goalkeeper(
    frame_paths: Sequence[Path],
    selected_features: np.ndarray,
    selected_indices: Sequence[int],
    metadata: dict[str, Any],
    config: SupervisedGoalkeeperConfig,
    *,
    tracker: PersonTracker | None = None,
    classifier: PlayerCropClassifier | None = None,
) -> dict[str, Any]:
    if not frame_paths:
        raise ValueError("Cannot classify goalkeeper in an empty clip")
    tracks = track_all_people(frame_paths, config.base_config, tracker)
    association = associate_handball_actor(
        tracks,
        _frame_shapes(frame_paths),
        selected_features,
        selected_indices,
        metadata,
        config.base_config,
    )
    actor_observations = (
        tracks.get(association.track_id, [])
        if association.track_id is not None
        else []
    )
    base_result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint(config),
        "evaluated": True,
        "actor_track_id": association.track_id,
        "association": association.as_dict(),
        "actor_observations": [
            item.as_dict() for item in actor_observations
        ],
        "tracked_people": len(tracks),
    }
    if not association.confident or not actor_observations:
        return {
            **base_result,
            "status": "unknown",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": None,
            "reason": "actor_association_uncertain",
            "valid_crops": 0,
            "frame_probabilities": [],
            "crop_metadata": [],
        }
    crops, crop_metadata = extract_track_crops(
        frame_paths, actor_observations, config
    )
    active_classifier = classifier or GoalkeeperClassifier(
        config.checkpoint,
        device=config.device,
        batch_size=config.batch_size,
    )
    probabilities = np.asarray(
        active_classifier.predict_goalkeeper_probability(crops), dtype=float
    )
    if probabilities.shape != (len(crops),):
        raise RuntimeError(
            "Goalkeeper probability count does not match player crops"
        )
    low = (
        config.not_goalkeeper_threshold
        if config.not_goalkeeper_threshold is not None
        else active_classifier.not_goalkeeper_threshold
    )
    high = (
        config.goalkeeper_threshold
        if config.goalkeeper_threshold is not None
        else active_classifier.goalkeeper_threshold
    )
    aggregate = aggregate_track_probabilities(
        probabilities,
        minimum_crops=config.minimum_samples,
        not_goalkeeper_threshold=low,
        goalkeeper_threshold=high,
        minimum_agreement=config.minimum_frame_agreement,
    )
    for crop_item, probability in zip(crop_metadata, probabilities):
        crop_item["goalkeeper_probability"] = float(probability)
    return {
        **base_result,
        "status": aggregate["status"],
        "is_goalkeeper": aggregate["is_goalkeeper"],
        "goalkeeper_evidence_score": aggregate["probability"],
        "reason": aggregate["reason"],
        "valid_crops": aggregate["valid_crops"],
        "frame_agreement": aggregate["agreement"],
        "decision_thresholds": {
            "not_goalkeeper": low,
            "goalkeeper": high,
        },
        "frame_probabilities": probabilities.tolist(),
        "crop_metadata": crop_metadata,
    }
