from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import select
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np
import yaml

from handball_annotator.runtime import get_device

from .config import PROJECT_ROOT, project_path
from .features import FEATURE_NAMES
from .role_detector import box_actor_coverage, box_iou, selected_player_box

ROLE_NAMES = ("ball", "goalkeeper", "other", "player", "referee")
ACTOR_ROLES = ("goalkeeper", "player", "referee")
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PRTReIDConfig:
    person_detector: Path
    prtreid_checkpoint: Path
    device: str
    tracker: str
    confidence: float
    image_size: int
    minimum_track_frames: int
    worker_command: tuple[str, ...]
    worker_timeout_seconds: float
    worker_batch_size: int
    center_box_margin: float
    minimum_anchor_coverage: float
    minimum_anchor_iou: float
    minimum_anchor_votes: int
    minimum_association_score: float
    minimum_association_margin: float
    minimum_predictions: int
    minimum_role_coverage: float
    role_threshold: float
    goalkeeper_threshold: float
    minimum_role_margin: float
    color_weight: float
    color_activation: float
    jersey_crop_top: float
    jersey_crop_bottom: float
    jersey_crop_left: float
    jersey_crop_right: float
    jersey_minimum_pixels: int
    jersey_minimum_comparison_tracks: int
    jersey_outlier_distance: float
    manifest: Path
    features_dir: Path
    roles_dir: Path
    audits_dir: Path
    report: Path
    logs_dir: Path


@dataclass
class TrackObservation:
    frame_index: int
    frame_path: Path
    box: tuple[float, float, float, float]
    detection_confidence: float
    worker_prediction: dict[str, Any] | None = None
    jersey_lab: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "frame_path": str(self.frame_path),
            "bbox": list(self.box),
            "detection_confidence": self.detection_confidence,
            "worker_prediction": self.worker_prediction,
            "jersey_lab": self.jersey_lab,
        }


@dataclass(frozen=True)
class ActorAssociation:
    track_id: int | None
    method: str
    score: float
    margin: float
    anchor_votes: int
    confident: bool
    conflicting: bool
    evidence: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "method": self.method,
            "score": self.score,
            "margin": self.margin,
            "anchor_votes": self.anchor_votes,
            "confident": self.confident,
            "conflicting": self.conflicting,
            "evidence": list(self.evidence),
        }


class PersonTracker(Protocol):
    def track(
        self, frame_paths: Sequence[Path]
    ) -> dict[int, list[TrackObservation]]:
        ...


class RoleWorker(Protocol):
    def predict(self, crops: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        ...


def load_prtreid_config(path: str | Path) -> PRTReIDConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"PRTReID configuration not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("configuration must be a mapping")
        models = raw["models"]
        tracking = raw.get("tracking", {})
        worker = raw["worker"]
        association = raw.get("association", {})
        aggregation = raw.get("aggregation", {})
        color = raw.get("jersey_color", {})
        paths = raw["paths"]
        command = worker["command"]
        if not isinstance(command, list) or not command:
            raise TypeError("worker.command must be a non-empty list")
        config = PRTReIDConfig(
            person_detector=project_path(models["person_detector"]),
            prtreid_checkpoint=project_path(models["prtreid_checkpoint"]),
            device=str(models.get("device", "auto")),
            tracker=str(tracking.get("tracker", "bytetrack.yaml")),
            confidence=float(tracking.get("confidence", 0.25)),
            image_size=int(tracking.get("image_size", 640)),
            minimum_track_frames=int(tracking.get("minimum_track_frames", 3)),
            worker_command=tuple(str(value) for value in command),
            worker_timeout_seconds=float(worker.get("timeout_seconds", 180)),
            worker_batch_size=int(worker.get("batch_size", 16)),
            center_box_margin=float(association.get("center_box_margin", 0.15)),
            minimum_anchor_coverage=float(
                association.get("minimum_anchor_coverage", 0.50)
            ),
            minimum_anchor_iou=float(
                association.get("minimum_anchor_iou", 0.15)
            ),
            minimum_anchor_votes=int(
                association.get("minimum_anchor_votes", 1)
            ),
            minimum_association_score=float(
                association.get("minimum_score", 0.35)
            ),
            minimum_association_margin=float(
                association.get("minimum_margin", 0.10)
            ),
            minimum_predictions=int(aggregation.get("minimum_predictions", 3)),
            minimum_role_coverage=float(
                aggregation.get("minimum_coverage", 0.25)
            ),
            role_threshold=float(aggregation.get("role_threshold", 0.60)),
            goalkeeper_threshold=float(
                aggregation.get("goalkeeper_threshold", 0.70)
            ),
            minimum_role_margin=float(
                aggregation.get("minimum_role_margin", 0.15)
            ),
            color_weight=float(aggregation.get("color_weight", 0.10)),
            color_activation=float(
                aggregation.get("color_activation", 0.40)
            ),
            jersey_crop_top=float(color.get("crop_top", 0.15)),
            jersey_crop_bottom=float(color.get("crop_bottom", 0.60)),
            jersey_crop_left=float(color.get("crop_left", 0.20)),
            jersey_crop_right=float(color.get("crop_right", 0.80)),
            jersey_minimum_pixels=int(color.get("minimum_pixels", 50)),
            jersey_minimum_comparison_tracks=int(
                color.get("minimum_comparison_tracks", 4)
            ),
            jersey_outlier_distance=float(color.get("outlier_distance", 35.0)),
            manifest=project_path(paths["manifest"]),
            features_dir=project_path(paths["features"]),
            roles_dir=project_path(paths["roles"]),
            audits_dir=project_path(paths["audits"]),
            report=project_path(paths["report"]),
            logs_dir=project_path(paths["logs"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid PRTReID configuration: {exc}") from exc
    _validate_config(config)
    return config


def _validate_config(config: PRTReIDConfig) -> None:
    if config.minimum_track_frames < 1:
        raise ValueError("minimum_track_frames must be at least 1")
    if config.minimum_predictions < 1:
        raise ValueError("minimum_predictions must be at least 1")
    if config.minimum_anchor_votes < 1:
        raise ValueError("minimum_anchor_votes must be at least 1")
    if config.worker_batch_size < 1:
        raise ValueError("worker.batch_size must be at least 1")
    if config.worker_timeout_seconds <= 0:
        raise ValueError("worker.timeout_seconds must be positive")
    if config.image_size < 32:
        raise ValueError("tracking.image_size is unexpectedly small")
    for name, value in (
        ("tracking.confidence", config.confidence),
        ("association.center_box_margin", config.center_box_margin),
        ("association.minimum_anchor_coverage", config.minimum_anchor_coverage),
        ("association.minimum_anchor_iou", config.minimum_anchor_iou),
        ("association.minimum_score", config.minimum_association_score),
        ("association.minimum_margin", config.minimum_association_margin),
        ("aggregation.minimum_coverage", config.minimum_role_coverage),
        ("aggregation.role_threshold", config.role_threshold),
        ("aggregation.goalkeeper_threshold", config.goalkeeper_threshold),
        ("aggregation.minimum_role_margin", config.minimum_role_margin),
        ("aggregation.color_weight", config.color_weight),
        ("aggregation.color_activation", config.color_activation),
        ("jersey_color.crop_top", config.jersey_crop_top),
        ("jersey_color.crop_bottom", config.jersey_crop_bottom),
        ("jersey_color.crop_left", config.jersey_crop_left),
        ("jersey_color.crop_right", config.jersey_crop_right),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.jersey_crop_top >= config.jersey_crop_bottom:
        raise ValueError("jersey crop top must be above bottom")
    if config.jersey_crop_left >= config.jersey_crop_right:
        raise ValueError("jersey crop left must be left of right")
    if config.jersey_minimum_pixels < 1:
        raise ValueError("jersey minimum_pixels must be positive")
    if config.jersey_minimum_comparison_tracks < 1:
        raise ValueError("jersey minimum_comparison_tracks must be positive")
    if config.jersey_outlier_distance <= 0:
        raise ValueError("jersey outlier_distance must be positive")
    forbidden_worker_arguments = {"--checkpoint", "--batch-size"}
    duplicated = forbidden_worker_arguments.intersection(config.worker_command)
    if duplicated:
        raise ValueError(
            "worker.command must not set arguments managed by the application: "
            f"{sorted(duplicated)}"
        )


class YOLOPersonTracker:
    """One reusable YOLO model with ByteTrack state reset for every clip."""

    def __init__(
        self,
        config: PRTReIDConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        if not config.person_detector.is_file():
            raise FileNotFoundError(
                f"Person detector not found: {config.person_detector}. "
                "Run `python -m training.download_models`."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Install Ultralytics in the main environment before tracking people."
            ) from exc
        self.config = config
        self.progress_callback = progress_callback
        self.device = get_device(config.device)
        self.model = YOLO(str(config.person_detector))

    def _reset_for_clip(self) -> None:
        predictor = self.model.predictor
        if predictor is None or not hasattr(predictor, "trackers"):
            return
        for tracker in predictor.trackers:
            tracker.reset()
        predictor.vid_path = [None] * len(predictor.trackers)

    def track(
        self, frame_paths: Sequence[Path]
    ) -> dict[int, list[TrackObservation]]:
        self._reset_for_clip()
        tracks: dict[int, list[TrackObservation]] = {}
        for frame_index, frame_path in enumerate(frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read tracking frame: {frame_path}")
            result = self.model.track(
                source=frame,
                persist=True,
                tracker=self.config.tracker,
                conf=self.config.confidence,
                imgsz=self.config.image_size,
                classes=[0],
                device=self.device,
                verbose=False,
            )[0]
            if self.progress_callback is not None:
                self.progress_callback(frame_index + 1, len(frame_paths))
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            ids = (
                result.boxes.id.detach().cpu().numpy().astype(int)
                if result.boxes.id is not None
                else None
            )
            if ids is None:
                continue
            for box, confidence, track_id in zip(boxes, confidences, ids):
                observation = TrackObservation(
                    frame_index=int(frame_index),
                    frame_path=Path(frame_path).resolve(),
                    box=tuple(float(value) for value in box),
                    detection_confidence=float(confidence),
                )
                existing = tracks.setdefault(int(track_id), [])
                duplicate = next(
                    (
                        item
                        for item in existing
                        if item.frame_index == observation.frame_index
                    ),
                    None,
                )
                if duplicate is None:
                    existing.append(observation)
                elif observation.detection_confidence > duplicate.detection_confidence:
                    existing[existing.index(duplicate)] = observation
        for observations in tracks.values():
            observations.sort(key=lambda item: item.frame_index)
        return tracks


def track_all_people(
    frame_paths: Sequence[Path],
    config: PRTReIDConfig,
    tracker: PersonTracker | None = None,
) -> dict[int, list[TrackObservation]]:
    active_tracker = tracker if tracker is not None else YOLOPersonTracker(config)
    tracks = active_tracker.track(frame_paths)
    normalized: dict[int, list[TrackObservation]] = {}
    for raw_track_id, observations in tracks.items():
        track_id = int(raw_track_id)
        best_by_frame: dict[int, TrackObservation] = {}
        for observation in observations:
            prior = best_by_frame.get(int(observation.frame_index))
            if (
                prior is None
                or observation.detection_confidence > prior.detection_confidence
            ):
                best_by_frame[int(observation.frame_index)] = observation
        normalized[track_id] = [
            best_by_frame[index] for index in sorted(best_by_frame)
        ]
    return normalized


def _frame_shapes(frame_paths: Sequence[Path]) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {path}")
        shapes.append((int(frame.shape[0]), int(frame.shape[1])))
    return shapes


def _observation_at(
    observations: Sequence[TrackObservation], frame_index: int
) -> TrackObservation | None:
    return next(
        (item for item in observations if item.frame_index == frame_index), None
    )


def _point_in_expanded_box(
    point: tuple[float, float],
    box: Sequence[float],
    margin: float,
) -> tuple[bool, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    expanded = (
        x1 - width * margin,
        y1 - height * margin,
        x2 + width * margin,
        y2 + height * margin,
    )
    inside = expanded[0] <= point[0] <= expanded[2] and expanded[1] <= point[1] <= expanded[3]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
    normalized_distance = float(
        np.linalg.norm(np.asarray(point, dtype=float) - center)
        / max(np.hypot(width, height), 1.0)
    )
    return inside, normalized_distance


def _feature_quality(feature_row: np.ndarray) -> float:
    indices = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    ball = max(0.0, float(feature_row[indices["ball_valid"]]))
    player = max(0.0, float(feature_row[indices["player_valid"]]))
    pose = max(0.0, float(feature_row[indices["pose_valid_fraction"]]))
    arm_distance = float(feature_row[indices["arm_min_distance"]])
    proximity = (
        max(0.0, 1.0 - min(arm_distance, 1.0)) if arm_distance > 0 else 0.0
    )
    return 1.0 + ball + player + pose + proximity


def associate_handball_actor(
    tracks: dict[int, list[TrackObservation]],
    frame_shapes: Sequence[tuple[int, int]],
    selected_features: np.ndarray,
    selected_indices: Sequence[int],
    metadata: dict[str, Any],
    config: PRTReIDConfig,
) -> ActorAssociation:
    if len(selected_features) != len(selected_indices):
        raise ValueError("Selected features and frame indices must have equal length")
    evidence: list[dict[str, Any]] = []

    closest_arm = metadata.get("closest_arm")
    if isinstance(closest_arm, dict):
        start, end = closest_arm.get("start"), closest_arm.get("end")
        if (
            isinstance(start, (list, tuple))
            and isinstance(end, (list, tuple))
            and len(start) >= 2
            and len(end) >= 2
        ):
            point = (
                (float(start[0]) + float(end[0])) / 2,
                (float(start[1]) + float(end[1])) / 2,
            )
            center_index = int(metadata.get("frames_before", len(frame_shapes) // 2))
            candidates: list[tuple[float, int, TrackObservation]] = []
            for track_id, observations in tracks.items():
                observation = _observation_at(observations, center_index)
                if observation is None:
                    continue
                inside, distance = _point_in_expanded_box(
                    point, observation.box, config.center_box_margin
                )
                if not inside:
                    continue
                score = max(0.0, min(1.0, 1.0 - 0.25 * distance))
                candidates.append((score, track_id, observation))
                evidence.append(
                    {
                        "source": "closest_arm",
                        "frame_index": center_index,
                        "track_id": track_id,
                        "score": score,
                        "normalized_center_distance": distance,
                        "point": list(point),
                        "bbox": list(observation.box),
                    }
                )
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                top_score, top_track, _ = candidates[0]
                runner_up = candidates[1][0] if len(candidates) > 1 else 0.0
                margin = top_score - runner_up
                conflicting = (
                    len(candidates) > 1
                    and margin < config.minimum_association_margin
                )
                confident = (
                    top_score >= config.minimum_association_score
                    and not conflicting
                )
                return ActorAssociation(
                    top_track,
                    "closest_arm",
                    top_score,
                    margin,
                    1,
                    confident,
                    conflicting,
                    tuple(evidence),
                )

    scores: dict[int, float] = {}
    votes: dict[int, int] = {}
    total_weight = 0.0
    seen_indices: set[int] = set()
    for feature_row, raw_frame_index in zip(selected_features, selected_indices):
        frame_index = int(raw_frame_index)
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)
        if frame_index < 0 or frame_index >= len(frame_shapes):
            continue
        height, width = frame_shapes[frame_index]
        actor_box = selected_player_box(feature_row, width, height)
        if actor_box is None:
            continue
        weight = _feature_quality(feature_row)
        total_weight += weight
        matches: list[tuple[float, float, float, int, TrackObservation]] = []
        for track_id, observations in tracks.items():
            observation = _observation_at(observations, frame_index)
            if observation is None:
                continue
            coverage = box_actor_coverage(actor_box, observation.box)
            iou = box_iou(actor_box, observation.box)
            if (
                coverage < config.minimum_anchor_coverage
                and iou < config.minimum_anchor_iou
            ):
                continue
            overlap_score = max(coverage, iou)
            matches.append(
                (overlap_score, coverage, iou, track_id, observation)
            )
        if not matches:
            evidence.append(
                {
                    "source": "selected_box",
                    "frame_index": frame_index,
                    "track_id": None,
                    "weight": weight,
                    "bbox": list(actor_box),
                }
            )
            continue
        matches.sort(key=lambda item: item[0], reverse=True)
        for overlap_score, coverage, iou, track_id, observation in matches:
            scores[track_id] = scores.get(track_id, 0.0) + weight * overlap_score
            votes[track_id] = votes.get(track_id, 0) + 1
            evidence.append(
                {
                    "source": "selected_box",
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "weight": weight,
                    "overlap_score": overlap_score,
                    "actor_coverage": coverage,
                    "iou": iou,
                    "selected_bbox": list(actor_box),
                    "tracked_bbox": list(observation.box),
                }
            )
    if not scores or total_weight <= 0:
        return ActorAssociation(
            None, "none", 0.0, 0.0, 0, False, False, tuple(evidence)
        )
    ranked = sorted(
        (
            (weighted_score / total_weight, track_id)
            for track_id, weighted_score in scores.items()
        ),
        reverse=True,
    )
    top_score, top_track = ranked[0]
    runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = top_score - runner_up
    top_votes = votes.get(top_track, 0)
    conflicting = len(ranked) > 1 and margin < config.minimum_association_margin
    confident = (
        top_score >= config.minimum_association_score
        and margin >= config.minimum_association_margin
        and top_votes >= config.minimum_anchor_votes
        and not conflicting
    )
    return ActorAssociation(
        top_track,
        "selected_boxes",
        top_score,
        margin,
        top_votes,
        confident,
        conflicting,
        tuple(evidence),
    )


class PRTReIDWorkerClient:
    """Persistent JSON-lines client for the isolated Python 3.9 worker."""

    def __init__(
        self,
        config: PRTReIDConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        environment = os.environ.copy()
        environment.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        environment.setdefault("MPLCONFIGDIR", "/tmp/prtreid-mpl")
        environment.setdefault("XDG_CACHE_HOME", "/tmp/prtreid-cache")
        self._stderr: deque[str] = deque(maxlen=40)
        command = [
            *config.worker_command,
            "--checkpoint",
            str(config.prtreid_checkpoint),
            "--batch-size",
            str(config.worker_batch_size),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not start PRTReID worker: {config.worker_command[0]}"
            ) from exc
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.close()
            raise RuntimeError("PRTReID worker did not expose standard streams")
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        for line in self.process.stderr:
            self._stderr.append(line.rstrip())

    def _error_context(self) -> str:
        return "\n".join(self._stderr)

    def _request(self, crops: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.process.poll() is not None:
            raise RuntimeError(
                "PRTReID worker exited before inference.\n" + self._error_context()
            )
        request_id = uuid.uuid4().hex
        payload = {"request_id": request_id, "crops": list(crops)}
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        try:
            self.process.stdin.write(
                json.dumps(payload, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(
                "PRTReID worker closed its input.\n" + self._error_context()
            ) from exc
        ready, _, _ = select.select(
            [self.process.stdout],
            [],
            [],
            self.config.worker_timeout_seconds,
        )
        if not ready:
            raise TimeoutError(
                "Timed out waiting for PRTReID worker.\n" + self._error_context()
            )
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError(
                "PRTReID worker exited without a response.\n" + self._error_context()
            )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"PRTReID worker returned invalid JSON: {response_line[:200]}"
            ) from exc
        if response.get("request_id") != request_id:
            raise RuntimeError("PRTReID worker response ID did not match request")
        if response.get("ok") is not True:
            raise RuntimeError(
                f"PRTReID worker rejected request: {response.get('error')}"
            )
        predictions = response.get("predictions", response.get("results"))
        if not isinstance(predictions, list) or len(predictions) != len(crops):
            raise RuntimeError("PRTReID worker returned the wrong prediction count")
        return predictions

    def predict(
        self, crops: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        predictions: list[dict[str, Any]] = []
        for start in range(0, len(crops), self.config.worker_batch_size):
            predictions.extend(
                self._request(crops[start : start + self.config.worker_batch_size])
            )
            if self.progress_callback is not None:
                self.progress_callback(len(predictions), len(crops))
        return predictions

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def __enter__(self) -> "PRTReIDWorkerClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _jersey_lab(
    frame: np.ndarray,
    box: Sequence[float],
    config: PRTReIDConfig,
) -> list[float] | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width, box_height = x2 - x1, y2 - y1
    left = int(np.clip(x1 + box_width * config.jersey_crop_left, 0, width))
    right = int(np.clip(x1 + box_width * config.jersey_crop_right, 0, width))
    top = int(np.clip(y1 + box_height * config.jersey_crop_top, 0, height))
    bottom = int(np.clip(y1 + box_height * config.jersey_crop_bottom, 0, height))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    not_dark = hsv[:, :, 2] >= 35
    likely_grass = (
        (hsv[:, :, 0] >= 35)
        & (hsv[:, :, 0] <= 90)
        & (hsv[:, :, 1] >= 45)
    )
    pixels = lab[not_dark & ~likely_grass]
    if len(pixels) < config.jersey_minimum_pixels:
        return None
    return [float(value) for value in np.median(pixels, axis=0)]


def _assign_jersey_colors(
    frame_paths: Sequence[Path],
    tracks: dict[int, list[TrackObservation]],
    config: PRTReIDConfig,
) -> dict[int, dict[str, Any]]:
    by_frame: dict[int, list[tuple[int, TrackObservation]]] = {}
    for track_id, observations in tracks.items():
        for observation in observations:
            by_frame.setdefault(observation.frame_index, []).append(
                (track_id, observation)
            )
    for frame_index, entries in by_frame.items():
        if frame_index < 0 or frame_index >= len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            raise RuntimeError(f"Could not read jersey frame: {frame_paths[frame_index]}")
        for _, observation in entries:
            observation.jersey_lab = _jersey_lab(frame, observation.box, config)

    medians: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}
    for track_id, observations in tracks.items():
        samples = [
            np.asarray(item.jersey_lab, dtype=float)
            for item in observations
            if item.jersey_lab is not None
        ]
        if (
            len(observations) >= config.minimum_track_frames
            and len(samples) >= min(2, config.minimum_track_frames)
        ):
            medians[track_id] = np.median(np.stack(samples), axis=0)
            sample_counts[track_id] = len(samples)
    result: dict[int, dict[str, Any]] = {}
    for track_id in tracks:
        median = medians.get(track_id)
        peers = [
            value
            for other_track, value in medians.items()
            if other_track != track_id
        ]
        nearest = (
            min(float(np.linalg.norm(median - peer)) for peer in peers)
            if median is not None and peers
            else None
        )
        outlier = (
            min(1.0, nearest / config.jersey_outlier_distance)
            if nearest is not None
            else 0.0
        )
        result[track_id] = {
            "median_lab": median.tolist() if median is not None else None,
            "sample_frames": sample_counts.get(track_id, 0),
            "comparison_tracks": len(peers),
            "nearest_distance": nearest,
            "outlier_score": outlier,
        }
    return result


def _aggregate_track(
    observations: Sequence[TrackObservation],
    color_evidence: dict[str, Any],
    source_frame_count: int,
    config: PRTReIDConfig,
) -> dict[str, Any]:
    unique: dict[int, dict[str, Any]] = {}
    for observation in observations:
        if observation.worker_prediction is not None:
            unique.setdefault(observation.frame_index, observation.worker_prediction)
    predictions = list(unique.values())
    votes = {role: 0 for role in ROLE_NAMES}
    probability_rows: list[list[float]] = []
    for prediction in predictions:
        role = str(prediction.get("predicted_role", ""))
        probabilities = prediction.get("role_probabilities")
        if role in votes:
            votes[role] += 1
        if not isinstance(probabilities, dict):
            continue
        try:
            probability_rows.append(
                [float(probabilities[name]) for name in ROLE_NAMES]
            )
        except (KeyError, TypeError, ValueError):
            continue
    prediction_count = len(probability_rows)
    mean_probabilities = (
        np.mean(np.asarray(probability_rows, dtype=float), axis=0)
        if probability_rows
        else np.zeros(len(ROLE_NAMES), dtype=float)
    )
    vote_total = sum(votes.values())
    vote_scores = {
        role: votes[role] / vote_total if vote_total else 0.0
        for role in ROLE_NAMES
    }
    model_scores = {
        role: 0.5 * float(mean_probabilities[index]) + 0.5 * vote_scores[role]
        for index, role in enumerate(ROLE_NAMES)
    }
    combined_scores = dict(model_scores)
    color_outlier = float(color_evidence.get("outlier_score", 0.0))
    comparison_tracks = int(color_evidence.get("comparison_tracks", 0))
    if (
        color_outlier >= config.color_activation
        and comparison_tracks >= config.jersey_minimum_comparison_tracks
    ):
        combined_scores["goalkeeper"] = min(
            1.0,
            combined_scores["goalkeeper"] + config.color_weight * color_outlier,
        )
    ranked = sorted(
        ((score, role) for role, score in combined_scores.items()), reverse=True
    )
    top_score, top_role = ranked[0]
    second_score = ranked[1][0]
    margin = top_score - second_score
    model_dominant = max(model_scores, key=model_scores.get)
    coverage = prediction_count / source_frame_count if source_frame_count else 0.0
    observation_coverage = (
        len(observations) / source_frame_count if source_frame_count else 0.0
    )
    enough = (
        prediction_count >= config.minimum_predictions
        and coverage >= config.minimum_role_coverage
        and len(observations) >= config.minimum_track_frames
    )
    predicted_role = "unknown"
    is_goalkeeper: bool | None = None
    if (
        enough
        and model_dominant == "goalkeeper"
        and top_role == "goalkeeper"
        and top_score >= config.goalkeeper_threshold
        and margin >= config.minimum_role_margin
    ):
        predicted_role = "goalkeeper"
        is_goalkeeper = True
    elif (
        enough
        and model_dominant in ("player", "referee")
        and top_role == model_dominant
        and top_score >= config.role_threshold
        and margin >= config.minimum_role_margin
    ):
        predicted_role = model_dominant
        is_goalkeeper = False
    return {
        "predicted_role": predicted_role,
        "is_goalkeeper": is_goalkeeper,
        "model_scores": model_scores,
        "combined_scores": combined_scores,
        "vote_counts": votes,
        "goalkeeper_score": combined_scores["goalkeeper"],
        "role_confidence": top_score,
        "margin": margin,
        "prediction_frames": prediction_count,
        "coverage": coverage,
        "track_observation_coverage": observation_coverage,
        "uncertain": is_goalkeeper is None,
    }


def classify_actor_role(
    frame_paths: Sequence[Path],
    selected_features: np.ndarray,
    selected_indices: Sequence[int],
    metadata: dict[str, Any],
    config: PRTReIDConfig,
    tracker: PersonTracker | None = None,
    worker: RoleWorker | None = None,
) -> dict[str, Any]:
    if not frame_paths:
        raise ValueError("Cannot classify roles for an empty frame sequence")
    tracks = track_all_people(frame_paths, config, tracker)
    shapes = _frame_shapes(frame_paths)
    color_by_track = _assign_jersey_colors(frame_paths, tracks, config)

    crop_references: list[tuple[int, TrackObservation]] = []
    crops: list[dict[str, Any]] = []
    for track_id, observations in sorted(tracks.items()):
        if len(observations) < config.minimum_track_frames:
            continue
        for observation in observations:
            height, width = shapes[observation.frame_index]
            x1, y1, x2, y2 = observation.box
            if min(x2, width) - max(x1, 0) < 8:
                continue
            if min(y2, height) - max(y1, 0) < 16:
                continue
            crop_references.append((track_id, observation))
            crops.append(
                {
                    "frame_path": str(observation.frame_path),
                    "bbox": list(observation.box),
                }
            )

    created_worker: PRTReIDWorkerClient | None = None
    if crops:
        active_worker = worker
        if active_worker is None:
            created_worker = PRTReIDWorkerClient(config)
            active_worker = created_worker
        try:
            predictions = active_worker.predict(crops)
        finally:
            if created_worker is not None:
                created_worker.close()
        if len(predictions) != len(crop_references):
            raise RuntimeError("Role worker prediction count does not match crops")
        for (_, observation), prediction in zip(crop_references, predictions):
            observation.worker_prediction = prediction

    track_results: list[dict[str, Any]] = []
    aggregate_by_track: dict[int, dict[str, Any]] = {}
    for track_id, observations in sorted(tracks.items()):
        color_evidence = color_by_track.get(
            track_id,
            {
                "median_lab": None,
                "sample_frames": 0,
                "comparison_tracks": 0,
                "nearest_distance": None,
                "outlier_score": 0.0,
            },
        )
        aggregate = _aggregate_track(
            observations, color_evidence, len(frame_paths), config
        )
        aggregate_by_track[track_id] = aggregate
        track_results.append(
            {
                "track_id": track_id,
                "frame_count": len(observations),
                "observations": [item.as_dict() for item in observations],
                "jersey_color": color_evidence,
                "aggregate": aggregate,
            }
        )

    association = associate_handball_actor(
        tracks,
        shapes,
        selected_features,
        selected_indices,
        metadata,
        config,
    )
    actor_aggregate = (
        aggregate_by_track.get(association.track_id)
        if association.track_id is not None
        else None
    )
    if association.confident and actor_aggregate is not None:
        predicted_role = actor_aggregate["predicted_role"]
        is_goalkeeper = actor_aggregate["is_goalkeeper"]
        goalkeeper_score = actor_aggregate["goalkeeper_score"]
        role_confidence = actor_aggregate["role_confidence"]
    else:
        predicted_role = "unknown"
        is_goalkeeper = None
        goalkeeper_score = (
            actor_aggregate["goalkeeper_score"]
            if actor_aggregate is not None
            else 0.0
        )
        role_confidence = (
            actor_aggregate["role_confidence"]
            if actor_aggregate is not None
            else 0.0
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": prtreid_config_fingerprint(config),
        "detector_checkpoint": str(config.person_detector),
        "prtreid_checkpoint": str(config.prtreid_checkpoint),
        "frame_count": len(frame_paths),
        "tracked_people": len(tracks),
        "classified_crops": len(crops),
        "actor_track_id": association.track_id,
        "predicted_role": predicted_role,
        "is_goalkeeper": is_goalkeeper,
        "goalkeeper_score": goalkeeper_score,
        "role_confidence": role_confidence,
        "uncertain": is_goalkeeper is None,
        "association": association.as_dict(),
        "tracks": track_results,
    }


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "size": None, "mtime_ns": None}
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prtreid_config_fingerprint(config: PRTReIDConfig) -> str:
    worker_source = PROJECT_ROOT / "training" / "prtreid_worker.py"
    worker_requirements = PROJECT_ROOT / "requirements-prtreid.txt"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "worker_sha256": (
            hashlib.sha256(worker_source.read_bytes()).hexdigest()
            if worker_source.is_file()
            else None
        ),
        "worker_requirements_sha256": (
            hashlib.sha256(worker_requirements.read_bytes()).hexdigest()
            if worker_requirements.is_file()
            else None
        ),
        "runtime_versions": {
            "ultralytics": _package_version("ultralytics"),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "person_detector": _file_identity(config.person_detector),
        "prtreid_checkpoint": _file_identity(config.prtreid_checkpoint),
        "device": config.device,
        "tracker": config.tracker,
        "confidence": config.confidence,
        "image_size": config.image_size,
        "minimum_track_frames": config.minimum_track_frames,
        "worker_command": list(config.worker_command),
        "worker_batch_size": config.worker_batch_size,
        "association": {
            "center_box_margin": config.center_box_margin,
            "minimum_anchor_coverage": config.minimum_anchor_coverage,
            "minimum_anchor_iou": config.minimum_anchor_iou,
            "minimum_anchor_votes": config.minimum_anchor_votes,
            "minimum_score": config.minimum_association_score,
            "minimum_margin": config.minimum_association_margin,
        },
        "aggregation": {
            "minimum_predictions": config.minimum_predictions,
            "minimum_coverage": config.minimum_role_coverage,
            "role_threshold": config.role_threshold,
            "goalkeeper_threshold": config.goalkeeper_threshold,
            "minimum_role_margin": config.minimum_role_margin,
            "color_weight": config.color_weight,
            "color_activation": config.color_activation,
        },
        "jersey_color": {
            "crop": [
                config.jersey_crop_top,
                config.jersey_crop_bottom,
                config.jersey_crop_left,
                config.jersey_crop_right,
            ],
            "minimum_pixels": config.jersey_minimum_pixels,
            "minimum_comparison_tracks": (
                config.jersey_minimum_comparison_tracks
            ),
            "outlier_distance": config.jersey_outlier_distance,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prtreid_source_fingerprint(
    feature_artifact: Path,
    frame_paths: Sequence[Path],
    selected_indices: Sequence[int],
    metadata_path: Path | None = None,
) -> str:
    if not feature_artifact.is_file():
        raise FileNotFoundError(f"Feature artifact not found: {feature_artifact}")
    frames: list[dict[str, Any]] = []
    for index, frame_path in enumerate(frame_paths):
        stat = frame_path.stat()
        frames.append(
            {
                "index": index,
                "path": str(frame_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_path": str(feature_artifact),
        "feature_sha256": hashlib.sha256(feature_artifact.read_bytes()).hexdigest(),
        "selected_indices": [int(index) for index in selected_indices],
        "all_frames": frames,
        "metadata": (
            {
                "path": str(metadata_path),
                "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            }
            if metadata_path is not None and metadata_path.is_file()
            else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prtreid_result_is_current(
    destination: Path,
    config: PRTReIDConfig,
    source_fingerprint: str,
) -> bool:
    if not destination.is_file():
        return False
    try:
        result = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == SCHEMA_VERSION
        and result.get("config_fingerprint") == prtreid_config_fingerprint(config)
        and result.get("source_fingerprint") == source_fingerprint
    )


def save_prtreid_result(result: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.json")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(destination)
