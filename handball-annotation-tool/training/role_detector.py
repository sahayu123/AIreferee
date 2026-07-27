from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np
import yaml

from handball_annotator.runtime import get_device

from .config import project_path
from .features import FEATURE_NAMES

ROLES = ("goalkeeper", "player", "referee")
ROLE_COLORS = {
    "goalkeeper": (0, 165, 255),
    "player": (80, 255, 80),
    "referee": (255, 120, 80),
}


@dataclass(frozen=True)
class RoleConfig:
    checkpoint: Path
    device: str
    confidence: float
    image_size: int
    minimum_actor_coverage: float
    minimum_matches: int
    minimum_coverage: float
    role_vote_threshold: float
    goalkeeper_vote_threshold: float
    manifest: Path
    features_dir: Path
    roles_dir: Path
    audits_dir: Path
    report: Path
    logs_dir: Path


@dataclass(frozen=True)
class RoleDetection:
    role: str
    confidence: float
    box: tuple[float, float, float, float]


class RoleDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[RoleDetection]:
        ...


def load_role_config(path: str | Path) -> RoleConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Role configuration not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        model = raw["model"]
        association = raw.get("association", {})
        aggregation = raw.get("aggregation", {})
        paths = raw["paths"]
        config = RoleConfig(
            checkpoint=project_path(model["checkpoint"]),
            device=str(model.get("device", "auto")),
            confidence=float(model.get("confidence", 0.20)),
            image_size=int(model.get("image_size", 640)),
            minimum_actor_coverage=float(
                association.get("minimum_actor_coverage", 0.20)
            ),
            minimum_matches=int(aggregation.get("minimum_matches", 3)),
            minimum_coverage=float(aggregation.get("minimum_coverage", 0.25)),
            role_vote_threshold=float(aggregation.get("role_vote_threshold", 0.55)),
            goalkeeper_vote_threshold=float(
                aggregation.get("goalkeeper_vote_threshold", 0.70)
            ),
            manifest=project_path(paths["manifest"]),
            features_dir=project_path(paths["features"]),
            roles_dir=project_path(paths["roles"]),
            audits_dir=project_path(paths["audits"]),
            report=project_path(paths["report"]),
            logs_dir=project_path(paths["logs"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid role configuration: {exc}") from exc
    if not 0 <= config.confidence <= 1:
        raise ValueError("Role detector confidence must be between 0 and 1")
    if not 0 <= config.minimum_actor_coverage <= 1:
        raise ValueError("minimum_actor_coverage must be between 0 and 1")
    if config.minimum_matches < 1:
        raise ValueError("minimum_matches must be at least 1")
    for name, value in (
        ("minimum_coverage", config.minimum_coverage),
        ("role_vote_threshold", config.role_vote_threshold),
        ("goalkeeper_vote_threshold", config.goalkeeper_vote_threshold),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    return config


class FootballRoleDetector:
    """Local Ultralytics wrapper for the football role checkpoint."""

    def __init__(self, config: RoleConfig):
        if not config.checkpoint.is_file():
            raise FileNotFoundError(
                f"Football role checkpoint not found: {config.checkpoint}. "
                "Run `python -m training.download_models --with-role-detector`."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics before detecting player roles.") from exc
        self.config = config
        self.device = get_device(config.device)
        self.model = YOLO(str(config.checkpoint))
        model_names = self.model.names
        names = model_names.values() if isinstance(model_names, dict) else model_names
        normalized = {
            "goalkeeper" if str(name).strip().lower() == "goalie"
            else str(name).strip().lower()
            for name in names
        }
        missing = set(ROLES) - normalized
        if missing:
            raise ValueError(
                f"Football role checkpoint is missing required classes: {sorted(missing)}"
            )

    def detect(self, frame: np.ndarray) -> list[RoleDetection]:
        result = self.model.predict(
            source=frame,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        detections: list[RoleDetection] = []
        for box, class_id, confidence in zip(boxes, classes, confidences):
            role = str(result.names[int(class_id)]).strip().lower()
            if role == "goalie":
                role = "goalkeeper"
            if role not in ROLES:
                continue
            detections.append(RoleDetection(
                role,
                float(confidence),
                tuple(float(value) for value in box),
            ))
        return detections


def box_iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def box_overlap_min_area(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    smaller = min(first_area, second_area)
    return intersection / smaller if smaller > 0 else 0.0


def box_actor_coverage(
    actor_box: Sequence[float],
    detection_box: Sequence[float],
) -> float:
    left = max(float(actor_box[0]), float(detection_box[0]))
    top = max(float(actor_box[1]), float(detection_box[1]))
    right = min(float(actor_box[2]), float(detection_box[2]))
    bottom = min(float(actor_box[3]), float(detection_box[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    actor_area = max(0.0, float(actor_box[2]) - float(actor_box[0])) * max(
        0.0, float(actor_box[3]) - float(actor_box[1])
    )
    return intersection / actor_area if actor_area > 0 else 0.0


def selected_player_box(
    feature_row: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float, float, float] | None:
    index = {name: FEATURE_NAMES.index(name) for name in (
        "player_x", "player_y", "player_w", "player_h", "player_valid"
    )}
    if float(feature_row[index["player_valid"]]) <= 0:
        return None
    center_x = float(feature_row[index["player_x"]]) * frame_width
    center_y = float(feature_row[index["player_y"]]) * frame_height
    width = float(feature_row[index["player_w"]]) * frame_width
    height = float(feature_row[index["player_h"]]) * frame_height
    if width <= 0 or height <= 0:
        return None
    return (
        max(0.0, center_x - width / 2),
        max(0.0, center_y - height / 2),
        min(float(frame_width), center_x + width / 2),
        min(float(frame_height), center_y + height / 2),
    )


def match_selected_player(
    selected_box: Sequence[float] | None,
    detections: Sequence[RoleDetection],
    minimum_actor_coverage: float,
) -> tuple[RoleDetection | None, float]:
    if selected_box is None or not detections:
        return None, 0.0
    scored = [
        (box_actor_coverage(selected_box, detection.box), detection)
        for detection in detections
    ]
    overlap, detection = max(
        scored,
        key=lambda item: (item[0], item[1].confidence),
    )
    return (
        (detection, overlap)
        if overlap >= minimum_actor_coverage
        else (None, overlap)
    )


def aggregate_role_evidence(
    evidence: Sequence[dict[str, object]],
    config: RoleConfig,
) -> dict[str, object]:
    unique_evidence: list[dict[str, object]] = []
    seen_frame_indices: set[int] = set()
    for item in evidence:
        frame_index = item.get("frame_index")
        if frame_index is not None:
            normalized_index = int(frame_index)
            if normalized_index in seen_frame_indices:
                continue
            seen_frame_indices.add(normalized_index)
        unique_evidence.append(item)
    valid_frames = sum(
        item["selected_box"] is not None for item in unique_evidence
    )
    matched = [
        item for item in unique_evidence if item["matched_role"] is not None
    ]
    weights = {role: 0.0 for role in ROLES}
    counts = {role: 0 for role in ROLES}
    for item in matched:
        role = str(item["matched_role"])
        weight = float(item["confidence"]) * float(item["actor_coverage"])
        weights[role] += weight
        counts[role] += 1
    total_weight = sum(weights.values())
    scores = {
        role: weights[role] / total_weight if total_weight > 0 else 0.0
        for role in ROLES
    }
    matched_count = len(matched)
    coverage = matched_count / valid_frames if valid_frames else 0.0
    enough_evidence = (
        matched_count >= config.minimum_matches
        and coverage >= config.minimum_coverage
        and total_weight > 0
    )
    dominant = max(ROLES, key=lambda role: scores[role]) if total_weight else "unknown"
    top_score = scores[dominant] if dominant != "unknown" else 0.0
    predicted_role = "unknown"
    if not enough_evidence:
        is_goalkeeper: bool | None = None
    elif (
        dominant == "goalkeeper"
        and scores["goalkeeper"] >= config.goalkeeper_vote_threshold
    ):
        is_goalkeeper = True
        predicted_role = "goalkeeper"
    elif dominant in ("player", "referee") and top_score >= config.role_vote_threshold:
        is_goalkeeper = False
        predicted_role = dominant
    else:
        is_goalkeeper = None
    return {
        "predicted_role": predicted_role,
        "is_goalkeeper": is_goalkeeper,
        "goalkeeper_score": scores["goalkeeper"],
        "role_confidence": top_score,
        "matched_frames": matched_count,
        "valid_selected_frames": valid_frames,
        "selected_frame_count": len(evidence),
        "unique_selected_frame_count": len(unique_evidence),
        "coverage": coverage,
        "vote_scores": scores,
        "vote_counts": counts,
        "uncertain": is_goalkeeper is None,
    }


def classify_selected_actor(
    detector: RoleDetector,
    frame_paths: Sequence[Path],
    features: np.ndarray,
    selected_indices: Sequence[int],
    config: RoleConfig,
    base_overlays: Sequence[np.ndarray] | None = None,
) -> tuple[dict[str, object], list[np.ndarray]]:
    if len(features) != len(selected_indices):
        raise ValueError(
            f"Feature rows ({len(features)}) do not match selected frames "
            f"({len(selected_indices)})"
        )
    if base_overlays is not None and len(base_overlays) != len(selected_indices):
        raise ValueError("Base overlays must match the selected frame count")
    evidence: list[dict[str, object]] = []
    overlays: list[np.ndarray] = []
    frame_cache: dict[int, np.ndarray] = {}
    detection_cache: dict[int, list[RoleDetection]] = {}
    for position, (feature_row, frame_index) in enumerate(zip(features, selected_indices)):
        if frame_index < 0 or frame_index >= len(frame_paths):
            raise IndexError(
                f"Selected frame index {frame_index} is outside 0..{len(frame_paths) - 1}"
            )
        if frame_index not in frame_cache:
            loaded_frame = cv2.imread(str(frame_paths[frame_index]))
            if loaded_frame is None:
                raise RuntimeError(
                    f"Could not read role-detection frame: {frame_paths[frame_index]}"
                )
            frame_cache[frame_index] = loaded_frame
        frame = frame_cache[frame_index]
        height, width = frame.shape[:2]
        selected_box = selected_player_box(feature_row, width, height)
        if selected_box is not None and frame_index not in detection_cache:
            detection_cache[frame_index] = detector.detect(frame)
        detections = detection_cache.get(frame_index, [])
        matched, overlap = match_selected_player(
            selected_box, detections, config.minimum_actor_coverage
        )
        item: dict[str, object] = {
            "position": position,
            "frame_index": int(frame_index),
            "frame_name": frame_paths[frame_index].name,
            "selected_box": list(selected_box) if selected_box is not None else None,
            "matched_role": matched.role if matched is not None else None,
            "confidence": matched.confidence if matched is not None else 0.0,
            "actor_coverage": overlap if matched is not None else 0.0,
            "iou": (
                box_iou(selected_box, matched.box)
                if selected_box is not None and matched is not None
                else 0.0
            ),
            "matched_box": list(matched.box) if matched is not None else None,
            "detections": [
                {
                    "role": detection.role,
                    "confidence": detection.confidence,
                    "box": list(detection.box),
                }
                for detection in detections
            ],
        }
        evidence.append(item)
        overlay = (
            base_overlays[position].copy()
            if base_overlays is not None
            else frame.copy()
        )
        if selected_box is not None:
            x1, y1, x2, y2 = (int(value) for value in selected_box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                overlay,
                "handball actor",
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.box)
            color = ROLE_COLORS[detection.role]
            thickness = 3 if detection is matched else 1
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                overlay,
                f"{detection.role} {detection.confidence:.2f}",
                (x1, min(overlay.shape[0] - 5, max(18, y1 + 18))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
                cv2.LINE_AA,
            )
        overlays.append(overlay)
    summary = aggregate_role_evidence(evidence, config)
    summary.update({
        "schema_version": 1,
        "config_fingerprint": role_config_fingerprint(config),
        "detector_checkpoint": str(config.checkpoint),
        "selected_frame_indices": [int(index) for index in selected_indices],
        "per_frame": evidence,
    })
    return summary, overlays


def role_config_fingerprint(config: RoleConfig) -> str:
    checkpoint = config.checkpoint
    checkpoint_identity = {
        "path": str(checkpoint),
        "size": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "mtime_ns": checkpoint.stat().st_mtime_ns if checkpoint.is_file() else None,
    }
    payload = {
        "schema_version": 1,
        "checkpoint": checkpoint_identity,
        "confidence": config.confidence,
        "image_size": config.image_size,
        "minimum_actor_coverage": config.minimum_actor_coverage,
        "minimum_matches": config.minimum_matches,
        "minimum_coverage": config.minimum_coverage,
        "role_vote_threshold": config.role_vote_threshold,
        "goalkeeper_vote_threshold": config.goalkeeper_vote_threshold,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def role_source_fingerprint(
    feature_artifact: Path,
    frame_paths: Sequence[Path],
    selected_indices: Sequence[int],
) -> str:
    if not feature_artifact.is_file():
        raise FileNotFoundError(f"Feature artifact not found: {feature_artifact}")
    selected_frames: list[dict[str, object]] = []
    for frame_index in selected_indices:
        if frame_index < 0 or frame_index >= len(frame_paths):
            raise IndexError(
                f"Selected frame index {frame_index} is outside "
                f"0..{len(frame_paths) - 1}"
            )
        frame_path = frame_paths[frame_index]
        frame_stat = frame_path.stat()
        selected_frames.append({
            "index": int(frame_index),
            "path": str(frame_path),
            "size": frame_stat.st_size,
            "mtime_ns": frame_stat.st_mtime_ns,
        })
    payload = {
        "schema_version": 1,
        "feature_path": str(feature_artifact),
        "feature_sha256": hashlib.sha256(feature_artifact.read_bytes()).hexdigest(),
        "selected_frames": selected_frames,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def role_result_is_current(
    destination: Path,
    config: RoleConfig,
    source_fingerprint: str,
) -> bool:
    if not destination.is_file():
        return False
    try:
        result = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        result.get("schema_version") == 1
        and result.get("config_fingerprint") == role_config_fingerprint(config)
        and result.get("source_fingerprint") == source_fingerprint
    )


def save_role_result(result: dict[str, object], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.json")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
