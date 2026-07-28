"""Independent goalkeeper evidence from team jerseys and wrist-localized gloves.

This module reuses the full-frame YOLO + ByteTrack implementation and actor
association from the PRTReID experiment.  It never modifies the raw handball
probability; callers may use a confirmed goalkeeper status as a final decision
veto.

The glove classifier is optional.  When no trained local checkpoint is
configured, glove evidence is represented as unavailable (``None``), never as
negative evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np
import yaml

from handball_annotator.runtime import get_device

from .config import project_path
from .glove_classifier import GloveClassifier
from .prtreid_role import (
    ActorAssociation,
    PRTReIDConfig,
    PRTReIDWorkerClient,
    PersonTracker,
    RoleWorker,
    TrackObservation,
    associate_handball_actor,
    load_prtreid_config,
    prtreid_config_fingerprint,
    track_all_people,
)
from .role_detector import box_iou

SCHEMA_VERSION = 1
ROLE_NAMES = ("ball", "goalkeeper", "other", "player", "referee")


@dataclass(frozen=True)
class JerseyGloveConfig:
    base_config_path: Path
    base_config: PRTReIDConfig
    use_prtreid_evidence: bool
    pose_model: Path
    pose_input_size: int
    pose_detection_confidence: float
    pose_presence_confidence: float
    pose_crop_margin: float
    wrist_confidence: float
    elbow_confidence: float
    shoulder_confidence: float
    minimum_forearm_ratio: float
    maximum_forearm_ratio: float
    minimum_upper_arm_ratio: float
    maximum_upper_arm_ratio: float
    hand_crop_side_ratio: float
    hand_minimum_native_side: int
    hand_maximum_native_side: int
    hand_maximum_clipped_fraction: float
    hand_minimum_blur_variance: float
    hand_minimum_distinct_frames: int
    hand_maximum_samples: int
    jersey_crop_top: float
    jersey_crop_bottom: float
    jersey_crop_left: float
    jersey_crop_right: float
    jersey_histogram_bins: int
    jersey_maximum_samples_per_track: int
    jersey_minimum_pixels: int
    jersey_minimum_track_frames: int
    jersey_minimum_comparison_tracks: int
    jersey_minimum_cluster_tracks: int
    jersey_cluster_candidates: int
    jersey_team_distance_scale: float
    glove_enabled: bool
    glove_checkpoint: Path
    glove_device: str
    glove_batch_size: int
    not_goalkeeper_team_match: float
    not_goalkeeper_player_score: float
    goalkeeper_jersey_outlier: float
    goalkeeper_glove_probability: float
    goalkeeper_maximum_player_score: float
    roles_dir: Path
    audits_dir: Path
    hand_crops_dir: Path
    report: Path
    logs_dir: Path


@dataclass
class HandCrop:
    frame_index: int
    side: str
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    pose_confidence: float
    blur_variance: float
    native_side: int
    clipped_fraction: float
    quality: float
    glove_probability: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "side": self.side,
            "bbox": list(self.bbox),
            "pose_confidence": self.pose_confidence,
            "blur_variance": self.blur_variance,
            "native_side": self.native_side,
            "clipped_fraction": self.clipped_fraction,
            "quality": self.quality,
            "glove_probability": self.glove_probability,
        }


class HandExtractor(Protocol):
    def extract(
        self,
        frame_paths: Sequence[Path],
        observations: Sequence[TrackObservation],
    ) -> tuple[list[HandCrop], dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


class GloveProbabilityModel(Protocol):
    def predict_glove_probability(
        self, crops: Sequence[np.ndarray]
    ) -> np.ndarray:
        ...


def load_jersey_glove_config(path: str | Path) -> JerseyGloveConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Jersey/glove configuration not found: {config_path}"
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("configuration must be a mapping")
        base = raw["base"]
        pose = raw["pose"]
        hands = raw["hands"]
        jersey = raw["jersey"]
        glove = raw["glove"]
        decision = raw["decision"]
        paths = raw["paths"]
        base_config_path = project_path(base["prtreid_config"])
        config = JerseyGloveConfig(
            base_config_path=base_config_path,
            base_config=load_prtreid_config(base_config_path),
            use_prtreid_evidence=bool(
                base.get("use_prtreid_evidence", True)
            ),
            pose_model=project_path(pose["model"]),
            pose_input_size=int(pose.get("input_size", 512)),
            pose_detection_confidence=float(
                pose.get("detection_confidence", 0.35)
            ),
            pose_presence_confidence=float(
                pose.get("presence_confidence", 0.35)
            ),
            pose_crop_margin=float(pose.get("crop_margin", 0.25)),
            wrist_confidence=float(pose.get("wrist_confidence", 0.65)),
            elbow_confidence=float(pose.get("elbow_confidence", 0.50)),
            shoulder_confidence=float(
                pose.get("shoulder_confidence", 0.40)
            ),
            minimum_forearm_ratio=float(
                pose.get("minimum_forearm_ratio", 0.06)
            ),
            maximum_forearm_ratio=float(
                pose.get("maximum_forearm_ratio", 0.30)
            ),
            minimum_upper_arm_ratio=float(
                pose.get("minimum_upper_arm_ratio", 0.04)
            ),
            maximum_upper_arm_ratio=float(
                pose.get("maximum_upper_arm_ratio", 0.30)
            ),
            hand_crop_side_ratio=float(
                hands.get("crop_side_ratio", 0.18)
            ),
            hand_minimum_native_side=int(
                hands.get("minimum_native_side", 32)
            ),
            hand_maximum_native_side=int(
                hands.get("maximum_native_side", 64)
            ),
            hand_maximum_clipped_fraction=float(
                hands.get("maximum_clipped_fraction", 0.25)
            ),
            hand_minimum_blur_variance=float(
                hands.get("minimum_blur_variance", 15.0)
            ),
            hand_minimum_distinct_frames=int(
                hands.get("minimum_distinct_frames", 3)
            ),
            hand_maximum_samples=int(
                hands.get("maximum_samples", 8)
            ),
            jersey_crop_top=float(jersey.get("crop_top", 0.15)),
            jersey_crop_bottom=float(jersey.get("crop_bottom", 0.60)),
            jersey_crop_left=float(jersey.get("crop_left", 0.20)),
            jersey_crop_right=float(jersey.get("crop_right", 0.80)),
            jersey_histogram_bins=int(jersey.get("histogram_bins", 8)),
            jersey_maximum_samples_per_track=int(
                jersey.get("maximum_samples_per_track", 8)
            ),
            jersey_minimum_pixels=int(jersey.get("minimum_pixels", 100)),
            jersey_minimum_track_frames=int(
                jersey.get("minimum_track_frames", 5)
            ),
            jersey_minimum_comparison_tracks=int(
                jersey.get("minimum_comparison_tracks", 4)
            ),
            jersey_minimum_cluster_tracks=int(
                jersey.get("minimum_cluster_tracks", 2)
            ),
            jersey_cluster_candidates=int(
                jersey.get("cluster_candidates", 3)
            ),
            jersey_team_distance_scale=float(
                jersey.get("team_distance_scale", 0.35)
            ),
            glove_enabled=bool(glove.get("enabled", False)),
            glove_checkpoint=project_path(glove["checkpoint"]),
            glove_device=str(glove.get("device", "cpu")),
            glove_batch_size=int(glove.get("batch_size", 16)),
            not_goalkeeper_team_match=float(
                decision.get("not_goalkeeper_team_match", 0.65)
            ),
            not_goalkeeper_player_score=float(
                decision.get("not_goalkeeper_player_score", 0.60)
            ),
            goalkeeper_jersey_outlier=float(
                decision.get("goalkeeper_jersey_outlier", 0.55)
            ),
            goalkeeper_glove_probability=float(
                decision.get("goalkeeper_glove_probability", 0.75)
            ),
            goalkeeper_maximum_player_score=float(
                decision.get("goalkeeper_maximum_player_score", 0.25)
            ),
            roles_dir=project_path(paths["roles"]),
            audits_dir=project_path(paths["audits"]),
            hand_crops_dir=project_path(paths["hand_crops"]),
            report=project_path(paths["report"]),
            logs_dir=project_path(paths["logs"]),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid jersey/glove configuration: {exc}") from exc
    _validate_config(config)
    return config


def _validate_config(config: JerseyGloveConfig) -> None:
    unit_values = (
        (
            "pose.detection_confidence",
            config.pose_detection_confidence,
        ),
        (
            "pose.presence_confidence",
            config.pose_presence_confidence,
        ),
        ("pose.crop_margin", config.pose_crop_margin),
        ("pose.wrist_confidence", config.wrist_confidence),
        ("pose.elbow_confidence", config.elbow_confidence),
        ("pose.shoulder_confidence", config.shoulder_confidence),
        ("pose.minimum_forearm_ratio", config.minimum_forearm_ratio),
        ("pose.maximum_forearm_ratio", config.maximum_forearm_ratio),
        ("pose.minimum_upper_arm_ratio", config.minimum_upper_arm_ratio),
        ("pose.maximum_upper_arm_ratio", config.maximum_upper_arm_ratio),
        ("hands.crop_side_ratio", config.hand_crop_side_ratio),
        (
            "hands.maximum_clipped_fraction",
            config.hand_maximum_clipped_fraction,
        ),
        ("jersey.crop_top", config.jersey_crop_top),
        ("jersey.crop_bottom", config.jersey_crop_bottom),
        ("jersey.crop_left", config.jersey_crop_left),
        ("jersey.crop_right", config.jersey_crop_right),
        (
            "decision.not_goalkeeper_team_match",
            config.not_goalkeeper_team_match,
        ),
        (
            "decision.not_goalkeeper_player_score",
            config.not_goalkeeper_player_score,
        ),
        (
            "decision.goalkeeper_jersey_outlier",
            config.goalkeeper_jersey_outlier,
        ),
        (
            "decision.goalkeeper_glove_probability",
            config.goalkeeper_glove_probability,
        ),
        (
            "decision.goalkeeper_maximum_player_score",
            config.goalkeeper_maximum_player_score,
        ),
    )
    for name, value in unit_values:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.minimum_forearm_ratio >= config.maximum_forearm_ratio:
        raise ValueError("minimum_forearm_ratio must be below maximum")
    if config.minimum_upper_arm_ratio >= config.maximum_upper_arm_ratio:
        raise ValueError("minimum_upper_arm_ratio must be below maximum")
    if config.jersey_crop_top >= config.jersey_crop_bottom:
        raise ValueError("jersey crop top must be above bottom")
    if config.jersey_crop_left >= config.jersey_crop_right:
        raise ValueError("jersey crop left must be left of right")
    positive_integers = (
        ("pose.input_size", config.pose_input_size),
        ("hands.minimum_native_side", config.hand_minimum_native_side),
        ("hands.maximum_native_side", config.hand_maximum_native_side),
        (
            "hands.minimum_distinct_frames",
            config.hand_minimum_distinct_frames,
        ),
        ("hands.maximum_samples", config.hand_maximum_samples),
        ("jersey.histogram_bins", config.jersey_histogram_bins),
        (
            "jersey.maximum_samples_per_track",
            config.jersey_maximum_samples_per_track,
        ),
        ("jersey.minimum_pixels", config.jersey_minimum_pixels),
        ("jersey.minimum_track_frames", config.jersey_minimum_track_frames),
        (
            "jersey.minimum_comparison_tracks",
            config.jersey_minimum_comparison_tracks,
        ),
        (
            "jersey.minimum_cluster_tracks",
            config.jersey_minimum_cluster_tracks,
        ),
        ("jersey.cluster_candidates", config.jersey_cluster_candidates),
        ("glove.batch_size", config.glove_batch_size),
    )
    for name, value in positive_integers:
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.hand_minimum_native_side > config.hand_maximum_native_side:
        raise ValueError("minimum hand crop side must not exceed maximum")
    if config.hand_maximum_samples < config.hand_minimum_distinct_frames:
        raise ValueError(
            "maximum hand samples must be at least minimum distinct frames"
        )
    if config.jersey_cluster_candidates < 2:
        raise ValueError("jersey cluster_candidates must be at least 2")
    if config.hand_minimum_blur_variance < 0:
        raise ValueError("minimum_blur_variance must be non-negative")
    if config.jersey_team_distance_scale <= 0:
        raise ValueError("team_distance_scale must be positive")
    if config.glove_enabled and not config.glove_checkpoint.is_file():
        raise FileNotFoundError(
            "Glove classification is enabled but the checkpoint is missing: "
            f"{config.glove_checkpoint}"
        )


def _sample_observations(
    observations: Sequence[TrackObservation], maximum: int
) -> list[TrackObservation]:
    if len(observations) <= maximum:
        return list(observations)
    indices = np.linspace(0, len(observations) - 1, maximum)
    return [observations[int(round(index))] for index in indices]


def torso_histogram(
    frame: np.ndarray,
    box: Sequence[float],
    config: JerseyGloveConfig,
) -> np.ndarray | None:
    """Return a normalized two-dimensional LAB chroma histogram."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Jersey frame must be a BGR uint8 image")
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width < 4 or box_height < 8:
        return None
    left = max(
        0, int(math.floor(x1 + config.jersey_crop_left * box_width))
    )
    right = min(
        width, int(math.ceil(x1 + config.jersey_crop_right * box_width))
    )
    top = max(
        0, int(math.floor(y1 + config.jersey_crop_top * box_height))
    )
    bottom = min(
        height, int(math.ceil(y1 + config.jersey_crop_bottom * box_height))
    )
    torso = frame[top:bottom, left:right]
    if torso.size == 0 or torso.shape[0] * torso.shape[1] < config.jersey_minimum_pixels:
        return None
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
    mask = np.zeros(torso.shape[:2], dtype=np.uint8)
    center = (torso.shape[1] // 2, torso.shape[0] // 2)
    axes = (
        max(1, int(0.46 * torso.shape[1])),
        max(1, int(0.48 * torso.shape[0])),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)
    if int(np.count_nonzero(mask)) < config.jersey_minimum_pixels:
        return None
    histogram = cv2.calcHist(
        [lab],
        [1, 2],
        mask,
        [config.jersey_histogram_bins, config.jersey_histogram_bins],
        [0, 256, 0, 256],
    ).astype(np.float32)
    total = float(histogram.sum())
    if total <= 0:
        return None
    return (histogram / total).reshape(-1)


def _histogram_embedding(histogram: np.ndarray) -> np.ndarray:
    """Map a probability histogram to unit Hellinger space."""

    embedded = np.sqrt(
        np.clip(np.asarray(histogram, dtype=np.float32), 0.0, None)
    )
    norm = float(np.linalg.norm(embedded))
    if norm <= 0:
        raise ValueError("Cannot embed an empty jersey histogram")
    return embedded / norm


def _hellinger_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(first, dtype=np.float32)
            - np.asarray(second, dtype=np.float32)
        )
        / math.sqrt(2.0)
    )


def _track_jersey_descriptors(
    frame_paths: Sequence[Path],
    tracks: dict[int, list[TrackObservation]],
    config: JerseyGloveConfig,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    descriptors: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}
    frame_cache: dict[int, np.ndarray] = {}
    for track_id, observations in sorted(tracks.items()):
        if len(observations) < config.jersey_minimum_track_frames:
            continue
        histograms: list[np.ndarray] = []
        for observation in _sample_observations(
            observations, config.jersey_maximum_samples_per_track
        ):
            if not 0 <= observation.frame_index < len(frame_paths):
                continue
            frame = frame_cache.get(observation.frame_index)
            if frame is None:
                frame = cv2.imread(str(frame_paths[observation.frame_index]))
                if frame is None:
                    raise RuntimeError(
                        "Could not read jersey frame: "
                        f"{frame_paths[observation.frame_index]}"
                    )
                frame_cache[observation.frame_index] = frame
            histogram = torso_histogram(frame, observation.box, config)
            if histogram is not None:
                histograms.append(histogram)
        if len(histograms) < min(2, config.jersey_minimum_track_frames):
            continue
        descriptor = np.median(np.stack(histograms), axis=0).astype(np.float32)
        total = float(descriptor.sum())
        if total <= 0:
            continue
        descriptors[int(track_id)] = descriptor / total
        sample_counts[int(track_id)] = len(histograms)
    return descriptors, sample_counts


def _overlapping_actor_fragment_ids(
    tracks: dict[int, list[TrackObservation]],
    actor_track_id: int | None,
    minimum_iou: float = 0.60,
) -> list[int]:
    if actor_track_id is None or actor_track_id not in tracks:
        return []
    actor_by_frame = {
        observation.frame_index: observation
        for observation in tracks[actor_track_id]
    }
    fragments: list[int] = []
    for track_id, observations in sorted(tracks.items()):
        if track_id == actor_track_id:
            continue
        maximum_overlap = max(
            (
                box_iou(
                    actor_by_frame[observation.frame_index].box,
                    observation.box,
                )
                for observation in observations
                if observation.frame_index in actor_by_frame
            ),
            default=0.0,
        )
        if maximum_overlap >= minimum_iou:
            fragments.append(track_id)
    return fragments


def compute_jersey_team_evidence(
    frame_paths: Sequence[Path],
    tracks: dict[int, list[TrackObservation]],
    actor_track_id: int | None,
    config: JerseyGloveConfig,
) -> dict[str, Any]:
    """Estimate whether the actor matches one of two dominant team colours."""

    descriptors, sample_counts = _track_jersey_descriptors(
        frame_paths, tracks, config
    )
    actor_descriptor = (
        descriptors.get(actor_track_id)
        if actor_track_id is not None
        else None
    )
    excluded_fragments = _overlapping_actor_fragment_ids(
        tracks, actor_track_id
    )
    peer_ids = [
        track_id
        for track_id in sorted(descriptors)
        if track_id != actor_track_id and track_id not in excluded_fragments
    ]
    unavailable: dict[str, Any] = {
        "available": False,
        "reason": None,
        "team_match_score": None,
        "outlier_score": None,
        "nearest_team_distance": None,
        "actor_sample_frames": sample_counts.get(actor_track_id, 0),
        "comparison_tracks": len(peer_ids),
        "team_cluster_sizes": [],
        "team_track_ids": [],
        "discarded_cluster_track_ids": [],
        "excluded_actor_fragment_tracks": excluded_fragments,
    }
    if actor_descriptor is None:
        unavailable["reason"] = "actor_jersey_unavailable"
        return unavailable
    if len(peer_ids) < config.jersey_minimum_comparison_tracks:
        unavailable["reason"] = "insufficient_comparison_tracks"
        return unavailable
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "Install scikit-learn before computing team jersey clusters."
        ) from exc

    embeddings = {
        track_id: _histogram_embedding(descriptors[track_id])
        for track_id in peer_ids
    }
    matrix = np.stack([embeddings[track_id] for track_id in peer_ids])
    weights = np.asarray(
        [sample_counts[track_id] for track_id in peer_ids], dtype=float
    )
    unique_colours = len(np.unique(matrix, axis=0))
    if unique_colours < 2:
        unavailable["reason"] = "comparison_colours_not_separable"
        return unavailable
    maximum_supported_clusters = max(
        2, len(peer_ids) // config.jersey_minimum_cluster_tracks
    )
    cluster_count = min(
        config.jersey_cluster_candidates,
        unique_colours,
        maximum_supported_clusters,
    )
    model = KMeans(n_clusters=cluster_count, n_init=10, random_state=0)
    labels = model.fit_predict(matrix, sample_weight=weights)
    cluster_track_ids = [
        [
            track_id
            for track_id, label in zip(peer_ids, labels)
            if int(label) == cluster_id
        ]
        for cluster_id in range(cluster_count)
    ]
    cluster_track_ids.sort(
        key=lambda members: (
            -sum(sample_counts[track_id] for track_id in members),
            tuple(members),
        )
    )
    valid_clusters = [
        members
        for members in cluster_track_ids
        if len(members) >= config.jersey_minimum_cluster_tracks
    ]
    unavailable["team_cluster_sizes"] = [
        len(members) for members in cluster_track_ids
    ]
    unavailable["team_track_ids"] = cluster_track_ids
    if len(valid_clusters) < 2:
        unavailable["reason"] = "team_clusters_too_small"
        return unavailable
    centroids: list[np.ndarray] = []
    for members in valid_clusters[:2]:
        member_matrix = np.stack([embeddings[track_id] for track_id in members])
        member_weights = np.asarray(
            [sample_counts[track_id] for track_id in members], dtype=float
        )
        centroid = np.average(
            member_matrix, axis=0, weights=member_weights
        ).astype(np.float32)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    actor_embedding = _histogram_embedding(actor_descriptor)
    distances = [
        _hellinger_distance(actor_embedding, centroid)
        for centroid in centroids
    ]
    nearest_distance = min(distances)
    team_match = math.exp(
        -((nearest_distance / config.jersey_team_distance_scale) ** 2)
    )
    return {
        "available": True,
        "reason": None,
        "team_match_score": float(np.clip(team_match, 0.0, 1.0)),
        "outlier_score": float(np.clip(1.0 - team_match, 0.0, 1.0)),
        "nearest_team_distance": nearest_distance,
        "actor_sample_frames": sample_counts.get(actor_track_id, 0),
        "comparison_tracks": len(peer_ids),
        "team_cluster_sizes": [
            len(members) for members in valid_clusters[:2]
        ],
        "team_track_ids": valid_clusters[:2],
        "discarded_cluster_track_ids": [
            members
            for members in cluster_track_ids
            if members not in valid_clusters[:2]
        ],
        "excluded_actor_fragment_tracks": excluded_fragments,
    }


def _pose_confidence(landmark: Any) -> float:
    return float(
        min(
            float(getattr(landmark, "visibility", 0.0) or 0.0),
            float(getattr(landmark, "presence", 0.0) or 0.0),
        )
    )


def hand_crop_from_landmarks(
    frame: np.ndarray,
    frame_index: int,
    side: str,
    actor_box: Sequence[float],
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    shoulder_confidence: float,
    elbow_confidence: float,
    wrist_confidence: float,
    config: JerseyGloveConfig,
) -> tuple[HandCrop | None, str | None]:
    """Validate limb geometry and crop slightly beyond the wrist."""

    if side not in ("left", "right"):
        raise ValueError("Hand side must be 'left' or 'right'")
    if wrist_confidence < config.wrist_confidence:
        return None, "low_wrist_confidence"
    if elbow_confidence < config.elbow_confidence:
        return None, "low_elbow_confidence"
    if shoulder_confidence < config.shoulder_confidence:
        return None, "low_shoulder_confidence"
    x1, y1, x2, y2 = (float(value) for value in actor_box)
    actor_height = max(0.0, y2 - y1)
    if actor_height < 1:
        return None, "invalid_actor_box"
    forearm_ratio = float(np.linalg.norm(wrist - elbow) / actor_height)
    upper_arm_ratio = float(np.linalg.norm(elbow - shoulder) / actor_height)
    if not (
        config.minimum_forearm_ratio
        <= forearm_ratio
        <= config.maximum_forearm_ratio
    ):
        return None, "implausible_forearm"
    if not (
        config.minimum_upper_arm_ratio
        <= upper_arm_ratio
        <= config.maximum_upper_arm_ratio
    ):
        return None, "implausible_upper_arm"

    direction = wrist - elbow
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return None, "degenerate_forearm"
    center = wrist + 0.15 * direction
    native_side = int(
        np.clip(
            round(config.hand_crop_side_ratio * actor_height),
            config.hand_minimum_native_side,
            config.hand_maximum_native_side,
        )
    )
    half = native_side / 2.0
    raw_left = int(math.floor(center[0] - half))
    raw_top = int(math.floor(center[1] - half))
    raw_right = raw_left + native_side
    raw_bottom = raw_top + native_side
    frame_height, frame_width = frame.shape[:2]
    left = max(0, raw_left)
    top = max(0, raw_top)
    right = min(frame_width, raw_right)
    bottom = min(frame_height, raw_bottom)
    clipped_area = max(0, right - left) * max(0, bottom - top)
    intended_area = max(1, native_side * native_side)
    clipped_fraction = 1.0 - clipped_area / intended_area
    if clipped_fraction > config.hand_maximum_clipped_fraction:
        return None, "heavily_clipped"
    crop = frame[top:bottom, left:right]
    if crop.size == 0 or min(crop.shape[:2]) < config.hand_minimum_native_side:
        return None, "undersized_crop"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_variance < config.hand_minimum_blur_variance:
        return None, "blurry_crop"
    blur_quality = min(
        1.0,
        math.sqrt(
            blur_variance
            / max(config.hand_minimum_blur_variance * 4.0, 1e-6)
        ),
    )
    quality = (
        wrist_confidence
        * blur_quality
        * max(0.0, 1.0 - clipped_fraction)
    )
    return (
        HandCrop(
            frame_index=frame_index,
            side=side,
            crop=np.ascontiguousarray(crop),
            bbox=(left, top, right, bottom),
            pose_confidence=wrist_confidence,
            blur_variance=blur_variance,
            native_side=native_side,
            clipped_fraction=float(clipped_fraction),
            quality=float(np.clip(quality, 0.0, 1.0)),
        ),
        None,
    )


class MediaPipeActorHandExtractor:
    """Run Pose Landmarker only on the actor track and return quality hand crops."""

    _indices = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
    }

    def __init__(self, config: JerseyGloveConfig):
        if not config.pose_model.is_file():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {config.pose_model}"
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "Install MediaPipe before extracting actor hand crops."
            ) from exc
        self.config = config
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(config.pose_model)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.pose_detection_confidence,
            min_pose_presence_confidence=config.pose_presence_confidence,
            min_tracking_confidence=0.3,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self.pose.close()

    def __enter__(self) -> "MediaPipeActorHandExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _actor_landmarks(
        self, frame: np.ndarray, box: Sequence[float]
    ) -> dict[str, tuple[np.ndarray, float]] | None:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in box)
        box_width, box_height = x2 - x1, y2 - y1
        margin = self.config.pose_crop_margin
        left = max(0, int(math.floor(x1 - margin * box_width)))
        top = max(0, int(math.floor(y1 - margin * box_height)))
        right = min(
            frame_width, int(math.ceil(x2 + margin * box_width))
        )
        bottom = min(
            frame_height, int(math.ceil(y2 + margin * box_height))
        )
        crop = frame[top:bottom, left:right]
        if crop.shape[0] < 24 or crop.shape[1] < 16:
            return None
        crop_height, crop_width = crop.shape[:2]
        square_side = max(crop_height, crop_width)
        pad_x = (square_side - crop_width) // 2
        pad_y = (square_side - crop_height) // 2
        square = np.zeros((square_side, square_side, 3), dtype=np.uint8)
        square[
            pad_y : pad_y + crop_height,
            pad_x : pad_x + crop_width,
        ] = crop
        resized = cv2.resize(
            square,
            (self.config.pose_input_size, self.config.pose_input_size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )
        result = self.pose.detect(image)
        if not result.pose_landmarks:
            return None
        raw_landmarks = result.pose_landmarks[0]
        landmarks: dict[str, tuple[np.ndarray, float]] = {}
        for name, index in self._indices.items():
            landmark = raw_landmarks[index]
            point = np.asarray(
                [
                    left + landmark.x * square_side - pad_x,
                    top + landmark.y * square_side - pad_y,
                ],
                dtype=float,
            )
            landmarks[name] = (point, _pose_confidence(landmark))
        return landmarks

    def extract(
        self,
        frame_paths: Sequence[Path],
        observations: Sequence[TrackObservation],
    ) -> tuple[list[HandCrop], dict[str, Any]]:
        accepted: list[HandCrop] = []
        pose_frames = 0
        rejection_counts: dict[str, int] = {}
        for observation in observations:
            if not 0 <= observation.frame_index < len(frame_paths):
                rejection_counts["invalid_frame_index"] = (
                    rejection_counts.get("invalid_frame_index", 0) + 1
                )
                continue
            frame = cv2.imread(str(frame_paths[observation.frame_index]))
            if frame is None:
                raise RuntimeError(
                    f"Could not read hand frame: "
                    f"{frame_paths[observation.frame_index]}"
                )
            landmarks = self._actor_landmarks(frame, observation.box)
            if landmarks is None:
                rejection_counts["pose_unavailable"] = (
                    rejection_counts.get("pose_unavailable", 0) + 1
                )
                continue
            pose_frames += 1
            for side in ("left", "right"):
                shoulder, shoulder_confidence = landmarks[
                    f"{side}_shoulder"
                ]
                elbow, elbow_confidence = landmarks[f"{side}_elbow"]
                wrist, wrist_confidence = landmarks[f"{side}_wrist"]
                hand, rejection = hand_crop_from_landmarks(
                    frame,
                    observation.frame_index,
                    side,
                    observation.box,
                    shoulder,
                    elbow,
                    wrist,
                    shoulder_confidence,
                    elbow_confidence,
                    wrist_confidence,
                    self.config,
                )
                if hand is not None:
                    accepted.append(hand)
                elif rejection is not None:
                    rejection_counts[rejection] = (
                        rejection_counts.get(rejection, 0) + 1
                    )
        accepted.sort(
            key=lambda item: (
                -item.quality,
                item.frame_index,
                item.side,
            )
        )
        best_per_frame: list[HandCrop] = []
        seen_frames: set[int] = set()
        for item in accepted:
            if item.frame_index in seen_frames:
                continue
            best_per_frame.append(item)
            seen_frames.add(item.frame_index)
            if (
                len(best_per_frame)
                >= self.config.hand_minimum_distinct_frames
            ):
                break
        selected = list(best_per_frame)
        selected_ids = {id(item) for item in selected}
        selected.extend(
            item
            for item in accepted
            if id(item) not in selected_ids
        )
        selected = selected[: self.config.hand_maximum_samples]
        distinct_frames = len({item.frame_index for item in selected})
        sufficient = (
            distinct_frames >= self.config.hand_minimum_distinct_frames
        )
        return selected, {
            "tracked_frames": len(observations),
            "pose_frames": pose_frames,
            "accepted_crops_before_limit": len(accepted),
            "selected_crops": len(selected),
            "distinct_frames": distinct_frames,
            "minimum_distinct_frames": (
                self.config.hand_minimum_distinct_frames
            ),
            "sufficient_for_glove_inference": sufficient,
            "rejection_counts": rejection_counts,
        }


def _frame_shapes(
    frame_paths: Sequence[Path],
) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        shapes.append((int(frame.shape[0]), int(frame.shape[1])))
    return shapes


def _actor_prtreid_evidence(
    observations: Sequence[TrackObservation],
    source_frame_count: int,
    config: JerseyGloveConfig,
    worker: RoleWorker | None,
) -> dict[str, Any]:
    if not config.use_prtreid_evidence:
        return {
            "available": False,
            "reason": "disabled",
            "prediction_frames": 0,
            "coverage": 0.0,
            "scores": None,
            "vote_counts": None,
        }
    crops = [
        {
            "frame_path": str(observation.frame_path),
            "bbox": list(observation.box),
        }
        for observation in observations
    ]
    if not crops:
        return {
            "available": False,
            "reason": "no_actor_crops",
            "prediction_frames": 0,
            "coverage": 0.0,
            "scores": None,
            "vote_counts": None,
        }
    created_worker: PRTReIDWorkerClient | None = None
    active_worker = worker
    if active_worker is None:
        created_worker = PRTReIDWorkerClient(config.base_config)
        active_worker = created_worker
    try:
        predictions = active_worker.predict(crops)
    finally:
        if created_worker is not None:
            created_worker.close()
    if len(predictions) != len(crops):
        raise RuntimeError(
            "PRTReID actor prediction count does not match actor crops"
        )
    votes = {name: 0 for name in ROLE_NAMES}
    probability_rows: list[list[float]] = []
    for prediction in predictions:
        role = str(prediction.get("predicted_role", ""))
        probabilities = prediction.get("role_probabilities")
        if role in votes:
            votes[role] += 1
        if not isinstance(probabilities, dict):
            continue
        try:
            row = [float(probabilities[name]) for name in ROLE_NAMES]
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in row):
            probability_rows.append(row)
    if not probability_rows:
        return {
            "available": False,
            "reason": "invalid_worker_predictions",
            "prediction_frames": 0,
            "coverage": 0.0,
            "scores": None,
            "vote_counts": votes,
        }
    prediction_count = len(probability_rows)
    coverage = (
        prediction_count / source_frame_count
        if source_frame_count > 0
        else 0.0
    )
    if (
        prediction_count < config.base_config.minimum_predictions
        or coverage < config.base_config.minimum_role_coverage
    ):
        return {
            "available": False,
            "reason": "insufficient_prtreid_evidence",
            "prediction_frames": prediction_count,
            "coverage": coverage,
            "minimum_predictions": (
                config.base_config.minimum_predictions
            ),
            "minimum_coverage": (
                config.base_config.minimum_role_coverage
            ),
            "scores": None,
            "vote_counts": votes,
        }
    means = np.mean(np.asarray(probability_rows, dtype=float), axis=0)
    total_votes = sum(votes.values())
    vote_scores = {
        name: votes[name] / total_votes if total_votes else 0.0
        for name in ROLE_NAMES
    }
    scores = {
        name: float(
            0.5 * means[index] + 0.5 * vote_scores[name]
        )
        for index, name in enumerate(ROLE_NAMES)
    }
    return {
        "available": True,
        "reason": None,
        "prediction_frames": prediction_count,
        "coverage": coverage,
        "scores": scores,
        "vote_counts": votes,
    }


def compute_glove_evidence(
    hand_crops: Sequence[HandCrop],
    hand_summary: dict[str, Any],
    config: JerseyGloveConfig,
    glove_model: GloveProbabilityModel | None = None,
) -> dict[str, Any]:
    crop_metadata = [item.as_dict() for item in hand_crops]
    if not config.glove_enabled:
        return {
            "available": False,
            "reason": "classifier_disabled",
            "glove_probability": None,
            "maximum_probability": None,
            "valid_crops": len(hand_crops),
            "distinct_frames": int(
                hand_summary.get("distinct_frames", 0)
            ),
            "hand_summary": hand_summary,
            "crops": crop_metadata,
        }
    if not bool(hand_summary.get("sufficient_for_glove_inference", False)):
        return {
            "available": False,
            "reason": "insufficient_quality_hand_crops",
            "glove_probability": None,
            "maximum_probability": None,
            "valid_crops": len(hand_crops),
            "distinct_frames": int(
                hand_summary.get("distinct_frames", 0)
            ),
            "hand_summary": hand_summary,
            "crops": crop_metadata,
        }
    active_model = (
        glove_model
        if glove_model is not None
        else GloveClassifier(
            config.glove_checkpoint,
            device=config.glove_device,
            batch_size=config.glove_batch_size,
        )
    )
    probabilities = np.asarray(
        active_model.predict_glove_probability(
            [item.crop for item in hand_crops]
        ),
        dtype=float,
    )
    if probabilities.shape != (len(hand_crops),):
        raise RuntimeError(
            "Glove classifier probability count does not match hand crops"
        )
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise RuntimeError("Glove classifier produced invalid probabilities")
    weights = np.asarray(
        [max(item.quality, 1e-6) for item in hand_crops], dtype=float
    )
    for hand_crop, probability in zip(hand_crops, probabilities):
        hand_crop.glove_probability = float(probability)
    aggregate = float(np.average(probabilities, weights=weights))
    return {
        "available": True,
        "reason": None,
        "glove_probability": aggregate,
        "maximum_probability": float(np.max(probabilities)),
        "valid_crops": len(hand_crops),
        "distinct_frames": int(hand_summary.get("distinct_frames", 0)),
        "hand_summary": hand_summary,
        "crops": [item.as_dict() for item in hand_crops],
    }


def decide_goalkeeper(
    association: ActorAssociation | dict[str, Any],
    jersey_evidence: dict[str, Any],
    prtreid_evidence: dict[str, Any],
    glove_evidence: dict[str, Any],
    config: JerseyGloveConfig,
) -> dict[str, Any]:
    if isinstance(association, ActorAssociation):
        association_confident = association.confident
    else:
        association_confident = bool(association.get("confident", False))
    if not association_confident:
        return {
            "status": "unknown",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": None,
            "reason": "actor_association_uncertain",
        }

    team_match = (
        float(jersey_evidence["team_match_score"])
        if jersey_evidence.get("available")
        and jersey_evidence.get("team_match_score") is not None
        else None
    )
    jersey_outlier = (
        float(jersey_evidence["outlier_score"])
        if jersey_evidence.get("available")
        and jersey_evidence.get("outlier_score") is not None
        else None
    )
    prt_scores = (
        prtreid_evidence.get("scores")
        if prtreid_evidence.get("available")
        else None
    )
    player_score = (
        float(prt_scores["player"])
        if isinstance(prt_scores, dict) and "player" in prt_scores
        else None
    )
    glove_probability = (
        float(glove_evidence["glove_probability"])
        if glove_evidence.get("available")
        and glove_evidence.get("glove_probability") is not None
        else None
    )

    evidence_values: list[tuple[float, float]] = []
    if jersey_outlier is not None:
        evidence_values.append((0.45, jersey_outlier))
    if glove_probability is not None:
        evidence_values.append((0.40, glove_probability))
    if player_score is not None:
        evidence_values.append((0.15, 1.0 - player_score))
    evidence_score = (
        sum(weight * value for weight, value in evidence_values)
        / sum(weight for weight, _ in evidence_values)
        if evidence_values
        else None
    )

    strong_team_match = (
        team_match is not None
        and team_match >= config.not_goalkeeper_team_match
    )
    strong_player_score = (
        player_score is not None
        and player_score >= config.not_goalkeeper_player_score
    )
    player_model_supports_field = (
        not config.use_prtreid_evidence or strong_player_score
    )
    if strong_team_match and player_model_supports_field:
        if (
            glove_probability is not None
            and glove_probability >= config.goalkeeper_glove_probability
        ):
            reason = "field_team_evidence_overrides_glove_only_signal"
        elif config.use_prtreid_evidence:
            reason = "field_team_jersey_and_player_model_agree"
        else:
            reason = "field_team_jersey_match"
        return {
            "status": "not_goalkeeper",
            "is_goalkeeper": False,
            "goalkeeper_evidence_score": evidence_score,
            "reason": reason,
        }

    jersey_supports_goalkeeper = (
        jersey_outlier is not None
        and jersey_outlier >= config.goalkeeper_jersey_outlier
    )
    glove_supports_goalkeeper = (
        glove_probability is not None
        and glove_probability >= config.goalkeeper_glove_probability
    )
    player_model_allows_goalkeeper = (
        player_score is None
        and not config.use_prtreid_evidence
    ) or (
        player_score is not None
        and player_score <= config.goalkeeper_maximum_player_score
    )
    if (
        jersey_supports_goalkeeper
        and glove_supports_goalkeeper
        and player_model_allows_goalkeeper
    ):
        return {
            "status": "goalkeeper",
            "is_goalkeeper": True,
            "goalkeeper_evidence_score": evidence_score,
            "reason": "jersey_glove_and_non_player_evidence_agree",
        }
    return {
        "status": "unknown",
        "is_goalkeeper": None,
        "goalkeeper_evidence_score": evidence_score,
        "reason": "insufficient_or_conflicting_goalkeeper_evidence",
    }


def classify_goalkeeper_after_handball(
    frame_paths: Sequence[Path],
    selected_features: np.ndarray,
    selected_indices: Sequence[int],
    metadata: dict[str, Any],
    handball_probability: float,
    handball_threshold: float,
    config: JerseyGloveConfig,
    tracker: PersonTracker | None = None,
    role_worker: RoleWorker | None = None,
    hand_extractor: HandExtractor | None = None,
    glove_model: GloveProbabilityModel | None = None,
    force_evaluation: bool = False,
) -> dict[str, Any]:
    """Classify the actor, optionally regardless of the raw GRU decision."""

    if not frame_paths:
        raise ValueError("Cannot classify a goalkeeper for an empty clip")
    if not math.isfinite(float(handball_probability)):
        raise ValueError("handball_probability must be finite")
    if not 0 <= handball_threshold <= 1:
        raise ValueError("handball_threshold must be between 0 and 1")
    if handball_probability < handball_threshold and not force_evaluation:
        return {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": jersey_glove_config_fingerprint(config),
            "evaluated": False,
            "status": "not_evaluated",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": None,
            "reason": "handball_below_threshold",
            "handball_probability_observed": float(handball_probability),
            "handball_threshold_observed": float(handball_threshold),
        }

    tracks = track_all_people(frame_paths, config.base_config, tracker)
    shapes = _frame_shapes(frame_paths)
    association = associate_handball_actor(
        tracks,
        shapes,
        selected_features,
        selected_indices,
        metadata,
        config.base_config,
    )
    jersey_evidence = compute_jersey_team_evidence(
        frame_paths, tracks, association.track_id, config
    )
    actor_observations = (
        tracks.get(association.track_id, [])
        if association.track_id is not None
        else []
    )

    if association.confident and actor_observations:
        prtreid_evidence = _actor_prtreid_evidence(
            actor_observations, len(frame_paths), config, role_worker
        )
        if config.glove_enabled or hand_extractor is not None:
            created_hand_extractor: MediaPipeActorHandExtractor | None = None
            active_hand_extractor = hand_extractor
            if active_hand_extractor is None:
                created_hand_extractor = MediaPipeActorHandExtractor(config)
                active_hand_extractor = created_hand_extractor
            try:
                hand_crops, hand_summary = active_hand_extractor.extract(
                    frame_paths, actor_observations
                )
            finally:
                if created_hand_extractor is not None:
                    created_hand_extractor.close()
            glove_evidence = compute_glove_evidence(
                hand_crops,
                hand_summary,
                config,
                glove_model,
            )
        else:
            glove_evidence = compute_glove_evidence(
                [],
                {
                    "tracked_frames": len(actor_observations),
                    "pose_frames": 0,
                    "accepted_crops_before_limit": 0,
                    "selected_crops": 0,
                    "distinct_frames": 0,
                    "minimum_distinct_frames": (
                        config.hand_minimum_distinct_frames
                    ),
                    "sufficient_for_glove_inference": False,
                    "rejection_counts": {},
                    "skipped": True,
                },
                config,
                glove_model,
            )
    else:
        prtreid_evidence = {
            "available": False,
            "reason": "actor_association_uncertain",
            "prediction_frames": 0,
            "coverage": 0.0,
            "scores": None,
            "vote_counts": None,
        }
        glove_evidence = {
            "available": False,
            "reason": "actor_association_uncertain",
            "glove_probability": None,
            "maximum_probability": None,
            "valid_crops": 0,
            "distinct_frames": 0,
            "hand_summary": None,
            "crops": [],
        }
    decision = decide_goalkeeper(
        association,
        jersey_evidence,
        prtreid_evidence,
        glove_evidence,
        config,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": jersey_glove_config_fingerprint(config),
        "evaluated": True,
        "status": decision["status"],
        "is_goalkeeper": decision["is_goalkeeper"],
        "goalkeeper_evidence_score": decision[
            "goalkeeper_evidence_score"
        ],
        "reason": decision["reason"],
        "handball_probability_observed": float(handball_probability),
        "handball_threshold_observed": float(handball_threshold),
        "frame_count": len(frame_paths),
        "tracked_people": len(tracks),
        "actor_track_id": association.track_id,
        "association": association.as_dict(),
        "actor_track_frames": len(actor_observations),
        "actor_observations": [
            observation.as_dict() for observation in actor_observations
        ],
        "jersey": jersey_evidence,
        "prtreid": prtreid_evidence,
        "glove": glove_evidence,
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


def jersey_glove_config_fingerprint(config: JerseyGloveConfig) -> str:
    glove_source = Path(__file__).with_name("glove_classifier.py")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "glove_source_sha256": (
            hashlib.sha256(glove_source.read_bytes()).hexdigest()
            if glove_source.is_file()
            else None
        ),
        "base_prtreid_fingerprint": prtreid_config_fingerprint(
            config.base_config
        ),
        "use_prtreid_evidence": config.use_prtreid_evidence,
        "pose_model": _file_identity(config.pose_model),
        "pose": {
            "input_size": config.pose_input_size,
            "detection_confidence": config.pose_detection_confidence,
            "presence_confidence": config.pose_presence_confidence,
            "crop_margin": config.pose_crop_margin,
            "wrist_confidence": config.wrist_confidence,
            "elbow_confidence": config.elbow_confidence,
            "shoulder_confidence": config.shoulder_confidence,
            "forearm_ratio": [
                config.minimum_forearm_ratio,
                config.maximum_forearm_ratio,
            ],
            "upper_arm_ratio": [
                config.minimum_upper_arm_ratio,
                config.maximum_upper_arm_ratio,
            ],
        },
        "hands": {
            "crop_side_ratio": config.hand_crop_side_ratio,
            "minimum_native_side": config.hand_minimum_native_side,
            "maximum_native_side": config.hand_maximum_native_side,
            "maximum_clipped_fraction": (
                config.hand_maximum_clipped_fraction
            ),
            "minimum_blur_variance": config.hand_minimum_blur_variance,
            "minimum_distinct_frames": (
                config.hand_minimum_distinct_frames
            ),
            "maximum_samples": config.hand_maximum_samples,
        },
        "jersey": {
            "crop": [
                config.jersey_crop_top,
                config.jersey_crop_bottom,
                config.jersey_crop_left,
                config.jersey_crop_right,
            ],
            "histogram_bins": config.jersey_histogram_bins,
            "maximum_samples_per_track": (
                config.jersey_maximum_samples_per_track
            ),
            "minimum_pixels": config.jersey_minimum_pixels,
            "minimum_track_frames": config.jersey_minimum_track_frames,
            "minimum_comparison_tracks": (
                config.jersey_minimum_comparison_tracks
            ),
            "minimum_cluster_tracks": (
                config.jersey_minimum_cluster_tracks
            ),
            "cluster_candidates": config.jersey_cluster_candidates,
            "team_distance_scale": config.jersey_team_distance_scale,
        },
        "glove": {
            "enabled": config.glove_enabled,
            "checkpoint": _file_identity(config.glove_checkpoint),
            "device": config.glove_device,
            "batch_size": config.glove_batch_size,
        },
        "decision": {
            "not_goalkeeper_team_match": (
                config.not_goalkeeper_team_match
            ),
            "not_goalkeeper_player_score": (
                config.not_goalkeeper_player_score
            ),
            "goalkeeper_jersey_outlier": (
                config.goalkeeper_jersey_outlier
            ),
            "goalkeeper_glove_probability": (
                config.goalkeeper_glove_probability
            ),
            "goalkeeper_maximum_player_score": (
                config.goalkeeper_maximum_player_score
            ),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def jersey_glove_result_is_current(
    destination: Path,
    config: JerseyGloveConfig,
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
        and result.get("config_fingerprint")
        == jersey_glove_config_fingerprint(config)
        and result.get("source_fingerprint") == source_fingerprint
    )


def save_jersey_glove_result(
    result: dict[str, Any], destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".temporary.json")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(destination)
