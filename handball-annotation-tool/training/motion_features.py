"""Add ball/arm trajectory and camera-compensated optical-flow features.

The extractor consumes the existing 12x56 cached feature artifacts.  It does
not rerun YOLO, ByteTrack, or MediaPipe.  Optical flow is measured on the raw
frames immediately before and after each of the same 12 selected moments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml

from .config import project_path
from .data import feature_metadata
from .features import FEATURE_NAMES, feature_path
from .logging_utils import configure_logging
from .manifest import sorted_frames

MOTION_FEATURE_NAMES = [
    # Smoothed ball kinematics in normalized screen-height units per second.
    "motion_ball_velocity_x",
    "motion_ball_velocity_y",
    "motion_ball_speed",
    "motion_ball_acceleration_x",
    "motion_ball_acceleration_y",
    "motion_ball_acceleration",
    "motion_ball_jerk",
    "motion_ball_speed_change",
    "motion_ball_heading_change",
    "motion_ball_curvature",
    "motion_ball_lower_frame_proximity",
    "motion_ball_vertical_reversal",
    "motion_bounce_score",
    # Closest arm geometry and pre/post interaction.
    "motion_closest_arm_x",
    "motion_closest_arm_y",
    "motion_closest_arm_distance",
    "motion_closest_arm_side",
    "motion_closest_arm_valid",
    "motion_arm_velocity_x",
    "motion_arm_velocity_y",
    "motion_arm_speed",
    "motion_relative_velocity_x",
    "motion_relative_velocity_y",
    "motion_relative_speed",
    "motion_ball_arm_velocity_cosine",
    "motion_arm_distance_change",
    "motion_ball_approach_speed",
    "motion_ball_separation_speed",
    "motion_time_to_closest_arm",
    "motion_ball_arm_overlap",
    "motion_local_contact_duration",
    "motion_trajectory_impact_score",
    # Wrist motion is retained as a vector instead of only a magnitude.
    "motion_left_wrist_velocity_x",
    "motion_left_wrist_velocity_y",
    "motion_left_wrist_speed",
    "motion_right_wrist_velocity_x",
    "motion_right_wrist_velocity_y",
    "motion_right_wrist_speed",
    # Global camera flow and camera-compensated ball flow.
    "flow_camera_x",
    "flow_camera_y",
    "flow_camera_magnitude",
    "flow_ball_pre_x",
    "flow_ball_pre_y",
    "flow_ball_pre_magnitude",
    "flow_ball_post_x",
    "flow_ball_post_y",
    "flow_ball_post_magnitude",
    "flow_ball_direction_change",
    "flow_ball_acceleration",
    "flow_ball_variance",
    "flow_ball_valid",
    # Both arms are measured; the closest arm is also exposed explicitly.
    "flow_left_arm_post_x",
    "flow_left_arm_post_y",
    "flow_left_arm_post_magnitude",
    "flow_left_arm_variance",
    "flow_left_arm_valid",
    "flow_right_arm_post_x",
    "flow_right_arm_post_y",
    "flow_right_arm_post_magnitude",
    "flow_right_arm_variance",
    "flow_right_arm_valid",
    "flow_closest_arm_pre_x",
    "flow_closest_arm_pre_y",
    "flow_closest_arm_pre_magnitude",
    "flow_closest_arm_post_x",
    "flow_closest_arm_post_y",
    "flow_closest_arm_post_magnitude",
    "flow_closest_arm_variance",
    "flow_closest_arm_valid",
    # Ball-versus-arm flow discontinuity and coupling.
    "flow_relative_ball_arm_x",
    "flow_relative_ball_arm_y",
    "flow_relative_ball_arm_magnitude",
    "flow_ball_arm_alignment_cosine",
    "flow_local_discontinuity",
    "flow_camera_motion_ratio",
    "flow_optical_contact_score",
    "flow_optical_valid",
    # Clip-level context repeated at every temporal step for the GRU.
    "motion_sequence_min_arm_distance",
    "motion_frames_to_closest_arm",
    "motion_sequence_max_ball_acceleration",
    "motion_sequence_max_heading_change",
    "motion_sequence_max_bounce_score",
    "motion_sequence_contact_fraction",
    "motion_trajectory_valid",
]
MOTION_GRU_FEATURE_NAMES = FEATURE_NAMES + MOTION_FEATURE_NAMES


@dataclass(frozen=True)
class MotionFeatureConfig:
    manifest: Path
    base_features_dir: Path
    motion_features_dir: Path
    logs_dir: Path
    flow_width: int
    flow_preset: str
    arm_mask_fraction: float
    contact_margin: float


def load_motion_feature_config(path: str | Path) -> MotionFeatureConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        extraction = raw["motion_extraction"]
        paths = raw["paths"]
        config = MotionFeatureConfig(
            manifest=project_path(paths["manifest"]),
            base_features_dir=project_path(paths["base_features"]),
            motion_features_dir=project_path(paths["motion_features"]),
            logs_dir=project_path(paths["motion_logs"]),
            flow_width=int(extraction.get("flow_width", 320)),
            flow_preset=str(extraction.get("flow_preset", "fast")),
            arm_mask_fraction=float(
                extraction.get("arm_mask_fraction", 0.08)
            ),
            contact_margin=float(extraction.get("contact_margin", 0.05)),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid motion feature configuration: {exc}") from exc
    if config.flow_width < 64:
        raise ValueError("motion_extraction.flow_width must be at least 64")
    if config.flow_preset not in {"ultrafast", "fast", "medium"}:
        raise ValueError(
            "motion_extraction.flow_preset must be ultrafast, fast, or medium"
        )
    if not 0 < config.arm_mask_fraction <= 0.5:
        raise ValueError(
            "motion_extraction.arm_mask_fraction must be in (0, 0.5]"
        )
    if not 0 <= config.contact_margin <= 1:
        raise ValueError(
            "motion_extraction.contact_margin must be between 0 and 1"
        )
    return config


def motion_feature_path(root: Path, row: pd.Series) -> Path:
    return feature_path(root, row)


def motion_schema_fingerprint(config: MotionFeatureConfig) -> str:
    payload = {
        "schema_version": 2,
        "feature_names": MOTION_GRU_FEATURE_NAMES,
        "flow_width": config.flow_width,
        "flow_preset": config.flow_preset,
        "arm_mask_fraction": config.arm_mask_fraction,
        "contact_margin": config.contact_margin,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float32, copy=True)
    padded = np.pad(values.astype(float), (1, 1), mode="edge")
    return (
        0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
    ).astype(np.float32)


def _interpolate(
    values: np.ndarray,
    valid: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, bool]:
    finite = valid.astype(bool) & np.isfinite(values)
    if finite.sum() == 0:
        return np.zeros_like(values, dtype=np.float32), False
    if finite.sum() == 1:
        value = float(values[finite][0])
        return np.full_like(values, value, dtype=np.float32), False
    interpolated = np.interp(times, times[finite], values[finite])
    return _smooth(interpolated), True


def _gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values.astype(float), times.astype(float)).astype(
        np.float32
    )


def _vector_cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-8:
        return 0.0
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _vector_angle(first: np.ndarray, second: np.ndarray) -> float:
    if np.linalg.norm(first) <= 1e-8 or np.linalg.norm(second) <= 1e-8:
        return 0.0
    return float(math.acos(_vector_cosine(first, second)) / math.pi)


def _closest_point(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[np.ndarray, float]:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-12:
        closest = start
    else:
        fraction = float(
            np.clip(np.dot(point - start, segment) / denominator, 0, 1)
        )
        closest = start + fraction * segment
    return closest, float(np.linalg.norm(point - closest))


def _landmark(
    row: np.ndarray,
    indices: Mapping[str, int],
    name: str,
    aspect: float,
) -> np.ndarray | None:
    if float(row[indices[f"{name}_valid"]]) <= 0:
        return None
    return np.array(
        [
            float(row[indices[f"{name}_x"]]) * aspect,
            float(row[indices[f"{name}_y"]]),
        ],
        dtype=np.float32,
    )


def closest_arm_geometry(
    row: np.ndarray,
    *,
    aspect: float,
) -> tuple[np.ndarray, float, int, bool]:
    """Return closest arm point, distance in player heights, and side."""

    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    if float(row[indices["ball_valid"]]) <= 0:
        return np.zeros(2, dtype=np.float32), 0.0, 0, False
    player_height = float(row[indices["player_h"]])
    if player_height <= 1e-6:
        return np.zeros(2, dtype=np.float32), 0.0, 0, False
    ball = np.array(
        [
            float(row[indices["ball_x"]]) * aspect,
            float(row[indices["ball_y"]]),
        ],
        dtype=np.float32,
    )
    candidates: list[tuple[float, np.ndarray, int]] = []
    for side_name, side_value in (("left", -1), ("right", 1)):
        shoulder = _landmark(row, indices, f"{side_name}_shoulder", aspect)
        elbow = _landmark(row, indices, f"{side_name}_elbow", aspect)
        wrist = _landmark(row, indices, f"{side_name}_wrist", aspect)
        for start, end in ((shoulder, elbow), (elbow, wrist)):
            if start is None or end is None:
                continue
            closest, distance = _closest_point(ball, start, end)
            candidates.append((distance, closest, side_value))
    if not candidates:
        return np.zeros(2, dtype=np.float32), 0.0, 0, False
    distance, closest, side = min(candidates, key=lambda value: value[0])
    return closest, float(distance / player_height), side, True


def _trajectory_features(
    base: np.ndarray,
    selected: Sequence[int],
    *,
    fps: float,
    aspect: float,
    contact_margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(base)
    output = np.zeros((count, len(MOTION_FEATURE_NAMES)), dtype=np.float32)
    output_indices = {
        name: index for index, name in enumerate(MOTION_FEATURE_NAMES)
    }
    base_indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    times = np.asarray(selected, dtype=np.float32) / max(fps, 1e-6)
    ball_valid = base[:, base_indices["ball_valid"]] > 0
    ball_x, x_usable = _interpolate(
        base[:, base_indices["ball_x"]] * aspect, ball_valid, times
    )
    ball_y, y_usable = _interpolate(
        base[:, base_indices["ball_y"]], ball_valid, times
    )
    ball_velocity_x = _gradient(ball_x, times)
    ball_velocity_y = _gradient(ball_y, times)
    ball_speed = np.hypot(ball_velocity_x, ball_velocity_y)
    ball_acceleration_x = _gradient(ball_velocity_x, times)
    ball_acceleration_y = _gradient(ball_velocity_y, times)
    ball_acceleration = np.hypot(
        ball_acceleration_x, ball_acceleration_y
    )
    ball_jerk = np.hypot(
        _gradient(ball_acceleration_x, times),
        _gradient(ball_acceleration_y, times),
    )
    ball_speed_change = _gradient(ball_speed, times)
    heading_change = np.zeros(count, dtype=np.float32)
    curvature = np.zeros(count, dtype=np.float32)
    vertical_reversal = np.zeros(count, dtype=np.float32)
    for index in range(1, count - 1):
        previous = np.array(
            [ball_velocity_x[index - 1], ball_velocity_y[index - 1]]
        )
        following = np.array(
            [ball_velocity_x[index + 1], ball_velocity_y[index + 1]]
        )
        heading_change[index] = _vector_angle(previous, following)
        velocity = np.array(
            [ball_velocity_x[index], ball_velocity_y[index]]
        )
        acceleration = np.array(
            [ball_acceleration_x[index], ball_acceleration_y[index]]
        )
        speed = max(float(np.linalg.norm(velocity)), 1e-6)
        curvature[index] = abs(
            velocity[0] * acceleration[1]
            - velocity[1] * acceleration[0]
        ) / (speed**3)
        vertical_reversal[index] = math.tanh(
            max(0.0, float(ball_velocity_y[index - 1]))
            * max(0.0, float(-ball_velocity_y[index + 1]))
            * 4.0
        )
    lower_frame = np.clip(
        (
            base[:, base_indices["ball_y"]]
            + base[:, base_indices["ball_h"]] / 2
            - 0.5
        )
        / 0.5,
        0,
        1,
    )

    arm_points = np.zeros((count, 2), dtype=np.float32)
    arm_distances = np.zeros(count, dtype=np.float32)
    arm_sides = np.zeros(count, dtype=np.float32)
    arm_valid = np.zeros(count, dtype=bool)
    for index, row in enumerate(base):
        point, distance, side, valid = closest_arm_geometry(
            row, aspect=aspect
        )
        arm_points[index] = point
        arm_distances[index] = distance
        arm_sides[index] = side
        arm_valid[index] = valid
    arm_x, arm_x_usable = _interpolate(arm_points[:, 0], arm_valid, times)
    arm_y, arm_y_usable = _interpolate(arm_points[:, 1], arm_valid, times)
    arm_velocity_x = _gradient(arm_x, times)
    arm_velocity_y = _gradient(arm_y, times)
    arm_speed = np.hypot(arm_velocity_x, arm_velocity_y)
    distance_values, distance_usable = _interpolate(
        arm_distances, arm_valid, times
    )
    distance_change = _gradient(distance_values, times)
    approach = np.clip(-distance_change, 0, None)
    separation = np.clip(distance_change, 0, None)
    relative_x = ball_velocity_x - arm_velocity_x
    relative_y = ball_velocity_y - arm_velocity_y
    relative_speed = np.hypot(relative_x, relative_y)
    velocity_cosine = np.array(
        [
            _vector_cosine(
                np.array([ball_velocity_x[index], ball_velocity_y[index]]),
                np.array([arm_velocity_x[index], arm_velocity_y[index]]),
            )
            for index in range(count)
        ],
        dtype=np.float32,
    )
    valid_distance_indices = np.flatnonzero(arm_valid & ball_valid)
    if len(valid_distance_indices):
        closest_index = int(
            valid_distance_indices[
                np.argmin(distance_values[valid_distance_indices])
            ]
        )
        sequence_min_distance = float(distance_values[closest_index])
    else:
        closest_index = count // 2
        sequence_min_distance = 0.0
    time_to_closest = times[closest_index] - times
    frames_to_closest = float(selected[closest_index]) - np.asarray(
        selected, dtype=np.float32
    )
    player_height = np.maximum(
        base[:, base_indices["player_h"]], 1e-6
    )
    ball_radius_player_heights = (
        np.maximum(
            base[:, base_indices["ball_w"]] * aspect,
            base[:, base_indices["ball_h"]],
        )
        / 2
        / player_height
    )
    overlap = np.clip(
        1
        - distance_values
        / np.maximum(ball_radius_player_heights + contact_margin, 1e-6),
        0,
        1,
    )
    overlap *= (arm_valid & ball_valid).astype(np.float32)
    local_duration = np.convolve(
        (overlap > 0).astype(np.float32),
        np.ones(3, dtype=np.float32),
        mode="same",
    )
    bounce_score = (
        vertical_reversal
        * lower_frame.astype(np.float32)
        * (1 - overlap)
    )
    trajectory_impact = (
        overlap
        * np.tanh(ball_acceleration)
        * np.maximum(heading_change, np.tanh(np.abs(ball_speed_change)))
    )

    def assign(name: str, values: np.ndarray | float) -> None:
        output[:, output_indices[name]] = values

    assign("motion_ball_velocity_x", ball_velocity_x)
    assign("motion_ball_velocity_y", ball_velocity_y)
    assign("motion_ball_speed", ball_speed)
    assign("motion_ball_acceleration_x", ball_acceleration_x)
    assign("motion_ball_acceleration_y", ball_acceleration_y)
    assign("motion_ball_acceleration", ball_acceleration)
    assign("motion_ball_jerk", ball_jerk)
    assign("motion_ball_speed_change", ball_speed_change)
    assign("motion_ball_heading_change", heading_change)
    assign("motion_ball_curvature", np.clip(curvature, 0, 100))
    assign("motion_ball_lower_frame_proximity", lower_frame)
    assign("motion_ball_vertical_reversal", vertical_reversal)
    assign("motion_bounce_score", bounce_score)
    assign("motion_closest_arm_x", arm_x / max(aspect, 1e-6))
    assign("motion_closest_arm_y", arm_y)
    assign("motion_closest_arm_distance", distance_values)
    assign("motion_closest_arm_side", arm_sides)
    assign("motion_closest_arm_valid", arm_valid.astype(np.float32))
    assign("motion_arm_velocity_x", arm_velocity_x)
    assign("motion_arm_velocity_y", arm_velocity_y)
    assign("motion_arm_speed", arm_speed)
    assign("motion_relative_velocity_x", relative_x)
    assign("motion_relative_velocity_y", relative_y)
    assign("motion_relative_speed", relative_speed)
    assign("motion_ball_arm_velocity_cosine", velocity_cosine)
    assign("motion_arm_distance_change", distance_change)
    assign("motion_ball_approach_speed", approach)
    assign("motion_ball_separation_speed", separation)
    assign("motion_time_to_closest_arm", time_to_closest)
    assign("motion_ball_arm_overlap", overlap)
    assign("motion_local_contact_duration", local_duration)
    assign("motion_trajectory_impact_score", trajectory_impact)
    for side in ("left", "right"):
        valid = base[:, base_indices[f"{side}_wrist_valid"]] > 0
        wrist_x, _ = _interpolate(
            base[:, base_indices[f"{side}_wrist_x"]] * aspect,
            valid,
            times,
        )
        wrist_y, _ = _interpolate(
            base[:, base_indices[f"{side}_wrist_y"]], valid, times
        )
        wrist_vx = _gradient(wrist_x, times)
        wrist_vy = _gradient(wrist_y, times)
        assign(f"motion_{side}_wrist_velocity_x", wrist_vx)
        assign(f"motion_{side}_wrist_velocity_y", wrist_vy)
        assign(f"motion_{side}_wrist_speed", np.hypot(wrist_vx, wrist_vy))
    assign("motion_sequence_min_arm_distance", sequence_min_distance)
    assign("motion_frames_to_closest_arm", frames_to_closest)
    assign(
        "motion_sequence_max_ball_acceleration",
        float(ball_acceleration.max(initial=0)),
    )
    assign(
        "motion_sequence_max_heading_change",
        float(heading_change.max(initial=0)),
    )
    assign(
        "motion_sequence_max_bounce_score",
        float(bounce_score.max(initial=0)),
    )
    assign(
        "motion_sequence_contact_fraction",
        float((overlap > 0).mean()),
    )
    trajectory_usable = bool(
        x_usable
        and y_usable
        and arm_x_usable
        and arm_y_usable
        and distance_usable
    )
    assign("motion_trajectory_valid", float(trajectory_usable))
    return output, arm_sides


def _create_dis(preset: str) -> Any:
    presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    return cv2.DISOpticalFlow_create(presets[preset])


def _read_gray(path: Path, flow_width: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read frame: {path}")
    if image.shape[1] == flow_width:
        return image
    scale = flow_width / image.shape[1]
    height = max(1, round(image.shape[0] * scale))
    return cv2.resize(image, (flow_width, height), interpolation=cv2.INTER_AREA)


def _mask_ball(
    shape: tuple[int, int],
    row: np.ndarray,
    indices: Mapping[str, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if float(row[indices["ball_valid"]]) <= 0:
        return mask
    height, width = shape
    center = (
        round(float(row[indices["ball_x"]]) * width),
        round(float(row[indices["ball_y"]]) * height),
    )
    axes = (
        max(2, round(float(row[indices["ball_w"]]) * width / 2)),
        max(2, round(float(row[indices["ball_h"]]) * height / 2)),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _mask_arm(
    shape: tuple[int, int],
    row: np.ndarray,
    indices: Mapping[str, int],
    side: str,
    thickness_fraction: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    player_height = max(
        1, round(float(row[indices["player_h"]]) * height)
    )
    thickness = max(2, round(player_height * thickness_fraction))
    points: dict[str, tuple[int, int]] = {}
    for landmark in ("shoulder", "elbow", "wrist"):
        name = f"{side}_{landmark}"
        if float(row[indices[f"{name}_valid"]]) <= 0:
            continue
        points[landmark] = (
            round(float(row[indices[f"{name}_x"]]) * width),
            round(float(row[indices[f"{name}_y"]]) * height),
        )
    if "shoulder" in points and "elbow" in points:
        cv2.line(
            mask,
            points["shoulder"],
            points["elbow"],
            255,
            thickness,
        )
    if "elbow" in points and "wrist" in points:
        cv2.line(
            mask,
            points["elbow"],
            points["wrist"],
            255,
            thickness,
        )
    return mask


def _flow_statistics(
    flow: np.ndarray | None,
    mask: np.ndarray,
    *,
    fps: float,
    camera: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    selected = mask > 0
    if flow is None or selected.sum() < 4:
        return np.zeros(2, dtype=np.float32), 0.0, False
    values = flow[selected].astype(np.float32)
    residual = values - camera[None]
    scale = fps / max(flow.shape[0], 1)
    vector = np.median(residual, axis=0) * scale
    variance = float(
        np.mean(np.linalg.norm(residual - np.median(residual, axis=0), axis=1))
        * scale
    )
    return vector.astype(np.float32), variance, True


def _optical_flow_features(
    base: np.ndarray,
    selected: Sequence[int],
    frame_paths: Sequence[Path],
    *,
    fps: float,
    config: MotionFeatureConfig,
    arm_sides: np.ndarray,
    trajectory: np.ndarray,
) -> np.ndarray:
    output = np.zeros(
        (len(base), len(MOTION_FEATURE_NAMES)), dtype=np.float32
    )
    output_indices = {
        name: index for index, name in enumerate(MOTION_FEATURE_NAMES)
    }
    base_indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    gray_cache: dict[int, np.ndarray] = {}
    flow_cache: dict[tuple[int, int], np.ndarray] = {}
    dis = _create_dis(config.flow_preset)

    def gray(index: int) -> np.ndarray:
        if index not in gray_cache:
            gray_cache[index] = _read_gray(
                frame_paths[index], config.flow_width
            )
        return gray_cache[index]

    def flow(first: int, second: int) -> np.ndarray | None:
        if first == second:
            return None
        key = (first, second)
        if key not in flow_cache:
            first_gray, second_gray = gray(first), gray(second)
            if first_gray.shape != second_gray.shape:
                raise ValueError(
                    f"Frame shapes changed within clip: "
                    f"{first_gray.shape} != {second_gray.shape}"
                )
            flow_cache[key] = dis.calc(first_gray, second_gray, None)
        return flow_cache[key]

    def camera_vector(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.zeros(2, dtype=np.float32)
        return np.median(value.reshape(-1, 2), axis=0).astype(np.float32)

    def assign(row_index: int, name: str, value: float) -> None:
        output[row_index, output_indices[name]] = value

    for row_index, frame_index in enumerate(selected):
        previous_index = max(0, int(frame_index) - 1)
        following_index = min(len(frame_paths) - 1, int(frame_index) + 1)
        pre_flow = flow(previous_index, int(frame_index))
        post_flow = flow(int(frame_index), following_index)
        reference = (
            post_flow
            if post_flow is not None
            else pre_flow
        )
        if reference is None:
            continue
        camera_pre = camera_vector(pre_flow)
        camera_post = camera_vector(post_flow)
        camera_average = (camera_pre + camera_post) / 2
        camera_normalized = camera_average * fps / reference.shape[0]
        assign(row_index, "flow_camera_x", float(camera_normalized[0]))
        assign(row_index, "flow_camera_y", float(camera_normalized[1]))
        assign(
            row_index,
            "flow_camera_magnitude",
            float(np.linalg.norm(camera_normalized)),
        )
        row = base[row_index]
        ball_mask = _mask_ball(reference.shape[:2], row, base_indices)
        left_mask = _mask_arm(
            reference.shape[:2],
            row,
            base_indices,
            "left",
            config.arm_mask_fraction,
        )
        right_mask = _mask_arm(
            reference.shape[:2],
            row,
            base_indices,
            "right",
            config.arm_mask_fraction,
        )
        ball_pre, _, ball_pre_valid = _flow_statistics(
            pre_flow, ball_mask, fps=fps, camera=camera_pre
        )
        ball_post, ball_variance, ball_post_valid = _flow_statistics(
            post_flow, ball_mask, fps=fps, camera=camera_post
        )
        left_pre, _, left_pre_valid = _flow_statistics(
            pre_flow, left_mask, fps=fps, camera=camera_pre
        )
        left_post, left_variance, left_post_valid = _flow_statistics(
            post_flow, left_mask, fps=fps, camera=camera_post
        )
        right_pre, _, right_pre_valid = _flow_statistics(
            pre_flow, right_mask, fps=fps, camera=camera_pre
        )
        right_post, right_variance, right_post_valid = _flow_statistics(
            post_flow, right_mask, fps=fps, camera=camera_post
        )
        for prefix, vector in (
            ("flow_ball_pre", ball_pre),
            ("flow_ball_post", ball_post),
            ("flow_left_arm_post", left_post),
            ("flow_right_arm_post", right_post),
        ):
            assign(row_index, f"{prefix}_x", float(vector[0]))
            assign(row_index, f"{prefix}_y", float(vector[1]))
            assign(
                row_index,
                f"{prefix}_magnitude",
                float(np.linalg.norm(vector)),
            )
        assign(row_index, "flow_ball_variance", ball_variance)
        assign(
            row_index,
            "flow_ball_valid",
            float(ball_pre_valid and ball_post_valid),
        )
        assign(row_index, "flow_left_arm_variance", left_variance)
        assign(
            row_index, "flow_left_arm_valid", float(left_post_valid)
        )
        assign(row_index, "flow_right_arm_variance", right_variance)
        assign(
            row_index, "flow_right_arm_valid", float(right_post_valid)
        )
        if arm_sides[row_index] < 0:
            arm_pre, arm_post = left_pre, left_post
            arm_variance = left_variance
            arm_valid = left_pre_valid and left_post_valid
        elif arm_sides[row_index] > 0:
            arm_pre, arm_post = right_pre, right_post
            arm_variance = right_variance
            arm_valid = right_pre_valid and right_post_valid
        else:
            arm_pre = arm_post = np.zeros(2, dtype=np.float32)
            arm_variance = 0.0
            arm_valid = False
        for prefix, vector in (
            ("flow_closest_arm_pre", arm_pre),
            ("flow_closest_arm_post", arm_post),
        ):
            assign(row_index, f"{prefix}_x", float(vector[0]))
            assign(row_index, f"{prefix}_y", float(vector[1]))
            assign(
                row_index,
                f"{prefix}_magnitude",
                float(np.linalg.norm(vector)),
            )
        assign(row_index, "flow_closest_arm_variance", arm_variance)
        assign(row_index, "flow_closest_arm_valid", float(arm_valid))
        relative = ball_post - arm_post
        assign(row_index, "flow_relative_ball_arm_x", float(relative[0]))
        assign(row_index, "flow_relative_ball_arm_y", float(relative[1]))
        assign(
            row_index,
            "flow_relative_ball_arm_magnitude",
            float(np.linalg.norm(relative)),
        )
        alignment = _vector_cosine(ball_post, arm_post)
        assign(row_index, "flow_ball_arm_alignment_cosine", alignment)
        discontinuity = float(np.linalg.norm(ball_post - ball_pre))
        assign(
            row_index,
            "flow_ball_direction_change",
            _vector_angle(ball_pre, ball_post),
        )
        assign(
            row_index,
            "flow_ball_acceleration",
            discontinuity * fps,
        )
        assign(row_index, "flow_local_discontinuity", discontinuity)
        ball_raw_magnitude = float(
            np.linalg.norm(ball_post + camera_normalized)
        )
        assign(
            row_index,
            "flow_camera_motion_ratio",
            float(
                np.linalg.norm(camera_normalized)
                / max(ball_raw_magnitude, 1e-6)
            ),
        )
        overlap = float(
            trajectory[
                row_index,
                output_indices["motion_ball_arm_overlap"],
            ]
        )
        optical_valid = bool(
            ball_pre_valid and ball_post_valid and arm_valid
        )
        contact_score = (
            overlap
            * math.tanh(discontinuity)
            * max(0.0, (alignment + 1) / 2)
        )
        assign(row_index, "flow_optical_contact_score", contact_score)
        assign(row_index, "flow_optical_valid", float(optical_valid))
    return output


def augment_motion_features(
    base: np.ndarray,
    selected: Sequence[int],
    frame_paths: Sequence[Path],
    *,
    fps: float,
    config: MotionFeatureConfig,
) -> np.ndarray:
    if base.ndim != 2 or base.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Base features must have shape [time, {len(FEATURE_NAMES)}]"
        )
    if len(base) != len(selected):
        raise ValueError("Base feature rows and selected indices disagree")
    if not frame_paths:
        raise ValueError("Cannot augment features without source frames")
    if any(index < 0 or index >= len(frame_paths) for index in selected):
        raise ValueError("At least one selected frame index is unavailable")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")
    aspect = first.shape[1] / max(first.shape[0], 1)
    trajectory, arm_sides = _trajectory_features(
        base,
        selected,
        fps=fps,
        aspect=aspect,
        contact_margin=config.contact_margin,
    )
    optical = _optical_flow_features(
        base,
        selected,
        frame_paths,
        fps=fps,
        config=config,
        arm_sides=arm_sides,
        trajectory=trajectory,
    )
    motion = trajectory + optical
    enhanced = np.concatenate([base.astype(np.float32), motion], axis=1)
    if (
        enhanced.shape
        != (len(base), len(MOTION_GRU_FEATURE_NAMES))
        or not np.isfinite(enhanced).all()
    ):
        raise RuntimeError(
            f"Invalid enhanced feature matrix: {enhanced.shape}"
        )
    return enhanced


def _artifact_is_current(
    destination: Path,
    fingerprint: str,
) -> bool:
    if not destination.is_file():
        return False
    try:
        metadata = feature_metadata(destination)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema") == "ai_referee.motion_features"
        and metadata.get("schema_version") == 2
        and metadata.get("config_fingerprint") == fingerprint
        and metadata.get("feature_names") == MOTION_GRU_FEATURE_NAMES
    )


def extract_motion_manifest(
    config_path: str | Path,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    config = load_motion_feature_config(config_path)
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    manifest = pd.read_csv(config.manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        manifest = manifest.head(limit)
    fingerprint = motion_schema_fingerprint(config)
    logger = configure_logging(config.logs_dir / "motion_features.log")
    logger.info(
        "examples=%d flow_width=%d preset=%s features=%d",
        len(manifest),
        config.flow_width,
        config.flow_preset,
        len(MOTION_GRU_FEATURE_NAMES),
    )
    for number, (_, row) in enumerate(manifest.iterrows(), start=1):
        destination = motion_feature_path(config.motion_features_dir, row)
        if (
            not overwrite
            and _artifact_is_current(destination, fingerprint)
        ):
            logger.info(
                "[%d/%d] cached %s %s",
                number,
                len(manifest),
                row["example_id"],
                row["view_id"],
            )
            continue
        base_path = feature_path(config.base_features_dir, row)
        if not base_path.is_file():
            raise FileNotFoundError(
                f"Base feature artifact not found: {base_path}"
            )
        with np.load(base_path, allow_pickle=False) as loaded:
            base = loaded["features"].astype(np.float32)
        base_metadata = feature_metadata(base_path)
        if base_metadata.get("feature_names") != FEATURE_NAMES:
            raise ValueError(f"Base feature schema mismatch in {base_path}")
        selected = [
            int(value)
            for value in base_metadata.get("selected_frame_indices", [])
        ]
        frames_dir = project_path(str(row["frames_dir"]))
        frame_paths = sorted_frames(frames_dir)
        fps_value = pd.to_numeric(row.get("fps"), errors="coerce")
        fps = (
            float(fps_value)
            if pd.notna(fps_value) and float(fps_value) > 0
            else 25.0
        )
        enhanced = augment_motion_features(
            base,
            selected,
            frame_paths,
            fps=fps,
            config=config,
        )
        metadata = {
            "schema": "ai_referee.motion_features",
            "schema_version": 2,
            "config_fingerprint": fingerprint,
            "feature_names": MOTION_GRU_FEATURE_NAMES,
            "base_feature_names": FEATURE_NAMES,
            "motion_feature_names": MOTION_FEATURE_NAMES,
            "selected_frame_indices": selected,
            "example_id": str(row["example_id"]),
            "view_id": str(row["view_id"]),
            "label": int(row["label"]),
            "domain": str(row["domain"]),
            "source_group": str(row["source_group"]),
            "fps": fps,
            "base_feature_path": str(base_path),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".temporary.npz")
        np.savez_compressed(
            temporary,
            features=enhanced,
            metadata=json.dumps(metadata, sort_keys=True),
        )
        temporary.replace(destination)
        logger.info(
            "[%d/%d] extracted %s %s shape=%s",
            number,
            len(manifest),
            row["example_id"],
            row["view_id"],
            tuple(enhanced.shape),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add ball/arm trajectory and optical-flow features to the "
            "existing temporal artifacts."
        )
    )
    parser.add_argument("--config", default="configs/motion_gru.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    extract_motion_manifest(
        args.config,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
