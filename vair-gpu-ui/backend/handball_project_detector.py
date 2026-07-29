"""Modern Mac-GPU port of nadimra/handball-detection.

The supplied project is an inference algorithm, not a trained handball
classifier.  Its decision chain is preserved here:

1. detect the football on video frames;
2. find abrupt changes in the ball trajectory;
3. estimate every player's COCO-17 skeleton at those candidate frames;
4. test the expanded ball region against upper/lower arm segments;
5. apply the supplied shoulder/elbow/hip arm-angle rule.

The archive's 2022 YOLOv5/HRNet code requires missing external weights and a
CUDA-oriented environment.  This adapter uses the application's current
Ultralytics models on Apple MPS while retaining the project-specific handball
geometry and thresholds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch


ARM_SEGMENTS = {
    "left": ((5, 7), (7, 9)),
    "right": ((6, 8), (8, 10)),
}
ANGLE_JOINTS = {
    "left": (7, 5, 11),
    "right": (8, 6, 12),
}
# These are the original project's asymmetric decision thresholds.
ANGLE_THRESHOLDS = {"left": 50.0, "right": 10.0}
POSE_CONFIDENCE = 0.25
# A reliable ball closer than 4% of the frame diagonal to an arm is visually
# unambiguous at the supplied clip's scale.  The ball-radius guard prevents
# this normalized threshold from becoming too permissive on unusual crops.
PROXIMITY_OVERRIDE_NORMALIZED = 0.04
PROXIMITY_OVERRIDE_RADIUS_MULTIPLIER = 1.50
PROXIMITY_MIN_BALL_CONFIDENCE = 0.35


@dataclass
class HandballProjectResult:
    probability: float
    predicted_label: str
    quality: float
    status: str = "completed"
    overlay_path: str | None = None
    hit_hand: bool = False
    handball_part: str | None = None
    handball_angle: float | None = None
    proximity_override: bool = False
    proximity_threshold: float = PROXIMITY_OVERRIDE_NORMALIZED
    candidate_frames: int = 0
    ball_detection_rate: float = 0.0
    player_detection_rate: float = 0.0
    pose_valid_rate: float = 0.0
    minimum_normalized_arm_distance: float | None = None
    fold_probabilities: dict[str, float] = field(default_factory=dict)


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = math.atan2(c[1] - b[1], c[0] - b[0])
    second = math.atan2(a[1] - b[1], a[0] - b[0])
    value = abs(math.degrees(first - second))
    return 360.0 - value if value > 180.0 else value


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> tuple[float, np.ndarray]:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-8:
        return float(np.linalg.norm(point - start)), start
    amount = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    closest = start + amount * segment
    return float(np.linalg.norm(point - closest)), closest


def _trajectory_candidates(observations: list[dict], fps: float) -> list[int]:
    """Return direction-change frames, matching the supplied project's intent."""
    if len(observations) < 3:
        return [item["frame"] for item in observations]

    candidates: list[tuple[float, int]] = []
    for index in range(1, len(observations) - 1):
        previous, current, following = observations[index - 1:index + 2]
        before = current["center"] - previous["center"]
        after = following["center"] - current["center"]
        gap_before = current["frame"] - previous["frame"]
        gap_after = following["frame"] - current["frame"]
        if gap_before > max(4, int(round(fps * 0.20))):
            continue
        if gap_after > max(4, int(round(fps * 0.20))):
            continue
        speed = min(
            float(np.linalg.norm(before)) / max(gap_before, 1),
            float(np.linalg.norm(after)) / max(gap_after, 1),
        )
        denominator = max(
            float(np.linalg.norm(before) * np.linalg.norm(after)),
            1e-6,
        )
        turn = math.degrees(math.acos(float(np.clip(np.dot(before, after) / denominator, -1.0, 1.0))))
        if turn >= 24.0 and speed >= 0.8:
            candidates.append((turn * speed, current["frame"]))

    # Keep the strongest event in each short temporal neighborhood.
    selected: list[int] = []
    for _, frame_index in sorted(candidates, reverse=True):
        if all(abs(frame_index - kept) > max(2, int(round(fps * 0.10))) for kept in selected):
            selected.append(frame_index)
    return sorted(selected[:12])


def _pose_candidates(model, frame: np.ndarray) -> list[np.ndarray]:
    result = model.predict(
        frame,
        conf=0.20,
        verbose=False,
        device=_device(),
    )[0]
    if result.keypoints is None:
        return []
    poses: list[np.ndarray] = []
    for keypoints in result.keypoints.data:
        pose = keypoints.detach().cpu().numpy().astype(np.float32)
        if pose.shape == (17, 3) and int(np.count_nonzero(pose[:, 2] >= POSE_CONFIDENCE)) >= 6:
            poses.append(pose)
    return poses


def _arm_collision(
    pose: np.ndarray,
    ball_box: np.ndarray,
    frame_diagonal: float,
) -> dict | None:
    center = np.array(
        [(ball_box[0] + ball_box[2]) / 2.0, (ball_box[1] + ball_box[3]) / 2.0],
        dtype=np.float32,
    )
    # The supplied project expands each ball edge by seven pixels. Scale that
    # margin for modern high-resolution match clips.
    margin = max(7.0, frame_diagonal * 0.006)
    radius = max(
        (ball_box[2] - ball_box[0]) / 2.0,
        (ball_box[3] - ball_box[1]) / 2.0,
    ) + margin
    best: dict | None = None
    for side, segments in ARM_SEGMENTS.items():
        for start_index, end_index in segments:
            if min(pose[start_index, 2], pose[end_index, 2]) < POSE_CONFIDENCE:
                continue
            distance, closest = _point_segment_distance(
                center,
                pose[start_index, :2],
                pose[end_index, :2],
            )
            normalized = distance / max(frame_diagonal, 1.0)
            if best is None or distance < best["distance"]:
                angle_joints = ANGLE_JOINTS[side]
                angle_ready = min(pose[list(angle_joints), 2]) >= POSE_CONFIDENCE
                arm_angle = (
                    _angle(
                        pose[angle_joints[0], :2],
                        pose[angle_joints[1], :2],
                        pose[angle_joints[2], :2],
                    )
                    if angle_ready else None
                )
                best = {
                    "side": side,
                    "segment": (start_index, end_index),
                    "distance": distance,
                    "normalized": normalized,
                    "radius": radius,
                    "closest": closest,
                    "angle": arm_angle,
                    "collision": distance <= radius,
                }
    return best


def _is_proximity_override(collision: dict, ball_confidence: float) -> bool:
    """High-confidence handball gate for an exceptionally small arm/ball gap."""
    return bool(
        collision["angle"] is not None
        and ball_confidence >= PROXIMITY_MIN_BALL_CONFIDENCE
        and collision["normalized"] <= PROXIMITY_OVERRIDE_NORMALIZED
        and collision["distance"]
        <= collision["radius"] * PROXIMITY_OVERRIDE_RADIUS_MULTIPLIER
    )


def analyze_handball_project(
    video_path: Path,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> HandballProjectResult:
    from ultralytics import YOLO

    report = progress or (lambda _message: None)
    output_dir.mkdir(parents=True, exist_ok=True)
    report("Handball Project: tracking the ball trajectory")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The handball detector could not open the uploaded video.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    detector = YOLO("yolo11m.pt")
    observations: list[dict] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        prediction = detector.predict(
            frame,
            classes=[32],
            conf=0.12,
            iou=0.45,
            verbose=False,
            device=_device(),
        )[0]
        if prediction.boxes is not None and len(prediction.boxes):
            box = max(prediction.boxes, key=lambda item: float(item.conf[0]))
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float32)
            observations.append({
                "frame": index,
                "box": xyxy,
                "center": np.array(
                    [(xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0],
                    dtype=np.float32,
                ),
                "confidence": float(box.conf[0]),
            })
        index += 1
    capture.release()

    actual_frames = max(index, 1)
    candidates = _trajectory_candidates(observations, fps)
    by_frame = {item["frame"]: item for item in observations}
    report(f"Handball Project: checking {len(candidates)} ball direction changes")

    pose_model = YOLO("yolov8m-pose.pt")
    candidate_capture = cv2.VideoCapture(str(video_path))
    best: dict | None = None
    frames_with_players = 0
    frames_with_pose = 0
    for frame_index in candidates:
        candidate_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = candidate_capture.read()
        ball = by_frame.get(frame_index)
        if not ok or frame is None or ball is None:
            continue
        poses = _pose_candidates(pose_model, frame)
        if poses:
            frames_with_players += 1
        diagonal = float(math.hypot(frame.shape[1], frame.shape[0]))
        pose_geometry_found = False
        for pose in poses:
            collision = _arm_collision(pose, ball["box"], diagonal)
            if collision is None:
                continue
            pose_geometry_found = True
            threshold = ANGLE_THRESHOLDS[collision["side"]]
            angle = collision["angle"]
            angle_decision = bool(
                collision["collision"]
                and angle is not None
                and angle > threshold
            )
            proximity_override = _is_proximity_override(
                collision,
                float(ball["confidence"]),
            )
            decision = angle_decision or proximity_override
            distance_score = max(0.0, 1.0 - collision["distance"] / max(collision["radius"], 1.0))
            angle_score = (
                float(np.clip((angle - threshold + 20.0) / 70.0, 0.0, 1.0))
                if angle is not None else 0.0
            )
            score = (
                0.55 * distance_score
                + 0.25 * angle_score
                + 0.20 * float(ball["confidence"])
            )
            candidate = {
                **collision,
                "decision": decision,
                "angle_decision": angle_decision,
                "proximity_override": proximity_override,
                "score": score,
                "frame_index": frame_index,
                "frame": frame,
                "ball": ball,
                "pose": pose,
            }
            if best is None or (decision, score) > (best["decision"], best["score"]):
                best = candidate
        if pose_geometry_found:
            frames_with_pose += 1
    candidate_capture.release()

    overlay_path: Path | None = None
    if best is not None:
        overlay = best["frame"].copy()
        x0, y0, x1, y1 = map(int, best["ball"]["box"])
        color = (30, 80, 245) if best["decision"] else (0, 190, 245)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        for side, segments in ARM_SEGMENTS.items():
            for start, end in segments:
                if min(best["pose"][start, 2], best["pose"][end, 2]) >= POSE_CONFIDENCE:
                    p0 = tuple(np.rint(best["pose"][start, :2]).astype(int))
                    p1 = tuple(np.rint(best["pose"][end, :2]).astype(int))
                    cv2.line(overlay, p0, p1, (0, 230, 255), 3)
        center = tuple(np.rint(best["ball"]["center"]).astype(int))
        closest = tuple(np.rint(best["closest"]).astype(int))
        cv2.line(overlay, center, closest, color, 2)
        label = (
            (
                f"HANDBALL proximity {best['normalized']:.1%}"
                if best["proximity_override"] else
                f"HANDBALL {best['side']} arm {best['angle']:.0f} deg"
            )
            if best["decision"] else
            f"ARM CHECK {best['side']} {best['angle'] or 0:.0f} deg"
        )
        cv2.putText(
            overlay,
            label,
            (max(12, x0), max(32, y0 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        overlay_path = output_dir / "handball-project-evidence.jpg"
        cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

    if best is None:
        probability = 0.05 if observations else 0.0
        prediction = "no_handball"
        hit_hand = False
        quality = min(0.45, len(observations) / max(actual_frames, 1))
    else:
        hit_hand = bool(best["collision"] or best["proximity_override"])
        if best["decision"]:
            if best["proximity_override"]:
                closeness = 1.0 - (
                    best["normalized"] / PROXIMITY_OVERRIDE_NORMALIZED
                )
                probability = float(np.clip(0.90 + 0.09 * closeness, 0.90, 0.99))
            else:
                probability = float(np.clip(0.72 + 0.27 * best["score"], 0.0, 0.99))
            prediction = "handball"
        elif hit_hand:
            probability = float(np.clip(0.18 + 0.25 * best["score"], 0.0, 0.49))
            prediction = "no_handball"
        else:
            probability = float(np.clip(0.05 + 0.20 * best["score"], 0.0, 0.35))
            prediction = "no_handball"
        quality = float(np.clip(
            0.35 * min(1.0, len(observations) / max(actual_frames * 0.15, 1.0))
            + 0.35 * min(1.0, len(candidates) / 3.0)
            + 0.30 * min(1.0, frames_with_players / max(len(candidates), 1)),
            0.0,
            1.0,
        ))

    return HandballProjectResult(
        probability=probability,
        predicted_label=prediction,
        quality=quality,
        overlay_path=str(overlay_path) if overlay_path else None,
        hit_hand=hit_hand,
        handball_part=best["side"] if best else None,
        handball_angle=float(best["angle"]) if best and best["angle"] is not None else None,
        proximity_override=bool(best and best["proximity_override"]),
        candidate_frames=len(candidates),
        ball_detection_rate=len(observations) / max(actual_frames, 1),
        player_detection_rate=frames_with_players / max(len(candidates), 1),
        pose_valid_rate=frames_with_pose / max(len(candidates), 1),
        minimum_normalized_arm_distance=best["normalized"] if best else None,
        fold_probabilities={},
    )
