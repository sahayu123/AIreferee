from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmMatch:
    side: str
    segment: str
    distance: float
    normalized_distance: float
    start: tuple[int, int]
    end: tuple[int, int]


def point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-9:
        return float(np.linalg.norm(point - start))
    scale = float(np.clip(np.dot(point - start, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + scale * delta)))


def closest_arm(
    ball: np.ndarray,
    keypoints: np.ndarray,
    player_height: float,
    minimum_keypoint_confidence: float = 0.5,
    ball_radius: float = 0.0,
) -> ArmMatch | None:
    if player_height <= 1.0 or keypoints.shape[0] < 11:
        return None
    best: ArmMatch | None = None
    for side, shoulder, elbow, wrist in (("left", 5, 7, 9), ("right", 6, 8, 10)):
        for segment, first, second in (("upper_arm", shoulder, elbow), ("forearm", elbow, wrist)):
            if min(float(keypoints[first, 2]), float(keypoints[second, 2])) < minimum_keypoint_confidence:
                continue
            start, end = keypoints[first, :2], keypoints[second, :2]
            # Contact is between the arm and the ball's edge, not its center.
            distance = max(0.0, point_to_segment(ball, start, end) - ball_radius)
            # NumPy scalar integers are not supported by json.dumps. Convert
            # every coordinate to a native Python int before metadata export.
            start_point = tuple(int(value) for value in start)
            end_point = tuple(int(value) for value in end)
            match = ArmMatch(side, segment, distance, distance / player_height,
                             start_point, end_point)
            if best is None or match.normalized_distance < best.normalized_distance:
                best = match
    return best
