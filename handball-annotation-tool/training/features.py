from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from handball_annotator.geometry import point_to_segment
from handball_annotator.runtime import get_device, seed_everything

from .config import FeatureConfig, PROJECT_ROOT, load_feature_config
from .logging_utils import configure_logging
from .manifest import sorted_frames

POSE_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
ARM_SEGMENTS = (
    ("left_upper_arm", "left_shoulder", "left_elbow"),
    ("left_forearm", "left_elbow", "left_wrist"),
    ("right_upper_arm", "right_shoulder", "right_elbow"),
    ("right_forearm", "right_elbow", "right_wrist"),
)

FEATURE_NAMES = [
    "ball_x", "ball_y", "ball_w", "ball_h", "ball_conf", "ball_valid",
    "player_x", "player_y", "player_w", "player_h", "player_conf", "player_valid",
]
for landmark in POSE_INDICES:
    FEATURE_NAMES.extend([f"{landmark}_x", f"{landmark}_y", f"{landmark}_visibility", f"{landmark}_valid"])
FEATURE_NAMES.extend([
    "left_wrist_distance", "right_wrist_distance",
    "left_elbow_distance", "right_elbow_distance",
    "left_upper_arm_distance", "left_forearm_distance",
    "right_upper_arm_distance", "right_forearm_distance",
    "left_arm_angle", "right_arm_angle",
    "ball_relative_x", "ball_relative_y",
    "ball_dx", "ball_dy", "ball_speed", "ball_direction_change",
    "left_wrist_motion", "right_wrist_motion",
    "pose_valid_fraction", "arm_min_distance",
])


@dataclass
class Detection:
    box: np.ndarray
    confidence: float
    track_id: int | None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2], dtype=float)


@dataclass
class FrameResult:
    vector: np.ndarray
    quality: float
    overlay: np.ndarray


def _box_distance(point: np.ndarray, box: np.ndarray) -> float:
    x = float(np.clip(point[0], box[0], box[2]))
    y = float(np.clip(point[1], box[1], box[3]))
    return float(np.linalg.norm(point - np.array([x, y])))


def _angle(first: np.ndarray, center: np.ndarray, last: np.ndarray) -> float:
    left, right = first - center, last - center
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-9:
        return 0.0
    return float(np.arccos(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)) / math.pi)


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


class FeatureExtractor:
    def __init__(self, config: FeatureConfig):
        if not config.detector.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {config.detector}")
        if not config.mediapipe_model.is_file():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {config.mediapipe_model}. "
                "Download pose_landmarker_lite.task as described in the README."
            )
        try:
            import mediapipe as mp
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install MediaPipe and Ultralytics before extracting features.") from exc
        self.mp = mp
        self.config = config
        self.device = get_device(config.device)
        self.detector = YOLO(str(config.detector))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(config.mediapipe_model)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=config.pose_confidence,
            min_pose_presence_confidence=config.pose_presence,
            min_tracking_confidence=0.3,
        )
        self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self.pose.close()

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _detections(self, frame: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        result = self.detector.track(
            frame, persist=True, tracker=self.config.tracker,
            conf=min(self.config.confidence, self.config.ball_confidence),
            classes=[0, 32], device=self.device, verbose=False,
        )[0]
        people: list[Detection] = []
        balls: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return people, balls
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy()
        ids = result.boxes.id.detach().cpu().numpy().astype(int) if result.boxes.id is not None else None
        for index, (box, class_id, confidence) in enumerate(zip(boxes, classes, confidences)):
            detection = Detection(box.astype(float), float(confidence), int(ids[index]) if ids is not None else None)
            if class_id == 0 and confidence >= self.config.confidence:
                people.append(detection)
            elif class_id == 32 and confidence >= self.config.ball_confidence:
                balls.append(detection)
        return people, balls

    def _landmarks(self, frame: np.ndarray, person: Detection) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        height, width = frame.shape[:2]
        margin = self.config.crop_margin
        box = person.box
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        x1 = max(0, int(box[0] - margin * box_w))
        y1 = max(0, int(box[1] - margin * box_h))
        x2 = min(width, int(box[2] + margin * box_w))
        y2 = min(height, int(box[3] + margin * box_h))
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 24 or crop.shape[1] < 16:
            return None
        # Tiny broadcast players benefit from explicit high-quality upscaling.
        # Square padding preserves body proportions and makes coordinate mapping deterministic.
        crop_height, crop_width = crop.shape[:2]
        side = max(crop_height, crop_width)
        pad_x, pad_y = (side - crop_width) // 2, (side - crop_height) // 2
        square = np.zeros((side, side, 3), dtype=np.uint8)
        square[pad_y:pad_y + crop_height, pad_x:pad_x + crop_width] = crop
        square = cv2.resize(
            square, (self.config.pose_input_size, self.config.pose_input_size),
            interpolation=cv2.INTER_LANCZOS4,
        )
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.pose.detect(image)
        if not result.pose_landmarks:
            return None
        all_landmarks = result.pose_landmarks[0]
        selected: dict[str, np.ndarray] = {}
        for name, index in POSE_INDICES.items():
            landmark = all_landmarks[index]
            selected[name] = np.array([
                x1 + landmark.x * side - pad_x,
                y1 + landmark.y * side - pad_y,
                float(landmark.visibility or 0.0),
                float(landmark.presence or 0.0),
            ], dtype=float)
        return selected, np.array([x1, y1, x2, y2], dtype=float)

    def _pose_score(
        self, landmarks: dict[str, np.ndarray], ball: Detection | None, player_height: float
    ) -> tuple[float, dict[str, float]]:
        visibility_threshold = self.config.pose_presence
        distances: dict[str, float] = {}
        valid_distances: list[float] = []
        if ball is not None and player_height > 1:
            ball_radius = max(ball.box[2] - ball.box[0], ball.box[3] - ball.box[1]) / 2
            for segment, first, second in ARM_SEGMENTS:
                first_point, second_point = landmarks[first], landmarks[second]
                valid = min(first_point[2], first_point[3], second_point[2], second_point[3]) >= visibility_threshold
                distance = (
                    max(point_to_segment(ball.center, first_point[:2], second_point[:2]) - ball_radius, 0.0)
                    / player_height if valid else 0.0
                )
                distances[segment] = distance
                if valid:
                    valid_distances.append(distance)
            score = min(valid_distances) if valid_distances else 1e6
        else:
            score = 1e6
        return score, distances

    def _frame(
        self, frame: np.ndarray, previous_ball: np.ndarray | None, preferred_player_id: int | None
    ) -> tuple[FrameResult, np.ndarray | None, int | None]:
        height, width = frame.shape[:2]
        people, balls = self._detections(frame)
        ball: Detection | None = None
        if balls:
            if previous_ball is None:
                ball = max(balls, key=lambda item: item.confidence)
            else:
                ball = min(balls, key=lambda item: np.linalg.norm(item.center - previous_ball) / max(item.confidence, 0.05))

        if ball is not None:
            people = sorted(people, key=lambda person: _box_distance(ball.center, person.box))
        elif preferred_player_id is not None:
            people = sorted(people, key=lambda person: person.track_id != preferred_player_id)
        else:
            people = sorted(people, key=lambda person: -person.confidence)

        chosen: tuple[Detection, dict[str, np.ndarray], dict[str, float]] | None = None
        best_score = 1e9
        for person in people[:self.config.max_nearby_players]:
            pose_result = self._landmarks(frame, person)
            if pose_result is None:
                continue
            landmarks, _ = pose_result
            player_height = max(person.box[3] - person.box[1], 1.0)
            score, distances = self._pose_score(landmarks, ball, player_height)
            if chosen is None or score < best_score:
                chosen, best_score = (person, landmarks, distances), score

        values = {name: 0.0 for name in FEATURE_NAMES}
        overlay = frame.copy()
        if ball is not None:
            center = ball.center
            values.update({
                "ball_x": center[0] / width, "ball_y": center[1] / height,
                "ball_w": (ball.box[2] - ball.box[0]) / width,
                "ball_h": (ball.box[3] - ball.box[1]) / height,
                "ball_conf": ball.confidence, "ball_valid": 1.0,
            })
            x1, y1, x2, y2 = map(int, ball.box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(overlay, f"ball {ball.confidence:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        selected_person = chosen[0] if chosen is not None else (people[0] if people else None)
        selected_id = preferred_player_id
        if selected_person is not None:
            selected_id = selected_person.track_id
            player_height = max(selected_person.box[3] - selected_person.box[1], 1.0)
            player_center = selected_person.center
            values.update({
                "player_x": player_center[0] / width, "player_y": player_center[1] / height,
                "player_w": (selected_person.box[2] - selected_person.box[0]) / width,
                "player_h": player_height / height,
                "player_conf": selected_person.confidence, "player_valid": 1.0,
            })
            x1, y1, x2, y2 = map(int, selected_person.box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (80, 255, 80), 2)
            cv2.putText(overlay, f"selected player {selected_person.track_id}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1, cv2.LINE_AA)
        if chosen is not None:
            person, landmarks, distances = chosen
            selected_id = person.track_id
            player_height = max(person.box[3] - person.box[1], 1.0)
            player_center = person.center
            values.update({
                "player_x": player_center[0] / width, "player_y": player_center[1] / height,
                "player_w": (person.box[2] - person.box[0]) / width,
                "player_h": player_height / height,
                "player_conf": person.confidence, "player_valid": 1.0,
            })
            valid_count = 0
            for name, point in landmarks.items():
                valid = min(point[2], point[3]) >= self.config.pose_presence
                values[f"{name}_x"] = point[0] / width if valid else 0.0
                values[f"{name}_y"] = point[1] / height if valid else 0.0
                values[f"{name}_visibility"] = min(point[2], point[3])
                values[f"{name}_valid"] = float(valid)
                valid_count += int(valid)
                if valid:
                    cv2.circle(overlay, tuple(point[:2].astype(int)), 3, (255, 80, 80), -1)
            values["pose_valid_fraction"] = valid_count / len(POSE_INDICES)
            for segment, first, second in ARM_SEGMENTS:
                values[f"{segment}_distance"] = distances.get(segment, 0.0)
                first_point, second_point = landmarks[first], landmarks[second]
                if min(first_point[2], first_point[3], second_point[2], second_point[3]) >= self.config.pose_presence:
                    cv2.line(overlay, tuple(first_point[:2].astype(int)), tuple(second_point[:2].astype(int)),
                             (255, 80, 80), 2)
            if ball is not None:
                for joint in ("left_wrist", "right_wrist", "left_elbow", "right_elbow"):
                    point = landmarks[joint]
                    if min(point[2], point[3]) >= self.config.pose_presence:
                        values[f"{joint}_distance"] = max(
                            np.linalg.norm(ball.center - point[:2])
                            - max(ball.box[2] - ball.box[0], ball.box[3] - ball.box[1]) / 2, 0.0
                        ) / player_height
                values["ball_relative_x"] = (ball.center[0] - player_center[0]) / player_height
                values["ball_relative_y"] = (ball.center[1] - player_center[1]) / player_height
                nonzero = [distances.get(name, 0.0) for name, _, _ in ARM_SEGMENTS if distances.get(name, 0.0) > 0]
                values["arm_min_distance"] = min(nonzero) if nonzero else 0.0
            for side in ("left", "right"):
                shoulder, elbow, wrist = (landmarks[f"{side}_{joint}"][:2] for joint in ("shoulder", "elbow", "wrist"))
                values[f"{side}_arm_angle"] = _angle(shoulder, elbow, wrist)

        vector = np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
        quality = values["ball_valid"] * 2 + values["pose_valid_fraction"] + values["player_valid"]
        return FrameResult(vector, quality, overlay), ball.center if ball is not None else previous_ball, selected_id

    def extract(self, frame_paths: list[Path]) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
        if not frame_paths:
            raise ValueError("Cannot extract features from an empty frame sequence")
        # Reset Ultralytics' persistent tracker between independent views.
        self.detector.predictor = None
        results: list[FrameResult] = []
        previous_ball: np.ndarray | None = None
        preferred_player_id: int | None = None
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {path}")
            result, previous_ball, preferred_player_id = self._frame(frame, previous_ball, preferred_player_id)
            results.append(result)
        matrix = np.stack([item.vector for item in results])
        self._add_motion(matrix)
        selected = self._temporal_indices(results, self.config.sequence_length)
        return matrix[selected], [results[index].overlay for index in selected], selected

    @staticmethod
    def _add_motion(matrix: np.ndarray) -> None:
        index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
        previous_direction: np.ndarray | None = None
        for frame_index in range(1, len(matrix)):
            if matrix[frame_index, index["ball_valid"]] and matrix[frame_index - 1, index["ball_valid"]]:
                delta = matrix[frame_index, [index["ball_x"], index["ball_y"]]] - matrix[
                    frame_index - 1, [index["ball_x"], index["ball_y"]]
                ]
                matrix[frame_index, index["ball_dx"]] = delta[0]
                matrix[frame_index, index["ball_dy"]] = delta[1]
                matrix[frame_index, index["ball_speed"]] = np.linalg.norm(delta)
                if previous_direction is not None and np.linalg.norm(delta) > 1e-8 and np.linalg.norm(previous_direction) > 1e-8:
                    cosine = np.dot(delta, previous_direction) / (np.linalg.norm(delta) * np.linalg.norm(previous_direction))
                    matrix[frame_index, index["ball_direction_change"]] = np.arccos(np.clip(cosine, -1, 1)) / math.pi
                previous_direction = delta
            for side in ("left", "right"):
                valid = index[f"{side}_wrist_valid"]
                if matrix[frame_index, valid] and matrix[frame_index - 1, valid]:
                    current = matrix[frame_index, [index[f"{side}_wrist_x"], index[f"{side}_wrist_y"]]]
                    previous = matrix[frame_index - 1, [index[f"{side}_wrist_x"], index[f"{side}_wrist_y"]]]
                    matrix[frame_index, index[f"{side}_wrist_motion"]] = np.linalg.norm(current - previous)

    @staticmethod
    def _temporal_indices(results: list[FrameResult], length: int) -> list[int]:
        if len(results) <= length:
            indices = list(range(len(results)))
            while len(indices) < length:
                indices.append(indices[-1])
            return indices
        bins = np.array_split(np.arange(len(results)), length)
        return [int(max(section, key=lambda index: results[int(index)].quality)) for section in bins]


def _contact_sheet(frames: Iterable[np.ndarray], destination: Path) -> None:
    tiles = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = 320 / max(width, 1)
        tile = cv2.resize(frame, (320, max(1, int(height * scale))))
        tiles.append(tile)
    target_height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.copyMakeBorder(tile, 0, target_height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT)
             for tile in tiles]
    rows = [cv2.hconcat(tiles[index:index + 4]) for index in range(0, len(tiles), 4)]
    max_width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT) for row in rows]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.vconcat(rows)):
        raise RuntimeError(f"Could not save overlay: {destination}")


def feature_path(root: Path, row: pd.Series) -> Path:
    return root / str(row["domain"]) / _safe_name(str(row["example_id"])) / f"{_safe_name(str(row['view_id']))}.npz"


def extract_manifest(
    config: FeatureConfig,
    overwrite: bool = False,
    domain: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> None:
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}. Run training.manifest first.")
    manifest = pd.read_csv(config.manifest)
    if domain:
        manifest = manifest[manifest["domain"] == domain]
    if limit is not None:
        manifest = manifest.head(limit)
    config.features_dir.mkdir(parents=True, exist_ok=True)
    config.overlays_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(config.features_dir.parent / "logs" / "feature_extraction.log", verbose)
    overlay_counts: dict[str, int] = {}
    with FeatureExtractor(config) as extractor:
        for item_number, (_, row) in enumerate(manifest.iterrows(), start=1):
            destination = feature_path(config.features_dir, row)
            if destination.is_file() and not overwrite:
                logger.info("[%d/%d] cached %s %s", item_number, len(manifest), row["example_id"], row["view_id"])
                continue
            frames_dir = PROJECT_ROOT / str(row["frames_dir"])
            frames = sorted_frames(frames_dir)
            logger.info("[%d/%d] extracting %s %s (%d frames)", item_number, len(manifest),
                        row["example_id"], row["view_id"], len(frames))
            features, overlays, selected = extractor.extract(frames)
            destination.parent.mkdir(parents=True, exist_ok=True)
            metadata = {
                "example_id": str(row["example_id"]), "view_id": str(row["view_id"]),
                "label": int(row["label"]), "domain": str(row["domain"]),
                "source_group": str(row["source_group"]), "selected_frame_indices": selected,
                "feature_names": FEATURE_NAMES,
            }
            temporary = destination.with_suffix(".temporary.npz")
            np.savez_compressed(temporary, features=features, metadata=json.dumps(metadata))
            temporary.replace(destination)
            domain_name = str(row["domain"])
            if overlay_counts.get(domain_name, 0) < config.overlay_examples_per_domain:
                overlay = config.overlays_dir / domain_name / f"{_safe_name(str(row['example_id']))}_{row['view_id']}.jpg"
                _contact_sheet(overlays, overlay)
                overlay_counts[domain_name] = overlay_counts.get(domain_name, 0) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLO/ByteTrack/MediaPipe temporal features.")
    parser.add_argument("--config", default="configs/mediapipe_features.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--domain", choices=["native", "imported"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    seed_everything(42)
    extract_manifest(load_feature_config(args.config), args.overwrite, args.domain, args.limit, args.verbose)


if __name__ == "__main__":
    main()
