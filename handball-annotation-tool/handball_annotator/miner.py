from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import AppConfig
from .geometry import ArmMatch, closest_arm
from .runtime import get_device, seed_everything

ProgressCallback = Callable[[float, str], None]


@dataclass
class PendingCandidate:
    candidate_id: str
    center_frame: int
    player_id: int
    match: ArmMatch
    frames_before: int
    clean_frames: list[np.ndarray]
    evidence_frames: list[np.ndarray]


def _json_value(value: object) -> object:
    """Convert NumPy values that the standard JSON encoder cannot handle."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _codec_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("Cannot encode an empty candidate clip")
    height, width = frames[0].shape[:2]
    temporary = path.with_name(f"{path.stem}.temporary.mp4")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {temporary}")
    for frame in frames:
        writer.write(frame)
    writer.release()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            temporary.replace(path)
        else:
            temporary.unlink(missing_ok=True)
    else:
        temporary.replace(path)


def _save_candidate(candidate: PendingCandidate, config: AppConfig, source: Path, fps: float) -> Path:
    directory = config.candidates_dir / candidate.candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    clean_dir = directory / "clean_frames"
    evidence_dir = directory / "evidence_frames"
    clean_dir.mkdir(exist_ok=True)
    evidence_dir.mkdir(exist_ok=True)
    for index, (clean, evidence) in enumerate(zip(candidate.clean_frames, candidate.evidence_frames)):
        cv2.imwrite(str(clean_dir / f"frame_{index:04d}.jpg"), clean)
        cv2.imwrite(str(evidence_dir / f"frame_{index:04d}.jpg"), evidence)
    _codec_video(candidate.clean_frames, directory / "clean.mp4", fps)
    _codec_video(candidate.evidence_frames, directory / "evidence.mp4", fps)
    metadata = {
        "candidate_id": candidate.candidate_id,
        "source_name": source.name,
        "source_path": str(source),
        "center_frame": candidate.center_frame,
        "center_time_seconds": round(candidate.center_frame / fps, 3),
        "frames_before": candidate.frames_before,
        "frames_after": len(candidate.clean_frames) - candidate.frames_before - 1,
        "fps": fps,
        "player_track_id": candidate.player_id,
        "closest_arm": asdict(candidate.match),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_value), encoding="utf-8"
    )
    return directory


def _candidate_id(source: Path, frame_index: int) -> str:
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{source.stem}_{digest}_f{frame_index:08d}"


def create_manual_candidate(
    parent_candidate: Path,
    center_frame: int,
    config: AppConfig,
) -> Path:
    """Create a clean 41-frame candidate centered where the annotator chooses."""
    parent_metadata = json.loads((parent_candidate / "metadata.json").read_text(encoding="utf-8"))
    return create_manual_candidate_from_video(
        Path(parent_metadata["source_path"]), center_frame, config,
        parent_candidate_id=str(parent_metadata["candidate_id"]),
    )


def create_manual_candidate_from_video(
    source: Path,
    center_frame: int,
    config: AppConfig,
    parent_candidate_id: str | None = None,
) -> Path:
    """Create a clean 41-frame candidate directly from an uploaded video."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Original uploaded video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Video cannot be opened: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = center_frame - config.frames_before
    end = center_frame + config.frames_after
    if start < 0 or (total_frames and end >= total_frames):
        capture.release()
        raise ValueError("The selected center is too close to the beginning or end for a full 41-frame window.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[np.ndarray] = []
    try:
        for _ in range(start, end + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    expected = config.frames_before + config.frames_after + 1
    if len(frames) != expected:
        raise RuntimeError(f"Could only read {len(frames)} of the required {expected} frames.")

    candidate_id = f"{_candidate_id(source, center_frame)}_manual"
    directory = config.candidates_dir / candidate_id
    clean_dir, evidence_dir = directory / "clean_frames", directory / "evidence_frames"
    clean_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in (*clean_dir.glob("*.jpg"), *evidence_dir.glob("*.jpg")):
        old_frame.unlink()
    for index, frame in enumerate(frames):
        cv2.imwrite(str(clean_dir / f"frame_{index:04d}.jpg"), frame)
        # Manual windows have no inferred evidence; the review toggle remains usable.
        cv2.imwrite(str(evidence_dir / f"frame_{index:04d}.jpg"), frame)
    _codec_video(frames, directory / "clean.mp4", fps)
    _codec_video(frames, directory / "evidence.mp4", fps)
    metadata = {
        "candidate_id": candidate_id,
        "source_name": source.name,
        "source_path": str(source),
        "center_frame": int(center_frame),
        "center_time_seconds": round(center_frame / fps, 3),
        "frames_before": config.frames_before,
        "frames_after": config.frames_after,
        "fps": fps,
        "player_track_id": None,
        "closest_arm": None,
        "manual_selection": True,
        "parent_candidate_id": parent_candidate_id,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def mine_video(source: Path, config: AppConfig, progress: ProgressCallback | None = None) -> list[Path]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install requirements.txt first.") from exc

    seed_everything()
    device = get_device(config.device)
    detector = YOLO(config.detector)
    pose_model = YOLO(config.pose_model)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Video cannot be opened: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=config.frames_before + 1)
    pending: list[PendingCandidate] = []
    saved: list[Path] = []
    ball_trail: deque[tuple[int, int]] = deque(maxlen=30)
    last_candidate_frame = -config.candidate_cooldown_frames
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            evidence = frame.copy()
            result = detector.track(frame, persist=True, tracker=config.tracker, conf=config.confidence,
                                    device=device, verbose=False)[0]
            frame_match: tuple[int, ArmMatch] | None = None
            if result.boxes is not None and result.boxes.cls is not None:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = (result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None
                             else np.arange(len(boxes), dtype=int))
                balls = np.flatnonzero(classes == config.ball_class_id)
                people = np.flatnonzero(classes == config.person_class_id)
                if balls.size:
                    ball_box = boxes[balls[np.argmax((boxes[balls, 2] - boxes[balls, 0]) *
                                                     (boxes[balls, 3] - boxes[balls, 1]))]]
                    ball = np.array([(ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2])
                    ball_radius = float(
                        ((ball_box[2] - ball_box[0]) + (ball_box[3] - ball_box[1])) / 4
                    )
                    ball_point = tuple(ball.astype(int))
                    ball_trail.append(ball_point)
                    cv2.rectangle(evidence, tuple(ball_box[:2].astype(int)), tuple(ball_box[2:].astype(int)), (0, 255, 255), 2)
                    for first, second in zip(ball_trail, list(ball_trail)[1:]):
                        cv2.line(evidence, first, second, (0, 255, 255), 2)
                    if people.size:
                        centers = np.column_stack(((boxes[people, 0] + boxes[people, 2]) / 2,
                                                   (boxes[people, 1] + boxes[people, 3]) / 2))
                        order = people[np.argsort(np.linalg.norm(centers - ball, axis=1))[:config.max_nearby_players]]
                        for person_index in order:
                            x1, y1, x2, y2 = boxes[person_index].astype(int)
                            margin = int(max(x2 - x1, y2 - y1) * config.nearby_player_margin)
                            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                            x2, y2 = min(frame.shape[1], x2 + margin), min(frame.shape[0], y2 + margin)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size == 0:
                                continue
                            pose = pose_model.predict(crop, conf=config.confidence, device=device, verbose=False)[0]
                            if pose.keypoints is None or len(pose.keypoints.data) == 0:
                                continue
                            keypoints = pose.keypoints.data[0].cpu().numpy()
                            keypoints[:, 0] += x1
                            keypoints[:, 1] += y1
                            match = closest_arm(
                                ball, keypoints, float(y2 - y1), config.pose_keypoint_confidence,
                                ball_radius,
                            )
                            player_id = int(track_ids[person_index])
                            cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 180, 0), 2)
                            for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)):
                                if min(keypoints[a, 2], keypoints[b, 2]) >= 0.25:
                                    cv2.line(evidence, tuple(keypoints[a, :2].astype(int)),
                                             tuple(keypoints[b, :2].astype(int)), (255, 100, 0), 3)
                            if match and (frame_match is None or match.normalized_distance < frame_match[1].normalized_distance):
                                frame_match = (player_id, match)
            if frame_match:
                player_id, match = frame_match
                color = (0, 0, 255) if match.normalized_distance <= config.arm_distance_threshold else (255, 100, 0)
                cv2.line(evidence, match.start, match.end, color, 5)
                cv2.putText(evidence, f"P{player_id} {match.side} {match.segment}: {match.normalized_distance:.3f} body heights",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            history.append((frame.copy(), evidence.copy()))
            for item in pending:
                if index > item.center_frame:
                    item.clean_frames.append(frame.copy())
                    item.evidence_frames.append(evidence.copy())
            completed = [item for item in pending if index - item.center_frame >= config.frames_after]
            for item in completed:
                saved.append(_save_candidate(item, config, source, fps))
                pending.remove(item)

            if (frame_match and frame_match[1].normalized_distance <= config.arm_distance_threshold
                    and index - last_candidate_frame >= config.candidate_cooldown_frames):
                clean = [pair[0].copy() for pair in history]
                overlays = [pair[1].copy() for pair in history]
                pending.append(PendingCandidate(_candidate_id(source, index), index, frame_match[0], frame_match[1],
                                                len(clean) - 1, clean, overlays))
                last_candidate_frame = index
            index += 1
            if progress and (index % 10 == 0 or index == frame_count):
                fraction = index / frame_count if frame_count else 0.0
                progress(min(fraction, 1.0), f"Processed {index:,} of {frame_count:,} frames; {len(saved)} candidates saved")
    finally:
        capture.release()
    for item in pending:
        if len(item.clean_frames) > config.frames_before:
            saved.append(_save_candidate(item, config, source, fps))
    if progress:
        progress(1.0, f"Finished: {index:,} frames processed and {len(saved)} candidates found")
    return saved
