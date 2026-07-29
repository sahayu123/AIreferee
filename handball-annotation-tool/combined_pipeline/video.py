from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from training.manifest import sorted_frames

from .schemas import VideoContext


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _video_inside(directory: Path) -> Path | None:
    preferred = ("clip.mp4", "clean.mp4", "candidate.mp4")
    for name in preferred:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    videos = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[0] if videos else None


def _probe_video(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return fps, width, height, count


def _decode(
    video: Path,
    frames_dir: Path,
    *,
    max_frames: int | None,
    incident_time_seconds: float | None,
    incident_video: Path,
) -> tuple[tuple[Path, ...], float, int, int, int]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is None or total <= 0 or total <= max_frames:
        start = 0
        limit = max_frames
    else:
        center = (
            int(round(incident_time_seconds * fps))
            if incident_time_seconds is not None
            else total // 2
        )
        center = min(max(center, 0), total - 1)
        start = min(
            max(0, center - max_frames // 2),
            max(0, total - max_frames),
        )
        limit = max_frames
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = cv2.VideoWriter(
        str(incident_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create incident video: {incident_video}")
    paths: list[Path] = []
    index = 0
    while limit is None or index < limit:
        ok, frame = capture.read()
        if not ok:
            break
        destination = frames_dir / f"frame_{start + index:06d}.jpg"
        if not cv2.imwrite(
            str(destination),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        ):
            writer.release()
            capture.release()
            raise OSError(f"Could not write decoded frame: {destination}")
        writer.write(frame)
        paths.append(destination)
        index += 1
    writer.release()
    capture.release()
    if not paths:
        raise ValueError(f"No frames decoded from video: {video}")
    return tuple(paths), fps, width, height, start


def _encode_frames(
    frame_paths: tuple[Path, ...],
    destination: Path,
    fps: float,
) -> tuple[int, int]:
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"Could not create incident video: {destination}")
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            writer.release()
            raise ValueError(f"Could not read frame: {path}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()
    return width, height


def prepare_video(
    source: str | Path,
    output_dir: str | Path,
    *,
    max_frames: int | None = None,
    incident_time_seconds: float | None = None,
) -> VideoContext:
    source = Path(source).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if incident_time_seconds is not None and incident_time_seconds < 0:
        raise ValueError("incident_time_seconds cannot be negative")

    if source.is_file():
        if source.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {source.suffix}")
        original = output_dir / f"original{source.suffix.lower()}"
        if original != source:
            shutil.copy2(source, original)
        incident_video = output_dir / "incident.mp4"
        frame_paths, fps, width, height, start = _decode(
            original,
            output_dir / "frames",
            max_frames=max_frames,
            incident_time_seconds=incident_time_seconds,
            incident_video=incident_video,
        )
        source_video: Path | None = incident_video
    elif source.is_dir():
        frames_dir_candidates = (
            source / "frames",
            source / "clean_frames",
            source,
        )
        frame_paths = ()
        frames_dir = source
        for candidate in frames_dir_candidates:
            found = sorted_frames(candidate)
            if found:
                frame_paths = tuple(found)
                frames_dir = candidate
                break
        if not frame_paths:
            raise FileNotFoundError(f"No JPG frames found in {source}")
        available_video = _video_inside(source)
        if available_video is not None:
            fps, _, _, _ = _probe_video(available_video)
        else:
            fps = 25.0
        if max_frames is not None and len(frame_paths) > max_frames:
            center = (
                int(round(incident_time_seconds * fps))
                if incident_time_seconds is not None
                else len(frame_paths) // 2
            )
            center = min(max(center, 0), len(frame_paths) - 1)
            start = min(
                max(0, center - max_frames // 2),
                len(frame_paths) - max_frames,
            )
            frame_paths = frame_paths[start:start + max_frames]
        else:
            start = 0
        source_video = output_dir / "incident.mp4"
        width, height = _encode_frames(frame_paths, source_video, fps)
        return VideoContext(
            source=source,
            source_video=source_video,
            frames_dir=frames_dir,
            frame_paths=frame_paths,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=len(frame_paths) / max(fps, 1e-6),
            source_start_frame=start,
        )
    else:
        raise FileNotFoundError(f"Input does not exist: {source}")

    return VideoContext(
        source=source,
        source_video=source_video,
        frames_dir=frame_paths[0].parent,
        frame_paths=frame_paths,
        fps=fps,
        width=width,
        height=height,
        duration_seconds=len(frame_paths) / max(fps, 1e-6),
        source_start_frame=start,
    )
