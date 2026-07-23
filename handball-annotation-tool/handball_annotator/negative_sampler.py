from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .config import AppConfig
from .miner import _codec_video


def source_id(source: Path) -> str:
    stat = source.stat()
    value = f"{source.resolve()}:{stat.st_size}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def sampled_candidate_id(source: Path, center_frame: int) -> str:
    return f"{source.stem}_{source_id(source)}_f{center_frame:08d}_negative_sample"


def create_sample(source: Path, center_frame: int, config: AppConfig, root: Path) -> Path:
    """Extract one clean 41-frame sample without analyzing the rest of the video."""
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Video cannot be opened: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start = center_frame - config.frames_before
    end = center_frame + config.frames_after
    if start < 0 or (total_frames and end >= total_frames):
        capture.release()
        raise ValueError("The selected position cannot provide a complete 41-frame window.")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    try:
        for _ in range(config.frames_before + config.frames_after + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    expected = config.frames_before + config.frames_after + 1
    if len(frames) != expected:
        raise RuntimeError(f"Could only read {len(frames)} of the required {expected} frames.")

    candidate_id = sampled_candidate_id(source, center_frame)
    directory = root / candidate_id
    clean_frames = directory / "clean_frames"
    existing_frames = sorted(clean_frames.glob("*.jpg"))
    if ((directory / "clean.mp4").is_file() and (directory / "metadata.json").is_file()
            and len(existing_frames) == expected):
        return directory
    clean_frames.mkdir(parents=True, exist_ok=True)
    for old_frame in clean_frames.glob("*.jpg"):
        old_frame.unlink()
    for index, frame in enumerate(frames):
        if not cv2.imwrite(str(clean_frames / f"frame_{index:04d}.jpg"), frame):
            raise RuntimeError(f"Could not save frame {index + 1} for {candidate_id}")
    _codec_video(frames, directory / "clean.mp4", fps)
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
        "manual_selection": False,
        "negative_sampling": True,
        "sampling_method": "fixed_interval",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


class NegativeReviewStore:
    """Persistent review decisions for the separate negative-sampling app."""

    def __init__(self, state_dir: Path):
        state_dir.mkdir(parents=True, exist_ok=True)
        self.database = state_dir / "negative_sampler.sqlite3"
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS negative_reviews (
                    source_id TEXT NOT NULL, center_frame INTEGER NOT NULL,
                    candidate_id TEXT NOT NULL, decision TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    PRIMARY KEY (source_id, center_frame)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def set_decision(self, source: Path, center_frame: int, candidate_id: str, decision: str) -> None:
        if decision not in {"accepted", "rejected"}:
            raise ValueError(f"Unknown review decision: {decision}")
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO negative_reviews VALUES (?, ?, ?, ?, ?)",
                (source_id(source), int(center_frame), candidate_id, decision,
                 datetime.now(timezone.utc).isoformat()),
            )

    def decisions(self, source: Path) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT center_frame, candidate_id, decision, decided_at
                   FROM negative_reviews WHERE source_id = ? ORDER BY center_frame""",
                (source_id(source),),
            ).fetchall()
        return [
            {"center_frame": row[0], "candidate_id": row[1], "decision": row[2], "decided_at": row[3]}
            for row in rows
        ]
