from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import AppConfig

LABELS = ("handball", "not_handball", "uncertain")


class AnnotationStore:
    def __init__(self, config: AppConfig):
        self.config = config
        for directory in (config.uploads_dir, config.candidates_dir, config.dataset_dir, config.state_dir):
            directory.mkdir(parents=True, exist_ok=True)
        for label in LABELS:
            (config.dataset_dir / label).mkdir(parents=True, exist_ok=True)
        self.database = config.state_dir / "annotations.sqlite3"
        self._write_lock = threading.RLock()
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS annotations (
                candidate_id TEXT PRIMARY KEY, source_name TEXT NOT NULL, label TEXT NOT NULL,
                labeled_at TEXT NOT NULL, metadata_json TEXT NOT NULL)""")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def labels(self) -> dict[str, str]:
        with self._connect() as connection:
            return dict(connection.execute("SELECT candidate_id, label FROM annotations"))

    def label(self, candidate_dir: Path, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label}")
        with self._write_lock:
            metadata_path = candidate_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_id = str(metadata["candidate_id"])
            destination = self.config.dataset_dir / label / candidate_id
            source_frames = sorted((candidate_dir / "clean_frames").glob("*.jpg"))

            with self._connect() as connection:
                old = connection.execute(
                    "SELECT label FROM annotations WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()

            # A repeated click on the current label is a no-op when the saved
            # example is complete. This makes Streamlit reruns idempotent.
            saved_frames = sorted((destination / "frames").glob("*.jpg")) if destination.exists() else []
            complete = (
                (destination / "clip.mp4").is_file()
                and (destination / "metadata.json").is_file()
                and len(source_frames) > 0
                and len(saved_frames) == len(source_frames)
            )
            if old and old[0] == label and complete:
                return

            # Build the complete example beside its destination, then swap it
            # into place. The visible dataset folder is never half-written.
            staging = destination.parent / f".{candidate_id}.{uuid.uuid4().hex}.tmp"
            backup = destination.parent / f".{candidate_id}.{uuid.uuid4().hex}.backup"
            try:
                staging.mkdir(parents=True)
                shutil.copy2(candidate_dir / "clean.mp4", staging / "clip.mp4")
                shutil.copytree(candidate_dir / "clean_frames", staging / "frames")
                shutil.copy2(metadata_path, staging / "metadata.json")
                if destination.exists():
                    destination.replace(backup)
                staging.replace(destination)

                with self._connect() as connection:
                    connection.execute(
                        "INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?)",
                        (candidate_id, str(metadata["source_name"]), label,
                         datetime.now(timezone.utc).isoformat(), json.dumps(metadata)),
                    )
            except Exception:
                if not destination.exists() and backup.exists():
                    backup.replace(destination)
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)

            if old and old[0] != label:
                previous = self.config.dataset_dir / old[0] / candidate_id
                if previous.exists():
                    shutil.rmtree(previous, ignore_errors=True)

    def export_jsonl(self) -> Path:
        output = self.config.dataset_dir / "annotations.jsonl"
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT candidate_id, source_name, label, labeled_at, metadata_json FROM annotations ORDER BY labeled_at"
            ).fetchall()
        lines = []
        for candidate_id, source, label, timestamp, metadata_json in rows:
            lines.append(json.dumps({"candidate_id": candidate_id, "source_name": source, "label": label,
                                     "labeled_at": timestamp, "candidate": json.loads(metadata_json)}))
        output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return output

    def unlabel(self, candidate_id: str) -> None:
        """Remove a label and its copied dataset artifacts, preserving review candidates."""
        with self._write_lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT label FROM annotations WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if not row:
                    return
                connection.execute("DELETE FROM annotations WHERE candidate_id = ?", (candidate_id,))
            destination = self.config.dataset_dir / str(row[0]) / candidate_id
            if destination.exists():
                shutil.rmtree(destination)
