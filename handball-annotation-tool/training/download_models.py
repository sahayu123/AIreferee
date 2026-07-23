from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from .config import project_path

POSE_LANDMARKER_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)


def download(url: str, destination: Path, overwrite: bool = False) -> Path:
    if destination.is_file() and not overwrite:
        print(f"Already present: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError(f"Downloaded model is unexpectedly small: {temporary.stat().st_size} bytes")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external MediaPipe model assets.")
    parser.add_argument("--output", default="models/pose_landmarker_full.task")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(POSE_LANDMARKER_FULL_URL, project_path(args.output), args.overwrite)


if __name__ == "__main__":
    main()
