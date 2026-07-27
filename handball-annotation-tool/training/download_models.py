from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from .config import project_path

POSE_LANDMARKER_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
YOLO11N_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
ROLE_DETECTOR_REPO = "gianpaj/football-players-detection-1"
ROLE_DETECTOR_FILE = "weights/best.pt"
ROLE_DETECTOR_REVISION = "cd7b76064c6122153ee0859f79e328a4b01c4d2b"


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


def download_huggingface(
    repo_id: str,
    filename: str,
    destination: Path,
    overwrite: bool = False,
    revision: str | None = None,
) -> Path:
    if destination.is_file() and not overwrite:
        print(f"Already present: {destination}")
        return destination
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Install huggingface_hub before downloading the football role detector."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    )
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        shutil.copy2(cached, temporary)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Downloaded role detector is unexpectedly small: "
                f"{temporary.stat().st_size} bytes"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {destination} ({destination.stat().st_size / 1024 / 1024:.1f} MB)")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download external YOLO and MediaPipe model assets.")
    parser.add_argument(
        "--output",
        default="models/pose_landmarker_full.task",
        help="MediaPipe Pose Landmarker output path.",
    )
    parser.add_argument("--detector-output", default="yolo11n.pt")
    parser.add_argument(
        "--with-role-detector",
        action="store_true",
        help="Also download the Hugging Face football player/goalkeeper/referee detector.",
    )
    parser.add_argument(
        "--role-detector-output",
        default="models/football_roles_yolov8x.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    download(POSE_LANDMARKER_FULL_URL, project_path(args.output), args.overwrite)
    download(YOLO11N_URL, project_path(args.detector_output), args.overwrite)
    if args.with_role_detector:
        download_huggingface(
            ROLE_DETECTOR_REPO,
            ROLE_DETECTOR_FILE,
            project_path(args.role_detector_output),
            args.overwrite,
            ROLE_DETECTOR_REVISION,
        )


if __name__ == "__main__":
    main()
