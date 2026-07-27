from __future__ import annotations

import argparse
import hashlib
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
PRTREID_URL = (
    "https://zenodo.org/records/10653453/files/"
    "prtreid-soccernet-baseline.pth.tar?download=1"
)
PRTREID_MD5 = "9633825232bc89f23a94522c5561650e"


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


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream publishes MD5 for integrity only
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    destination: Path,
    expected_md5: str,
    overwrite: bool = False,
) -> Path:
    downloaded = download(url, destination, overwrite)
    actual_md5 = _md5(downloaded)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"Checksum mismatch for {downloaded}: expected {expected_md5}, "
            f"found {actual_md5}. Re-run with --overwrite."
        )
    print(f"Verified {downloaded} (MD5 {actual_md5})")
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify external detector, pose, and role-model assets."
    )
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
    parser.add_argument(
        "--with-prtreid",
        action="store_true",
        help="Also download the official SoccerNet PRTReID role checkpoint.",
    )
    parser.add_argument(
        "--prtreid-output",
        default="models/prtreid-soccernet-baseline.pth.tar",
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
    if args.with_prtreid:
        download_verified(
            PRTREID_URL,
            project_path(args.prtreid_output),
            PRTREID_MD5,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
