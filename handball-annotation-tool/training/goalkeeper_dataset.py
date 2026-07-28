"""Create reviewed full-player crops for supervised goalkeeper training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import project_path

IMAGE_SUFFIXES = {
    "",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".bmp",
}
LABELS = ("not_goalkeeper", "goalkeeper")


class PersonDetector(Protocol):
    def detect(self, image: np.ndarray) -> Sequence[dict[str, Any]]:
        ...


class YOLOPersonDetector:
    def __init__(
        self,
        checkpoint: Path,
        *,
        confidence: float = 0.25,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        if not checkpoint.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {checkpoint}")
        self.model = YOLO(str(checkpoint))
        self.confidence = confidence
        self.device = device

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = self.model.predict(
            source=image,
            classes=[0],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        detections: list[dict[str, Any]] = []
        if result.boxes is None:
            return detections
        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            detections.append(
                {
                    "bbox": [float(value) for value in box],
                    "confidence": float(confidence),
                }
            )
        return detections


def source_group(root: Path, image_path: Path) -> str:
    relative = image_path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    # A root-level image is conservatively its own group. Users should put
    # related images in a shared match/source subdirectory.
    return image_path.stem


def collect_source_images(
    goalkeeper_source: Path,
    not_goalkeeper_source: Path,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for label, root in (
        ("goalkeeper", goalkeeper_source),
        ("not_goalkeeper", not_goalkeeper_source),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{label} source directory not found: {root}")
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        ):
            records.append(
                {
                    "source_image": str(path.resolve()),
                    "source_label": label,
                    # Matching subdirectory names across the two class roots
                    # intentionally map to the same match/source group.
                    "source_group": source_group(root, path),
                }
            )
    if not records:
        raise ValueError("No supported source images were found")
    return records


def expanded_crop(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    margin: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR HxWx3 image")
    if len(bbox) != 4 or not 0 <= margin <= 1:
        raise ValueError("Invalid crop box or margin")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Detection box has non-positive area")
    box_width, box_height = x2 - x1, y2 - y1
    left = max(0, int(np.floor(x1 - margin * box_width)))
    right = min(width, int(np.ceil(x2 + margin * box_width)))
    top = max(0, int(np.floor(y1 - margin * box_height)))
    bottom = min(height, int(np.ceil(y2 + margin * box_height)))
    crop = image[top:bottom, left:right].copy()
    if crop.size == 0:
        raise ValueError("Detection produced an empty crop")
    return crop, (left, top, right, bottom)


def extract_player_candidates(
    sources: Sequence[dict[str, str]],
    detector: PersonDetector,
    output_root: Path,
    *,
    margin: float = 0.15,
    minimum_player_height: int = 100,
    auto_accept_single: bool = True,
) -> pd.DataFrame:
    crop_root = output_root / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source_number, source in enumerate(sources, start=1):
        source_path = Path(source["source_image"])
        image = cv2.imread(str(source_path))
        if image is None:
            rows.append(
                {
                    **source,
                    "status": "source_read_error",
                    "review_label": "",
                }
            )
            continue
        detections = list(detector.detect(image))
        usable = [
            detection
            for detection in detections
            if float(detection["bbox"][3]) - float(detection["bbox"][1])
            >= minimum_player_height
        ]
        if not usable:
            rows.append(
                {
                    **source,
                    "status": "no_usable_person",
                    "review_label": "",
                }
            )
            continue
        single = len(usable) == 1
        for detection_number, detection in enumerate(usable):
            crop, expanded_box = expanded_crop(
                image, detection["bbox"], margin=margin
            )
            identity = hashlib.sha256(
                (
                    f"{source_path.resolve()}::{detection_number}::"
                    f"{expanded_box}"
                ).encode("utf-8")
            ).hexdigest()[:16]
            relative = Path(source["source_label"]) / f"{identity}.jpg"
            destination = crop_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(destination), crop):
                raise OSError(f"Could not write player crop: {destination}")
            review_label = (
                source["source_label"]
                if auto_accept_single and single
                else ""
            )
            rows.append(
                {
                    **source,
                    "crop_path": str(destination.resolve()),
                    "crop_id": identity,
                    "detection_index": detection_number,
                    "detection_count": len(usable),
                    "detection_confidence": float(
                        detection.get("confidence", 0)
                    ),
                    "bbox": ",".join(str(value) for value in expanded_box),
                    "crop_width": int(crop.shape[1]),
                    "crop_height": int(crop.shape[0]),
                    "status": (
                        "auto_accepted_single"
                        if review_label
                        else "needs_review"
                    ),
                    "review_label": review_label,
                }
            )
        print(
            f"[{source_number}/{len(sources)}] {source_path.name}: "
            f"{len(usable)} usable player crop(s)",
            flush=True,
        )
    manifest = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_root / "candidates.csv", index=False)
    return manifest


def assign_group_folds(
    reviewed: pd.DataFrame,
    *,
    folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    required = {"crop_path", "review_label", "source_group"}
    missing = required - set(reviewed.columns)
    if missing:
        raise ValueError(f"Reviewed manifest missing columns: {sorted(missing)}")
    accepted = reviewed[
        reviewed["review_label"].astype(str).isin(LABELS)
    ].copy()
    if accepted.empty:
        raise ValueError("No reviewed goalkeeper/not_goalkeeper crops found")
    if accepted["review_label"].nunique() != 2:
        raise ValueError("Both goalkeeper classes are required")
    unique_groups = accepted["source_group"].nunique()
    if unique_groups < folds:
        raise ValueError(
            f"At least {folds} source groups are required; found {unique_groups}"
        )
    accepted["fold"] = -1
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    labels = accepted["review_label"].map(
        {"not_goalkeeper": 0, "goalkeeper": 1}
    )
    for fold, (_, validation_indices) in enumerate(
        splitter.split(accepted, labels, accepted["source_group"])
    ):
        accepted.iloc[
            validation_indices,
            accepted.columns.get_loc("fold"),
        ] = fold
    if (accepted["fold"] < 0).any():
        raise RuntimeError("Failed to assign every crop to a fold")
    incomplete_folds = [
        int(fold)
        for fold, partition in accepted.groupby("fold")
        if partition["review_label"].nunique() != 2
    ]
    if incomplete_folds:
        raise ValueError(
            "Every fold must contain both classes. Add more independent "
            f"source groups; incomplete folds: {incomplete_folds}"
        )
    return accepted.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect and crop players for goalkeeper classification."
    )
    parser.add_argument("--goalkeeper-source", required=True, type=Path)
    parser.add_argument("--not-goalkeeper-source", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        default="workspace/goalkeeper_classifier",
        type=Path,
    )
    parser.add_argument("--detector", default="yolo11n.pt", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--minimum-player-height", type=int, default=100)
    parser.add_argument(
        "--require-review-all",
        action="store_true",
        help="Do not auto-label images containing exactly one usable person.",
    )
    args = parser.parse_args()
    goalkeeper_source = project_path(args.goalkeeper_source)
    not_goalkeeper_source = project_path(args.not_goalkeeper_source)
    output_root = project_path(args.output_root)
    sources = collect_source_images(
        goalkeeper_source, not_goalkeeper_source
    )
    detector = YOLOPersonDetector(
        project_path(args.detector),
        confidence=args.confidence,
        device=args.device,
    )
    manifest = extract_player_candidates(
        sources,
        detector,
        output_root,
        margin=args.margin,
        minimum_player_height=args.minimum_player_height,
        auto_accept_single=not args.require_review_all,
    )
    print(
        f"Wrote {len(manifest)} manifest rows to "
        f"{output_root / 'candidates.csv'}"
    )


if __name__ == "__main__":
    main()
