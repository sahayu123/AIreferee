from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SimilarCandidate:
    candidate_id: str
    source_name: str
    center_time_seconds: float
    similarity: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=50_000)
def _perceptual_hash(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read dataset frame: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8].reshape(-1)
    median = float(np.median(low_frequency[1:]))
    return low_frequency > median


def _frame_hashes(directory: Path) -> list[np.ndarray]:
    return [_perceptual_hash(path) for path in sorted(directory.glob("*.jpg"))]


def _sequence_similarity(first: list[np.ndarray], second: list[np.ndarray]) -> float:
    """Best aligned average perceptual-hash similarity across two clips."""
    if not first or not second:
        return 0.0
    minimum_overlap = min(8, len(first), len(second))
    best = 0.0
    for shift in range(-len(second) + minimum_overlap, len(first) - minimum_overlap + 1):
        first_start, second_start = max(shift, 0), max(-shift, 0)
        overlap = min(len(first) - first_start, len(second) - second_start)
        if overlap < minimum_overlap:
            continue
        first_array = np.asarray(first[first_start:first_start + overlap])
        second_array = np.asarray(second[second_start:second_start + overlap])
        score = 1.0 - float(np.not_equal(first_array, second_array).mean())
        best = max(best, score)
    return best


def find_handball_duplicates(
    candidate_dir: Path,
    dataset_dir: Path,
    threshold: float = 0.88,
    limit: int = 5,
) -> list[SimilarCandidate]:
    candidate_metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
    candidate_id = str(candidate_metadata["candidate_id"])
    candidate_hashes = _frame_hashes(candidate_dir / "clean_frames")
    matches: list[SimilarCandidate] = []
    for labeled_dir in sorted((dataset_dir / "handball").iterdir()):
        if not labeled_dir.is_dir() or labeled_dir.name == candidate_id:
            continue
        metadata_path, frames_dir = labeled_dir / "metadata.json", labeled_dir / "frames"
        if not metadata_path.is_file() or not frames_dir.is_dir():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        visual_similarity = _sequence_similarity(candidate_hashes, _frame_hashes(frames_dir))

        same_source = metadata.get("source_name") == candidate_metadata.get("source_name")
        first_center = int(candidate_metadata.get("center_frame", -1000000))
        second_center = int(metadata.get("center_frame", 1000000))
        center_difference = abs(first_center - second_center)
        window_length = len(candidate_hashes) or 41
        overlap_similarity = max(0.0, 1.0 - center_difference / window_length) if same_source else 0.0
        similarity = max(visual_similarity, overlap_similarity)
        if similarity >= threshold:
            reason = "overlapping frames from the same source" if overlap_similarity >= visual_similarity else "visually similar frame sequence"
            matches.append(SimilarCandidate(
                candidate_id=str(metadata["candidate_id"]),
                source_name=str(metadata["source_name"]),
                center_time_seconds=float(metadata.get("center_time_seconds", 0.0)),
                similarity=round(similarity, 3), reason=reason,
            ))
    return sorted(matches, key=lambda match: match.similarity, reverse=True)[:limit]
