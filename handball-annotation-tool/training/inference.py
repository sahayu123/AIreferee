from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from handball_annotator.runtime import get_device

from .config import load_feature_config, load_train_config, project_path
from .features import FEATURE_NAMES, FeatureExtractor, _contact_sheet
from .gru import TemporalGRU
from .manifest import sorted_frames
from .prtreid_role import classify_actor_role, load_prtreid_config
from .role_detector import (
    FootballRoleDetector,
    classify_selected_actor,
    load_role_config,
)


def resolve_frames(input_path: Path) -> list[Path]:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Candidate directory not found: {input_path}")
    for child in ("frames", "clean_frames"):
        directory = input_path / child
        if directory.is_dir():
            frames = sorted_frames(directory)
            if frames:
                return frames
    direct = sorted_frames(input_path)
    if direct:
        return direct
    raise FileNotFoundError(f"No JPG frames found in {input_path}, frames/, or clean_frames/")


def _candidate_metadata(input_path: Path, frames: list[Path]) -> dict[str, object]:
    candidates = [input_path / "metadata.json"]
    if frames:
        candidates.append(frames[0].parent.parent / "metadata.json")
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid candidate metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Candidate metadata is not an object: {metadata_path}")
    return metadata


def _add_prtreid_overlays(
    overlays: list[np.ndarray],
    selected_indices: list[int],
    role_result: dict[str, object],
) -> list[np.ndarray]:
    actor_track_id = role_result.get("actor_track_id")
    tracks = role_result.get("tracks", [])
    actor_track = next(
        (
            track
            for track in tracks
            if isinstance(track, dict) and track.get("track_id") == actor_track_id
        ),
        None,
    )
    observations = {
        int(item["frame_index"]): item
        for item in actor_track.get("observations", [])
    } if actor_track else {}
    role = str(role_result.get("predicted_role", "unknown"))
    score = float(role_result.get("goalkeeper_score", 0.0))
    association = role_result.get("association", {})
    confident = (
        bool(association.get("confident", False))
        if isinstance(association, dict)
        else False
    )
    updated: list[np.ndarray] = []
    for overlay, frame_index in zip(overlays, selected_indices):
        rendered = overlay.copy()
        observation = observations.get(int(frame_index))
        if observation is not None:
            x1, y1, x2, y2 = (
                int(float(value)) for value in observation["bbox"]
            )
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 215, 255), 3)
        cv2.putText(
            rendered,
            f"PRTReID: {role} GK={score:.2f} link={confident}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )
        updated.append(rendered)
    return updated


def infer(
    input_path: Path,
    checkpoint_path: Path,
    feature_config_path: str | Path,
    train_config_path: str | Path,
    output_path: Path,
    overlay_path: Path | None = None,
    threshold: float = 0.5,
    role_config_path: str | Path | None = None,
    prtreid_config_path: str | Path | None = None,
) -> dict[str, object]:
    if role_config_path is not None and prtreid_config_path is not None:
        raise ValueError(
            "Choose either --role-config or --prtreid-config, not both."
        )
    feature_config = load_feature_config(feature_config_path)
    train_config = load_train_config(train_config_path)
    device = get_device(train_config.device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if list(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature schema does not match the current extractor")
    model = TemporalGRU(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    frames = resolve_frames(input_path)
    with FeatureExtractor(feature_config) as extractor:
        features, overlays, selected = extractor.extract(frames)
    normalized = (features - checkpoint["mean"]) / np.maximum(checkpoint["std"], 1e-6)
    with torch.no_grad():
        tensor = torch.from_numpy(normalized[None].astype(np.float32)).to(device)
        probability = float(torch.sigmoid(model(tensor))[0].cpu())
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    ball_rate = float(features[:, index["ball_valid"]].mean())
    player_rate = float(features[:, index["player_valid"]].mean())
    pose_rate = float(features[:, index["pose_valid_fraction"]].mean())
    low_confidence = ball_rate < 0.25 or player_rate < 0.5 or pose_rate < 0.35
    valid_distances = features[:, index["arm_min_distance"]]
    valid_distances = valid_distances[valid_distances > 0]
    result: dict[str, object] = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "handball_probability": probability,
        "predicted_label": "handball" if probability >= threshold else "not_handball",
        "threshold": threshold,
        "selected_frame_indices": selected,
        "ball_detection_rate": ball_rate,
        "player_detection_rate": player_rate,
        "pose_valid_rate": pose_rate,
        "minimum_normalized_arm_distance": float(valid_distances.min()) if len(valid_distances) else None,
        "low_confidence_warning": low_confidence,
    }
    if role_config_path is not None:
        role_config = load_role_config(role_config_path)
        role_detector = FootballRoleDetector(role_config)
        role_result, overlays = classify_selected_actor(
            role_detector,
            frames,
            features,
            selected,
            role_config,
            base_overlays=overlays,
        )
        result.update({
            "actor_role": role_result["predicted_role"],
            "is_goalkeeper": role_result["is_goalkeeper"],
            "goalkeeper_score": role_result["goalkeeper_score"],
            "role_confidence": role_result["role_confidence"],
            "role_frame_coverage": role_result["coverage"],
            "role_detection": role_result,
        })
    if prtreid_config_path is not None:
        prtreid_config = load_prtreid_config(prtreid_config_path)
        role_result = classify_actor_role(
            frames,
            features,
            selected,
            _candidate_metadata(input_path, frames),
            prtreid_config,
        )
        overlays = _add_prtreid_overlays(overlays, selected, role_result)
        result.update({
            "actor_role": role_result["predicted_role"],
            "is_goalkeeper": role_result["is_goalkeeper"],
            "goalkeeper_score": role_result["goalkeeper_score"],
            "role_confidence": role_result["role_confidence"],
            "role_detection_backend": "soccernet_prtreid_full_track",
            "role_detection": role_result,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if overlay_path is not None:
        _contact_sheet(overlays, overlay_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one candidate with YOLO + MediaPipe + GRU.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--feature-config", default="configs/mediapipe_features.yaml")
    parser.add_argument("--train-config", default="configs/temporal_classifier.yaml")
    parser.add_argument("--output", default="outputs/mediapipe_prediction.json", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    role_group = parser.add_mutually_exclusive_group()
    role_group.add_argument(
        "--role-config",
        help="Optional legacy 12-frame goalkeeper/player/referee configuration.",
    )
    role_group.add_argument(
        "--prtreid-config",
        help="Optional all-frame SoccerNet PRTReID role configuration.",
    )
    args = parser.parse_args()
    result = infer(
        project_path(args.input), project_path(args.checkpoint), args.feature_config, args.train_config,
        project_path(args.output), project_path(args.overlay) if args.overlay else None, args.threshold,
        args.role_config, args.prtreid_config,
    )
    console_result = dict(result)
    role_details = console_result.get("role_detection")
    if isinstance(role_details, dict):
        console_result["role_detection"] = {
            key: role_details.get(key)
            for key in (
                "frame_count",
                "tracked_people",
                "classified_crops",
                "actor_track_id",
                "predicted_role",
                "is_goalkeeper",
                "goalkeeper_score",
                "role_confidence",
                "association",
            )
        }
        console_result["full_role_details_path"] = str(project_path(args.output))
    print(json.dumps(console_result, indent=2))


if __name__ == "__main__":
    main()
