from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from handball_annotator.runtime import get_device

from .config import load_feature_config, load_train_config, project_path
from .features import FEATURE_NAMES, FeatureExtractor, _contact_sheet
from .gru import TemporalGRU
from .manifest import sorted_frames


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


def infer(
    input_path: Path,
    checkpoint_path: Path,
    feature_config_path: str | Path,
    train_config_path: str | Path,
    output_path: Path,
    overlay_path: Path | None = None,
    threshold: float = 0.5,
) -> dict[str, object]:
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
    args = parser.parse_args()
    result = infer(
        project_path(args.input), project_path(args.checkpoint), args.feature_config, args.train_config,
        project_path(args.output), project_path(args.overlay) if args.overlay else None, args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

