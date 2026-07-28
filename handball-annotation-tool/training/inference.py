from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from handball_annotator.runtime import get_device

from .config import load_feature_config, load_train_config, project_path
from .features import FEATURE_NAMES, FeatureExtractor, _contact_sheet
from .gru import TemporalGRU
from .jersey_glove_role import (
    classify_goalkeeper_after_handball,
    load_jersey_glove_config,
)
from .manifest import sorted_frames
from .prtreid_role import classify_actor_role, load_prtreid_config
from .role_detector import (
    FootballRoleDetector,
    classify_selected_actor,
    load_role_config,
)
from .supervised_goalkeeper import (
    classify_supervised_goalkeeper,
    load_supervised_goalkeeper_config,
)

_GOALKEEPER_STATUSES = {
    "goalkeeper",
    "not_goalkeeper",
    "unknown",
    "unavailable",
    "not_evaluated",
}


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


def _add_jersey_glove_overlays(
    overlays: list[np.ndarray],
    selected_indices: list[int],
    goalkeeper_result: dict[str, object],
) -> list[np.ndarray]:
    observations = {
        int(item["frame_index"]): item
        for item in goalkeeper_result.get("actor_observations", [])
        if isinstance(item, dict) and "frame_index" in item
    }
    status = str(goalkeeper_result.get("status", "unknown"))
    evidence_score = goalkeeper_result.get("goalkeeper_evidence_score")
    score_text = (
        f"{float(evidence_score):.2f}"
        if evidence_score is not None
        else "n/a"
    )
    updated: list[np.ndarray] = []
    for overlay, frame_index in zip(overlays, selected_indices):
        rendered = overlay.copy()
        observation = observations.get(int(frame_index))
        if observation is not None:
            x1, y1, x2, y2 = (
                int(float(value)) for value in observation["bbox"]
            )
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 165, 0), 3)
        cv2.putText(
            rendered,
            f"GK evidence: {status} score={score_text}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 165, 0),
            2,
            cv2.LINE_AA,
        )
        updated.append(rendered)
    return updated


def _sanitize_goalkeeper_result(
    value: object,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("Goalkeeper analysis must return an object")
    result = dict(value)
    status = result.get("status")
    if status not in _GOALKEEPER_STATUSES:
        raise ValueError(f"Invalid goalkeeper status: {status!r}")
    is_goalkeeper = result.get("is_goalkeeper")
    if status == "goalkeeper" and is_goalkeeper is not True:
        raise ValueError(
            "A goalkeeper status requires is_goalkeeper=true"
        )
    if status == "not_goalkeeper" and is_goalkeeper is not False:
        raise ValueError(
            "A not_goalkeeper status requires is_goalkeeper=false"
        )
    if status not in {"goalkeeper", "not_goalkeeper"}:
        result["is_goalkeeper"] = None
    actor_observations = result.get("actor_observations", [])
    if not isinstance(actor_observations, list):
        raise TypeError("actor_observations must be a list")
    result["actor_observations"] = actor_observations
    result.setdefault("evaluated", status not in {"unavailable", "not_evaluated"})
    result.setdefault("goalkeeper_evidence_score", None)
    result.setdefault("reason", None)
    return result


def _skipped_goalkeeper_result() -> dict[str, object]:
    return {
        "evaluated": False,
        "status": "not_evaluated",
        "is_goalkeeper": None,
        "goalkeeper_evidence_score": None,
        "reason": "raw_not_handball_gate",
        "actor_observations": [],
    }


def _apply_goalkeeper_decision(
    result: dict[str, object],
    raw_predicted_label: str,
    goalkeeper_result: dict[str, object],
    *,
    backend: str,
    invoked: bool,
) -> None:
    """Publish both the hierarchical event and final binary foul decision."""
    goalkeeper_status = str(goalkeeper_result["status"])
    raw_handball = raw_predicted_label == "handball"
    goalkeeper_veto = raw_handball and goalkeeper_status == "goalkeeper"
    result["predicted_label"] = (
        "not_handball" if goalkeeper_veto else raw_predicted_label
    )
    if not raw_handball:
        final_reason = "raw_not_handball_goalkeeper_skipped"
        combined_event_label = "not_handball"
        actor_role = None
    elif goalkeeper_status == "goalkeeper":
        final_reason = "confirmed_goalkeeper_veto"
        combined_event_label = "handball_goalkeeper"
        actor_role = "goalkeeper"
    elif goalkeeper_status == "not_goalkeeper":
        final_reason = "raw_handball_confirmed_outfield"
        combined_event_label = "handball_outfield"
        actor_role = "outfield"
    else:
        final_reason = (
            f"raw_handball_preserved_{goalkeeper_status}"
        )
        combined_event_label = "handball_actor_unknown"
        actor_role = "unknown"
    result.update(
        {
            "raw_handball_detected": raw_handball,
            "combined_event_label": combined_event_label,
            "handball_actor_role": actor_role,
            "goalkeeper_status": goalkeeper_status,
            "is_goalkeeper": goalkeeper_result["is_goalkeeper"],
            "goalkeeper_evidence_score": goalkeeper_result[
                "goalkeeper_evidence_score"
            ],
            "goalkeeper_analysis_required": raw_handball,
            "goalkeeper_analysis_invoked": bool(invoked),
            "goalkeeper_analysis_completed": bool(
                goalkeeper_result.get("evaluated", False)
            ),
            "goalkeeper_veto_applied": goalkeeper_veto,
            "handball_suppressed_by_goalkeeper": goalkeeper_veto,
            "final_decision_reason": final_reason,
            "goalkeeper_detection_backend": backend,
            "goalkeeper_analysis": goalkeeper_result,
        }
    )


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
    jersey_glove_config_path: str | Path | None = None,
    supervised_goalkeeper_config_path: str | Path | None = None,
) -> dict[str, object]:
    role_options = (
        role_config_path,
        prtreid_config_path,
        jersey_glove_config_path,
        supervised_goalkeeper_config_path,
    )
    if sum(value is not None for value in role_options) > 1:
        raise ValueError(
            "Choose only one goalkeeper/role configuration."
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
    raw_predicted_label = (
        "handball" if probability >= threshold else "not_handball"
    )
    result: dict[str, object] = {
        "input": str(input_path),
        "checkpoint": str(checkpoint_path),
        "handball_probability": probability,
        "raw_handball_probability": probability,
        "raw_predicted_label": raw_predicted_label,
        "predicted_label": raw_predicted_label,
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
    if jersey_glove_config_path is not None:
        invoked = raw_predicted_label == "handball"
        if not invoked:
            goalkeeper_result = _skipped_goalkeeper_result()
        else:
            try:
                jersey_glove_config = load_jersey_glove_config(
                    jersey_glove_config_path
                )
                goalkeeper_result = _sanitize_goalkeeper_result(
                    classify_goalkeeper_after_handball(
                        frames,
                        features,
                        selected,
                        _candidate_metadata(input_path, frames),
                        probability,
                        threshold,
                        jersey_glove_config,
                        force_evaluation=True,
                    )
                )
                overlays = _add_jersey_glove_overlays(
                    overlays, selected, goalkeeper_result
                )
            except Exception as exc:
                warnings.warn(
                    (
                        "Goalkeeper analysis is unavailable; the raw "
                        f"handball result is preserved: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                goalkeeper_result = {
                    "evaluated": False,
                    "status": "unavailable",
                    "is_goalkeeper": None,
                    "goalkeeper_evidence_score": None,
                    "reason": "goalkeeper_analysis_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "actor_observations": [],
                }
        _apply_goalkeeper_decision(
            result,
            raw_predicted_label,
            goalkeeper_result,
            backend="jersey_glove_actor_track",
            invoked=invoked,
        )
    if supervised_goalkeeper_config_path is not None:
        invoked = raw_predicted_label == "handball"
        if not invoked:
            goalkeeper_result = _skipped_goalkeeper_result()
        else:
            try:
                supervised_config = load_supervised_goalkeeper_config(
                    supervised_goalkeeper_config_path
                )
                goalkeeper_result = _sanitize_goalkeeper_result(
                    classify_supervised_goalkeeper(
                        frames,
                        features,
                        selected,
                        _candidate_metadata(input_path, frames),
                        supervised_config,
                    )
                )
                overlays = _add_jersey_glove_overlays(
                    overlays, selected, goalkeeper_result
                )
            except Exception as exc:
                warnings.warn(
                    (
                        "Supervised goalkeeper analysis is unavailable; the "
                        f"raw handball result is preserved: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                goalkeeper_result = {
                    "evaluated": False,
                    "status": "unavailable",
                    "is_goalkeeper": None,
                    "goalkeeper_evidence_score": None,
                    "reason": "supervised_goalkeeper_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "actor_observations": [],
                }
        _apply_goalkeeper_decision(
            result,
            raw_predicted_label,
            goalkeeper_result,
            backend="supervised_player_crop_actor_track",
            invoked=invoked,
        )
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
    role_group.add_argument(
        "--jersey-glove-config",
        help=(
            "Optional pre-decision actor goalkeeper evidence using team "
            "jerseys and wrist-localized gloves; a confirmed goalkeeper "
            "vetoes a raw handball prediction."
        ),
    )
    role_group.add_argument(
        "--supervised-goalkeeper-config",
        help=(
            "Pre-decision actor tracking plus the trained full-player "
            "goalkeeper image classifier."
        ),
    )
    role_group.add_argument(
        "--no-goalkeeper-filter",
        action="store_true",
        help=(
            "Return the raw GRU decision without running a goalkeeper filter."
        ),
    )
    args = parser.parse_args()
    if not any(
        (
            args.role_config,
            args.prtreid_config,
            args.jersey_glove_config,
            args.supervised_goalkeeper_config,
            args.no_goalkeeper_filter,
        )
    ):
        supervised_checkpoint = project_path(
            "artifacts/checkpoints/goalkeeper_mobilenet_v3_best.pt"
        )
        if supervised_checkpoint.is_file():
            args.supervised_goalkeeper_config = (
                "configs/supervised_goalkeeper.yaml"
            )
        else:
            args.jersey_glove_config = (
                "configs/jersey_glove_goalkeeper.yaml"
            )
    result = infer(
        project_path(args.input), project_path(args.checkpoint), args.feature_config, args.train_config,
        project_path(args.output), project_path(args.overlay) if args.overlay else None, args.threshold,
        args.role_config, args.prtreid_config, args.jersey_glove_config,
        args.supervised_goalkeeper_config,
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
    goalkeeper_details = console_result.get("goalkeeper_analysis")
    if isinstance(goalkeeper_details, dict):
        console_result["goalkeeper_analysis"] = {
            key: goalkeeper_details.get(key)
            for key in (
                "evaluated",
                "status",
                "is_goalkeeper",
                "goalkeeper_evidence_score",
                "reason",
                "actor_track_id",
                "association",
                "jersey",
                "prtreid",
                "glove",
            )
        }
        console_result["full_goalkeeper_details_path"] = str(
            project_path(args.output)
        )
    print(json.dumps(console_result, indent=2))


if __name__ == "__main__":
    main()
