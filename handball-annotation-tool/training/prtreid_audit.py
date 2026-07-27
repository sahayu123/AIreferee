from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .data import feature_metadata
from .features import FEATURE_NAMES, _contact_sheet, _safe_name, feature_path
from .logging_utils import configure_logging
from .manifest import sorted_frames
from .prtreid_role import (
    PRTReIDConfig,
    PRTReIDWorkerClient,
    YOLOPersonTracker,
    classify_actor_role,
    load_prtreid_config,
    prtreid_result_is_current,
    prtreid_source_fingerprint,
    save_prtreid_result,
)


def result_path(config: PRTReIDConfig, row: pd.Series) -> Path:
    return (
        config.roles_dir
        / str(row["domain"])
        / _safe_name(str(row["example_id"]))
        / f"{_safe_name(str(row['view_id']))}.json"
    )


def audit_path(config: PRTReIDConfig, row: pd.Series) -> Path:
    return (
        config.audits_dir
        / str(row["domain"])
        / f"{_safe_name(str(row['example_id']))}_{_safe_name(str(row['view_id']))}.jpg"
    )


def _load_inputs(
    config: PRTReIDConfig,
    row: pd.Series,
) -> tuple[np.ndarray, list[int], list[Path], dict[str, Any], Path | None, str]:
    artifact = feature_path(config.features_dir, row)
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Base handball features not found: {artifact}. "
            "Run `python -m training.features` first."
        )
    loaded = np.load(artifact, allow_pickle=False)
    features = loaded["features"].astype(np.float32)
    stored_metadata = feature_metadata(artifact)
    selected = [
        int(index)
        for index in stored_metadata.get("selected_frame_indices", [])
    ]
    names = [str(name) for name in stored_metadata.get("feature_names", [])]
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected feature shape in {artifact}: {features.shape}")
    if names != FEATURE_NAMES:
        raise ValueError(
            f"Base feature schema in {artifact} does not match the 56-feature model"
        )
    if len(selected) != len(features):
        raise ValueError(
            f"Feature metadata in {artifact} has {len(selected)} selected indices "
            f"for {len(features)} rows"
        )
    frames_dir = PROJECT_ROOT / str(row["frames_dir"])
    frames = sorted_frames(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {frames_dir}")
    metadata_path = frames_dir.parent / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid candidate metadata: {metadata_path}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Candidate metadata is not an object: {metadata_path}")
    else:
        metadata = {}
        metadata_path = None
    fingerprint = prtreid_source_fingerprint(
        artifact, frames, selected, metadata_path
    )
    return features, selected, frames, metadata, metadata_path, fingerprint


def _row_is_current(
    config: PRTReIDConfig,
    row: pd.Series,
    require_audit: bool,
) -> bool:
    destination = result_path(config, row)
    if not destination.is_file():
        return False
    try:
        *_, fingerprint = _load_inputs(config, row)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return (
        prtreid_result_is_current(destination, config, fingerprint)
        and (not require_audit or audit_path(config, row).is_file())
    )


def _evenly_spaced(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[int(index)] for index in indices]


def _actor_context_tile(
    frame: np.ndarray,
    observation: dict[str, Any],
    association_text: str,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in observation["bbox"])
    box_width, box_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    context_left = int(np.clip(x1 - 0.5 * box_width, 0, width))
    context_right = int(np.clip(x2 + 0.5 * box_width, 0, width))
    context_top = int(np.clip(y1 - 0.25 * box_height, 0, height))
    context_bottom = int(np.clip(y2 + 0.25 * box_height, 0, height))
    if context_right <= context_left or context_bottom <= context_top:
        tile = frame.copy()
        context_left = context_top = 0
    else:
        tile = frame[
            context_top:context_bottom, context_left:context_right
        ].copy()
    left = int(x1 - context_left)
    top = int(y1 - context_top)
    right = int(x2 - context_left)
    bottom = int(y2 - context_top)
    cv2.rectangle(tile, (left, top), (right, bottom), (0, 215, 255), 3)
    prediction = observation.get("worker_prediction") or {}
    role = str(prediction.get("predicted_role", "unscored"))
    probabilities = prediction.get("role_probabilities") or {}
    goalkeeper = float(probabilities.get("goalkeeper", 0.0))
    player = float(probabilities.get("player", 0.0))
    lines = (
        f"frame {observation.get('frame_index')} | {role}",
        f"GK {goalkeeper:.2f}  player {player:.2f}",
        association_text,
    )
    header_height = 70
    tile = cv2.copyMakeBorder(
        tile, header_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    for line_number, line in enumerate(lines):
        cv2.putText(
            tile,
            line,
            (8, 20 + line_number * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return tile


def save_audit_contact_sheet(
    result: dict[str, Any],
    frame_paths: Sequence[Path],
    destination: Path,
) -> None:
    association = result.get("association") or {}
    actor_track_id = result.get("actor_track_id")
    tracks = result.get("tracks") or []
    actor_track = next(
        (
            track
            for track in tracks
            if track.get("track_id") == actor_track_id
        ),
        None,
    )
    observations = (
        _evenly_spaced(actor_track.get("observations", []), 12)
        if actor_track is not None
        else []
    )
    association_text = (
        f"{association.get('method', 'none')} "
        f"score={float(association.get('score', 0.0)):.2f} "
        f"confident={bool(association.get('confident', False))}"
    )
    tiles: list[np.ndarray] = []
    for observation in observations:
        frame_index = int(observation["frame_index"])
        if frame_index < 0 or frame_index >= len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            continue
        tiles.append(_actor_context_tile(frame, observation, association_text))
    if not tiles:
        fallback_indices = (
            np.linspace(0, len(frame_paths) - 1, min(4, len(frame_paths)))
            .round()
            .astype(int)
        )
        for frame_index in fallback_indices:
            frame = cv2.imread(str(frame_paths[int(frame_index)]))
            if frame is None:
                continue
            cv2.putText(
                frame,
                f"UNKNOWN ACTOR | {association_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(frame)
    if not tiles:
        raise RuntimeError("Could not render any PRTReID audit frames")
    _contact_sheet(tiles, destination)


def _track_for_actor(result: dict[str, Any]) -> dict[str, Any] | None:
    actor_track_id = result.get("actor_track_id")
    return next(
        (
            track
            for track in result.get("tracks", [])
            if track.get("track_id") == actor_track_id
        ),
        None,
    )


def build_report(config: PRTReIDConfig, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        path = result_path(config, item)
        if not _row_is_current(config, item, require_audit=False):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        association = result.get("association") or {}
        actor_track = _track_for_actor(result)
        aggregate = actor_track.get("aggregate", {}) if actor_track else {}
        color = actor_track.get("jersey_color", {}) if actor_track else {}
        rows.append(
            {
                "example_id": str(item["example_id"]),
                "view_id": str(item["view_id"]),
                "handball_label": int(item["label"]),
                "domain": str(item["domain"]),
                "predicted_role": result["predicted_role"],
                "is_goalkeeper": result["is_goalkeeper"],
                "goalkeeper_score": result["goalkeeper_score"],
                "role_confidence": result["role_confidence"],
                "uncertain": result["uncertain"],
                "actor_track_id": result.get("actor_track_id"),
                "association_method": association.get("method"),
                "association_score": association.get("score"),
                "association_margin": association.get("margin"),
                "association_confident": association.get("confident"),
                "tracked_people": result.get("tracked_people"),
                "classified_crops": result.get("classified_crops"),
                "actor_track_frames": (
                    actor_track.get("frame_count") if actor_track else 0
                ),
                "actor_prediction_frames": aggregate.get("prediction_frames", 0),
                "jersey_outlier_score": color.get("outlier_score", 0.0),
                "result_path": str(path),
            }
        )
    report = pd.DataFrame(rows)
    config.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.report, index=False)
    return report


def summarize_report(report: pd.DataFrame) -> dict[str, Any]:
    if report.empty:
        return {
            "examples": 0,
            "goalkeepers": 0,
            "not_goalkeepers": 0,
            "uncertain": 0,
            "confident_actor_associations": 0,
            "classified_crops": 0,
        }
    decisions = report["is_goalkeeper"]
    return {
        "examples": len(report),
        "goalkeepers": int((decisions == True).sum()),  # noqa: E712
        "not_goalkeepers": int((decisions == False).sum()),  # noqa: E712
        "uncertain": int(report["uncertain"].sum()),
        "confident_actor_associations": int(
            report["association_confident"].fillna(False).sum()
        ),
        "classified_crops": int(report["classified_crops"].fillna(0).sum()),
    }


def audit_manifest(
    config: PRTReIDConfig,
    overwrite: bool = False,
    domain: str | None = None,
    example_contains: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
    save_audits: bool = True,
) -> dict[str, Any]:
    if not config.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    full_manifest = pd.read_csv(config.manifest)
    manifest = full_manifest
    if domain:
        manifest = manifest[manifest["domain"] == domain]
    if example_contains:
        manifest = manifest[
            manifest["example_id"].astype(str).str.contains(
                example_contains, case=False, regex=False
            )
        ]
    if limit is not None:
        manifest = manifest.head(limit)
    manifest = manifest.reset_index(drop=True)
    config.roles_dir.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(config.logs_dir / "prtreid_audit.log", verbose)
    pending = [
        row
        for _, row in manifest.iterrows()
        if overwrite
        or not _row_is_current(config, row, require_audit=save_audits)
    ]

    def tracking_progress(current: int, total: int) -> None:
        if current == 1 or current == total or current % 5 == 0:
            logger.info("tracking frames %d/%d", current, total)

    def role_progress(current: int, total: int) -> None:
        logger.info("PRTReID crops %d/%d", current, total)

    tracker = (
        YOLOPersonTracker(config, progress_callback=tracking_progress)
        if pending
        else None
    )
    worker = (
        PRTReIDWorkerClient(config, progress_callback=role_progress)
        if pending
        else None
    )
    try:
        for item_number, (_, row) in enumerate(manifest.iterrows(), start=1):
            destination = result_path(config, row)
            if (
                not overwrite
                and _row_is_current(config, row, require_audit=save_audits)
            ):
                logger.info(
                    "[%d/%d] cached %s %s",
                    item_number,
                    len(manifest),
                    row["example_id"],
                    row["view_id"],
                )
                continue
            logger.info(
                "[%d/%d] tracking and classifying %s %s (%s frames)",
                item_number,
                len(manifest),
                row["example_id"],
                row["view_id"],
                row["frame_count"],
            )
            features, selected, frames, metadata, _, fingerprint = _load_inputs(
                config, row
            )
            if tracker is None or worker is None:
                raise RuntimeError("PRTReID resources were not initialized")
            result = classify_actor_role(
                frames,
                features,
                selected,
                metadata,
                config,
                tracker=tracker,
                worker=worker,
            )
            result.update(
                {
                    "example_id": str(row["example_id"]),
                    "view_id": str(row["view_id"]),
                    "handball_label": int(row["label"]),
                    "domain": str(row["domain"]),
                    "source_fingerprint": fingerprint,
                }
            )
            save_prtreid_result(result, destination)
            if save_audits:
                save_audit_contact_sheet(result, frames, audit_path(config, row))
            logger.info(
                "[%d/%d] role=%s actor=%s association=%s crops=%s",
                item_number,
                len(manifest),
                result["predicted_role"],
                result["actor_track_id"],
                result["association"]["confident"],
                result["classified_crops"],
            )
    finally:
        if worker is not None:
            worker.close()
    report = build_report(config, full_manifest)
    selected_paths = {
        str(result_path(config, row)) for _, row in manifest.iterrows()
    }
    selected_report = (
        report[report["result_path"].isin(selected_paths)]
        if not report.empty
        else report
    )
    summary = summarize_report(selected_report)
    summary["cumulative_report_examples"] = len(report)
    logger.info("summary=%s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Track every player across all clip frames and audit the selected "
            "handball actor with SoccerNet PRTReID."
        )
    )
    parser.add_argument(
        "--config", default="configs/prtreid_goalkeeper.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--domain", choices=["native", "imported"])
    parser.add_argument(
        "--example-contains",
        help="Only process manifest example IDs containing this text.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--no-audits",
        action="store_true",
        help="Skip visual contact sheets while still saving JSON results.",
    )
    args = parser.parse_args()
    summary = audit_manifest(
        load_prtreid_config(args.config),
        overwrite=args.overwrite,
        domain=args.domain,
        example_contains=args.example_contains,
        limit=args.limit,
        verbose=args.verbose,
        save_audits=not args.no_audits,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
