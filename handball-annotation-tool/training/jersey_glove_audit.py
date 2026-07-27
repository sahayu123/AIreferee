from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd

from .features import _contact_sheet, _safe_name
from .glove_classifier import GloveClassifier
from .jersey_glove_role import (
    JerseyGloveConfig,
    MediaPipeActorHandExtractor,
    classify_goalkeeper_after_handball,
    jersey_glove_result_is_current,
    load_jersey_glove_config,
    save_jersey_glove_result,
)
from .logging_utils import configure_logging
from .prtreid_audit import _load_inputs
from .prtreid_role import PRTReIDWorkerClient, YOLOPersonTracker


def result_path(config: JerseyGloveConfig, row: pd.Series) -> Path:
    return (
        config.roles_dir
        / str(row["domain"])
        / _safe_name(str(row["example_id"]))
        / f"{_safe_name(str(row['view_id']))}.json"
    )


def audit_path(config: JerseyGloveConfig, row: pd.Series) -> Path:
    return (
        config.audits_dir
        / str(row["domain"])
        / (
            f"{_safe_name(str(row['example_id']))}_"
            f"{_safe_name(str(row['view_id']))}.jpg"
        )
    )


def hand_crop_path(
    config: JerseyGloveConfig,
    row: pd.Series,
    config_fingerprint: str,
) -> Path:
    return (
        config.hand_crops_dir
        / str(row["domain"])
        / _safe_name(str(row["example_id"]))
        / _safe_name(str(row["view_id"]))
        / config_fingerprint[:12]
    )


def _row_is_current(
    config: JerseyGloveConfig,
    row: pd.Series,
    require_audit: bool,
) -> bool:
    destination = result_path(config, row)
    if not destination.is_file():
        return False
    try:
        *_, fingerprint = _load_inputs(config.base_config, row)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return (
        jersey_glove_result_is_current(destination, config, fingerprint)
        and (not require_audit or audit_path(config, row).is_file())
    )


def _evenly_spaced(items: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return list(items)
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[int(index)] for index in indices]


def _score_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}" if np.isfinite(number) else "n/a"


def _context_tile(
    frame: np.ndarray,
    observation: dict[str, Any],
    hand_boxes: Sequence[dict[str, Any]],
    result: dict[str, Any],
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in observation["bbox"])
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    left = max(0, int(x1 - 0.5 * box_width))
    right = min(width, int(x2 + 0.5 * box_width))
    top = max(0, int(y1 - 0.25 * box_height))
    bottom = min(height, int(y2 + 0.25 * box_height))
    tile = frame[top:bottom, left:right].copy()
    if tile.size == 0:
        tile = frame.copy()
        left = top = 0
    cv2.rectangle(
        tile,
        (int(x1 - left), int(y1 - top)),
        (int(x2 - left), int(y2 - top)),
        (0, 215, 255),
        3,
    )
    for hand in hand_boxes:
        hx1, hy1, hx2, hy2 = (int(value) for value in hand["bbox"])
        cv2.rectangle(
            tile,
            (hx1 - left, hy1 - top),
            (hx2 - left, hy2 - top),
            (255, 120, 0),
            2,
        )
    association = result.get("association") or {}
    jersey = result.get("jersey") or {}
    prtreid = result.get("prtreid") or {}
    player_score = (prtreid.get("scores") or {}).get("player")
    glove = result.get("glove") or {}
    lines = (
        (
            f"frame {observation.get('frame_index')} | "
            f"{result.get('status', 'unknown')}"
        ),
        (
            f"jersey outlier={_score_text(jersey.get('outlier_score'))} "
            f"glove={_score_text(glove.get('glove_probability'))} "
            f"player={_score_text(player_score)}"
        ),
        (
            f"actor link={bool(association.get('confident', False))} "
            f"score={_score_text(association.get('score'))}"
        ),
    )
    tile = cv2.copyMakeBorder(
        tile, 72, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    for line_number, line in enumerate(lines):
        cv2.putText(
            tile,
            line,
            (8, 20 + 22 * line_number),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return tile


def _hand_tile(
    frame: np.ndarray,
    hand: dict[str, Any],
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in hand["bbox"])
    x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
    y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_NEAREST)
    crop = cv2.copyMakeBorder(
        crop, 54, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    lines = (
        (
            f"frame {hand.get('frame_index')} {hand.get('side')} | "
            f"glove={_score_text(hand.get('glove_probability'))}"
        ),
        (
            f"quality={_score_text(hand.get('quality'))} "
            f"blur={_score_text(hand.get('blur_variance'))}"
        ),
    )
    for line_number, line in enumerate(lines):
        cv2.putText(
            crop,
            line,
            (6, 18 + 22 * line_number),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return crop


def save_audit_contact_sheet(
    result: dict[str, Any],
    frame_paths: Sequence[Path],
    destination: Path,
) -> None:
    observations = [
        item
        for item in result.get("actor_observations", [])
        if isinstance(item, dict)
    ]
    hands = [
        item
        for item in (result.get("glove") or {}).get("crops", [])
        if isinstance(item, dict)
    ]
    hands_by_frame: dict[int, list[dict[str, Any]]] = {}
    for hand in hands:
        hands_by_frame.setdefault(int(hand["frame_index"]), []).append(hand)
    hand_frame_ids = list(dict.fromkeys(int(item["frame_index"]) for item in hands))
    hand_frame_set = set(hand_frame_ids)
    selected_observations = [
        item
        for item in observations
        if int(item["frame_index"]) in hand_frame_set
    ]
    if len(selected_observations) < 8:
        selected_ids = {
            int(item["frame_index"]) for item in selected_observations
        }
        for item in _evenly_spaced(observations, 8):
            if int(item["frame_index"]) not in selected_ids:
                selected_observations.append(item)
                selected_ids.add(int(item["frame_index"]))
            if len(selected_observations) >= 8:
                break
    tiles: list[np.ndarray] = []
    for observation in selected_observations[:8]:
        frame_index = int(observation["frame_index"])
        if not 0 <= frame_index < len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is not None:
            tiles.append(
                _context_tile(
                    frame,
                    observation,
                    hands_by_frame.get(frame_index, []),
                    result,
                )
            )
    for hand in hands[:8]:
        frame_index = int(hand["frame_index"])
        if not 0 <= frame_index < len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            continue
        tile = _hand_tile(frame, hand)
        if tile is not None:
            tiles.append(tile)
    if not tiles:
        raise RuntimeError("Could not render a jersey/glove audit contact sheet")
    _contact_sheet(tiles, destination)


def save_hand_crops(
    result: dict[str, Any],
    frame_paths: Sequence[Path],
    config: JerseyGloveConfig,
    row: pd.Series,
) -> list[str]:
    fingerprint = str(result["config_fingerprint"])
    destination = hand_crop_path(config, row, fingerprint)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    hands = [
        item
        for item in (result.get("glove") or {}).get("crops", [])
        if isinstance(item, dict)
    ]
    for number, hand in enumerate(hands, start=1):
        frame_index = int(hand["frame_index"])
        if not 0 <= frame_index < len(frame_paths):
            continue
        frame = cv2.imread(str(frame_paths[frame_index]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in hand["bbox"])
        x1, x2 = int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))
        y1, y2 = int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        filename = (
            f"{number:02d}_frame_{frame_index:04d}_"
            f"{_safe_name(str(hand.get('side', 'unknown')))}.jpg"
        )
        path = destination / filename
        if not cv2.imwrite(str(path), crop):
            raise RuntimeError(f"Could not save hand crop: {path}")
        saved.append(str(path))
    metadata = {
        "schema_version": 1,
        "config_fingerprint": fingerprint,
        "example_id": str(row["example_id"]),
        "view_id": str(row["view_id"]),
        "files": saved,
        "crops": hands,
    }
    metadata_path = destination / "metadata.json"
    temporary = destination / "metadata.temporary.json"
    temporary.write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(metadata_path)
    return saved


def build_report(
    config: JerseyGloveConfig,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        path = result_path(config, item)
        if not _row_is_current(config, item, require_audit=False):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        association = result.get("association") or {}
        jersey = result.get("jersey") or {}
        prtreid = result.get("prtreid") or {}
        glove = result.get("glove") or {}
        rows.append(
            {
                "example_id": str(item["example_id"]),
                "view_id": str(item["view_id"]),
                "domain": str(item["domain"]),
                "handball_label": int(item["label"]),
                "goalkeeper_status": result.get("status"),
                "is_goalkeeper": result.get("is_goalkeeper"),
                "goalkeeper_evidence_score": result.get(
                    "goalkeeper_evidence_score"
                ),
                "reason": result.get("reason"),
                "actor_track_id": result.get("actor_track_id"),
                "association_confident": association.get("confident"),
                "association_score": association.get("score"),
                "actor_track_frames": result.get("actor_track_frames"),
                "jersey_team_match": jersey.get("team_match_score"),
                "jersey_outlier": jersey.get("outlier_score"),
                "excluded_actor_fragment_tracks": json.dumps(
                    jersey.get("excluded_actor_fragment_tracks", [])
                ),
                "prtreid_player": (prtreid.get("scores") or {}).get("player"),
                "glove_probability": glove.get("glove_probability"),
                "valid_hand_crops": glove.get("valid_crops"),
                "hand_crop_frames": glove.get("distinct_frames"),
                "saved_hand_crops": len(
                    result.get("hand_crop_files", [])
                ),
                "result_path": str(path),
                "audit_path": str(audit_path(config, item)),
            }
        )
    report = pd.DataFrame(rows)
    config.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.report, index=False)
    return report


def summarize_report(report: pd.DataFrame) -> dict[str, int]:
    if report.empty:
        return {
            "examples": 0,
            "goalkeepers": 0,
            "not_goalkeepers": 0,
            "unknown": 0,
            "confident_actor_associations": 0,
            "with_jersey_evidence": 0,
            "with_glove_evidence": 0,
        }
    statuses = report["goalkeeper_status"]
    return {
        "examples": len(report),
        "goalkeepers": int((statuses == "goalkeeper").sum()),
        "not_goalkeepers": int((statuses == "not_goalkeeper").sum()),
        "unknown": int((statuses == "unknown").sum()),
        "confident_actor_associations": int(
            report["association_confident"].fillna(False).sum()
        ),
        "with_jersey_evidence": int(report["jersey_outlier"].notna().sum()),
        "with_glove_evidence": int(
            report["glove_probability"].notna().sum()
        ),
    }


def audit_manifest(
    config: JerseyGloveConfig,
    *,
    overwrite: bool = False,
    domain: str | None = None,
    example_contains: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
    save_audits: bool = True,
) -> dict[str, Any]:
    manifest_path = config.base_config.manifest
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    full_manifest = pd.read_csv(manifest_path)
    handball_manifest = full_manifest[full_manifest["label"] == 1]
    manifest = handball_manifest
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
    logger = configure_logging(
        config.logs_dir / "jersey_glove_audit.log", verbose
    )
    pending = [
        row
        for _, row in manifest.iterrows()
        if overwrite
        or not _row_is_current(config, row, require_audit=save_audits)
    ]

    def tracking_progress(current: int, total: int) -> None:
        logger.info("tracking frame %d/%d", current, total)

    def role_progress(current: int, total: int) -> None:
        logger.info("PRTReID actor crop %d/%d", current, total)

    tracker: YOLOPersonTracker | None = None
    worker: PRTReIDWorkerClient | None = None
    hand_extractor: MediaPipeActorHandExtractor | None = None
    glove_model: GloveClassifier | None = None
    try:
        if pending:
            tracker = YOLOPersonTracker(
                config.base_config, progress_callback=tracking_progress
            )
            if config.use_prtreid_evidence:
                worker = PRTReIDWorkerClient(
                    config.base_config, progress_callback=role_progress
                )
            hand_extractor = MediaPipeActorHandExtractor(config)
            if config.glove_enabled:
                glove_model = GloveClassifier(
                    config.glove_checkpoint,
                    device=config.glove_device,
                    batch_size=config.glove_batch_size,
                )
        for number, (_, row) in enumerate(manifest.iterrows(), start=1):
            if (
                not overwrite
                and _row_is_current(
                    config, row, require_audit=save_audits
                )
            ):
                logger.info(
                    "[%d/%d] cached %s %s",
                    number,
                    len(manifest),
                    row["example_id"],
                    row["view_id"],
                )
                continue
            logger.info(
                "[%d/%d] %s %s (%s frames)",
                number,
                len(manifest),
                row["example_id"],
                row["view_id"],
                row["frame_count"],
            )
            features, selected, frames, metadata, _, fingerprint = (
                _load_inputs(config.base_config, row)
            )
            if tracker is None or hand_extractor is None:
                raise RuntimeError("Audit resources were not initialized")
            result = classify_goalkeeper_after_handball(
                frames,
                features,
                selected,
                metadata,
                1.0,
                0.5,
                config,
                tracker=tracker,
                role_worker=worker,
                hand_extractor=hand_extractor,
                glove_model=glove_model,
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
            result["hand_crop_files"] = save_hand_crops(
                result, frames, config, row
            )
            destination = result_path(config, row)
            save_jersey_glove_result(result, destination)
            if save_audits:
                save_audit_contact_sheet(
                    result, frames, audit_path(config, row)
                )
            logger.info(
                (
                    "[%d/%d] status=%s actor=%s link=%s "
                    "jersey=%s glove=%s"
                ),
                number,
                len(manifest),
                result["status"],
                result.get("actor_track_id"),
                (result.get("association") or {}).get("confident"),
                _score_text(
                    (result.get("jersey") or {}).get("outlier_score")
                ),
                _score_text(
                    (result.get("glove") or {}).get("glove_probability")
                ),
            )
    finally:
        if hand_extractor is not None:
            hand_extractor.close()
        if worker is not None:
            worker.close()
    report = build_report(config, handball_manifest)
    selected_paths = {
        str(result_path(config, row)) for _, row in manifest.iterrows()
    }
    selected_report = (
        report[report["result_path"].isin(selected_paths)]
        if not report.empty
        else report
    )
    summary: dict[str, Any] = summarize_report(selected_report)
    summary["cumulative_report_examples"] = len(report)
    logger.info("summary=%s", json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit jersey and wrist-localized glove evidence for handball actors."
        )
    )
    parser.add_argument(
        "--config", default="configs/jersey_glove_goalkeeper.yaml"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--domain", choices=["native", "imported"])
    parser.add_argument("--example-contains")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-audits", action="store_true")
    args = parser.parse_args()
    summary = audit_manifest(
        load_jersey_glove_config(args.config),
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
