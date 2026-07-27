from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .data import feature_metadata
from .features import FEATURE_NAMES, _contact_sheet, _safe_name, feature_path
from .logging_utils import configure_logging
from .manifest import sorted_frames
from .role_detector import (
    FootballRoleDetector,
    RoleConfig,
    classify_selected_actor,
    load_role_config,
    role_result_is_current,
    role_source_fingerprint,
    save_role_result,
)


def role_path(config: RoleConfig, row: pd.Series) -> Path:
    return (
        config.roles_dir
        / str(row["domain"])
        / _safe_name(str(row["example_id"]))
        / f"{_safe_name(str(row['view_id']))}.json"
    )


def audit_path(config: RoleConfig, row: pd.Series) -> Path:
    return (
        config.audits_dir
        / str(row["domain"])
        / f"{_safe_name(str(row['example_id']))}_{_safe_name(str(row['view_id']))}.jpg"
    )


def _load_selected_features(
    config: RoleConfig,
    row: pd.Series,
) -> tuple[np.ndarray, list[int]]:
    path = feature_path(config.features_dir, row)
    if not path.is_file():
        raise FileNotFoundError(
            f"Base handball features not found: {path}. "
            "Run `python -m training.features` first."
        )
    loaded = np.load(path, allow_pickle=False)
    features = loaded["features"].astype(np.float32)
    metadata = feature_metadata(path)
    selected = [int(index) for index in metadata.get("selected_frame_indices", [])]
    feature_names = [str(name) for name in metadata.get("feature_names", [])]
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected feature shape in {path}: {features.shape}")
    if feature_names != FEATURE_NAMES:
        raise ValueError(
            f"Base feature schema in {path} does not match the 56-feature model"
        )
    if len(selected) != len(features):
        raise ValueError(
            f"Feature metadata in {path} has {len(selected)} selected frame indices "
            f"for {len(features)} rows"
        )
    return features, selected


def _source_inputs(
    config: RoleConfig,
    row: pd.Series,
) -> tuple[np.ndarray, list[int], list[Path], str]:
    artifact = feature_path(config.features_dir, row)
    features, selected = _load_selected_features(config, row)
    frames = sorted_frames(PROJECT_ROOT / str(row["frames_dir"]))
    if not frames:
        raise FileNotFoundError(f"No frames found in {row['frames_dir']}")
    fingerprint = role_source_fingerprint(artifact, frames, selected)
    return features, selected, frames, fingerprint


def _row_result_is_current(
    config: RoleConfig,
    row: pd.Series,
    require_audit: bool = False,
) -> bool:
    destination = role_path(config, row)
    if not destination.is_file():
        return False
    try:
        _, _, _, source_fingerprint = _source_inputs(config, row)
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return False
    return (
        role_result_is_current(destination, config, source_fingerprint)
        and (not require_audit or audit_path(config, row).is_file())
    )


def build_report(config: RoleConfig, manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, item in manifest.iterrows():
        path = role_path(config, item)
        if not _row_result_is_current(config, item):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "example_id": str(item["example_id"]),
            "view_id": str(item["view_id"]),
            "handball_label": int(item["label"]),
            "domain": str(item["domain"]),
            "predicted_role": result["predicted_role"],
            "is_goalkeeper": result["is_goalkeeper"],
            "goalkeeper_score": result["goalkeeper_score"],
            "role_confidence": result["role_confidence"],
            "coverage": result["coverage"],
            "matched_frames": result["matched_frames"],
            "valid_selected_frames": result["valid_selected_frames"],
            "uncertain": result["uncertain"],
            "result_path": str(path),
        })
    report = pd.DataFrame(rows)
    config.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.report, index=False)
    return report


def summarize_report(report: pd.DataFrame) -> dict[str, object]:
    if report.empty:
        return {
            "examples": 0,
            "goalkeepers": 0,
            "not_goalkeepers": 0,
            "uncertain": 0,
            "mean_coverage": None,
        }
    decisions = report["is_goalkeeper"]
    return {
        "examples": len(report),
        "goalkeepers": int((decisions == True).sum()),  # noqa: E712
        "not_goalkeepers": int((decisions == False).sum()),  # noqa: E712
        "uncertain": int(report["uncertain"].sum()),
        "mean_coverage": float(report["coverage"].mean()),
    }


def audit_manifest(
    config: RoleConfig,
    overwrite: bool = False,
    domain: str | None = None,
    example_contains: str | None = None,
    limit: int | None = None,
    verbose: bool = False,
    save_audits: bool = True,
) -> dict[str, object]:
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
    logger = configure_logging(config.logs_dir / "role_audit.log", verbose)
    pending = [
        row
        for _, row in manifest.iterrows()
        if overwrite
        or not _row_result_is_current(config, row, require_audit=save_audits)
    ]
    detector = FootballRoleDetector(config) if pending else None
    for item_number, (_, row) in enumerate(manifest.iterrows(), start=1):
        destination = role_path(config, row)
        if (
            not overwrite
            and _row_result_is_current(config, row, require_audit=save_audits)
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
            "[%d/%d] detecting role for %s %s",
            item_number,
            len(manifest),
            row["example_id"],
            row["view_id"],
        )
        features, selected, frames, source_fingerprint = _source_inputs(config, row)
        if detector is None:
            raise RuntimeError("Role detector was not initialized")
        result, overlays = classify_selected_actor(
            detector,
            frames,
            features,
            selected,
            config,
        )
        result.update({
            "example_id": str(row["example_id"]),
            "view_id": str(row["view_id"]),
            "handball_label": int(row["label"]),
            "domain": str(row["domain"]),
            "source_fingerprint": source_fingerprint,
        })
        save_role_result(result, destination)
        if save_audits:
            _contact_sheet(overlays, audit_path(config, row))
    report = build_report(config, full_manifest)
    selected_paths = {
        str(role_path(config, row)) for _, row in manifest.iterrows()
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
        description="Classify the selected handball actor as goalkeeper/player/referee."
    )
    parser.add_argument("--config", default="configs/hf_goalkeeper.yaml")
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
        load_role_config(args.config),
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
