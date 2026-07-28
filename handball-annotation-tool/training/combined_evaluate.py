"""Fold-separated evaluation of the gated handball/goalkeeper pipeline.

The GRU is evaluated first. Goalkeeper analysis runs only for a raw-positive
clip, and only a confirmed goalkeeper vetoes the final handball-foul result:

    raw_negative -> skip goalkeeper -> final_negative
    raw_positive -> classify actor -> final = actor is not confirmed goalkeeper

Unknown or failed actor analysis preserves a raw-positive handball decision.
Positive-clip role results are cached for resumable full-manifest runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import torch

from handball_annotator.runtime import get_device

from .config import TrainConfig, load_train_config, project_path
from .data import FeatureView, feature_metadata, load_views
from .features import FEATURE_NAMES, feature_path
from .glove_classifier import GloveClassifier
from .goalkeeper_classifier import GoalkeeperClassifier
from .gru import TemporalGRU, predict_views
from .jersey_glove_role import (
    JerseyGloveConfig,
    MediaPipeActorHandExtractor,
    classify_goalkeeper_after_handball,
    jersey_glove_config_fingerprint,
    load_jersey_glove_config,
    save_jersey_glove_result,
)
from .manifest import sorted_frames
from .prtreid_role import (
    PRTReIDWorkerClient,
    YOLOPersonTracker,
    prtreid_source_fingerprint,
)
from .supervised_goalkeeper import (
    SupervisedGoalkeeperConfig,
    classify_supervised_goalkeeper,
    config_fingerprint as supervised_config_fingerprint,
    load_supervised_goalkeeper_config,
)

SCHEMA_VERSION = 2
EXPECTED_FOLDS = (0, 1, 2, 3, 4)
FINAL_RULE = "raw_negative_skips_role_raw_positive_vetoes_confirmed_goalkeeper"
CANONICAL_ROLE_STATUSES = (
    "goalkeeper",
    "not_goalkeeper",
    "unknown",
    "not_evaluated",
    "error",
)
REQUIRED_MANIFEST_COLUMNS = {
    "example_id",
    "view_id",
    "label",
    "domain",
    "frames_dir",
    "fold",
    "source_group",
}


@dataclass(frozen=True)
class OOFResult:
    predictions: pd.DataFrame
    checkpoint_fingerprints: dict[str, dict[str, Any]]
    device: str


@dataclass(frozen=True)
class RoleInputs:
    features: np.ndarray
    selected_indices: list[int]
    frame_paths: list[Path]
    metadata: dict[str, Any]
    source_fingerprint: str
    feature_artifact: Path


class RoleAnalyzer(Protocol):
    def __call__(
        self,
        row: pd.Series,
        raw_probability: float,
        threshold: float,
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        ...


def decision_policy(threshold: float) -> dict[str, Any]:
    return {
        "rule": FINAL_RULE,
        "threshold": float(threshold),
        "raw_not_handball": "skip_goalkeeper_and_publish_not_handball",
        "confirmed_goalkeeper": "veto_raw_positive",
        "not_goalkeeper": "publish_handball_outfield",
        "unknown": "preserve_raw_positive_as_handball_actor_unknown",
        "error": "preserve_raw_positive_as_handball_actor_unknown",
        "positive_class": "handball",
        "metric_denominator": "all_manifest_rows",
        "goalkeeper_metric_denominator": "raw_positive_rows_only",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_fingerprint(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_component(value: object, maximum: int = 96) -> str:
    text = str(value).replace("/", "_").replace("\\", "_").strip()
    text = text if text not in ("", ".", "..") else "unnamed"
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{text[:maximum]}_{digest}"


def load_evaluation_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    manifest = pd.read_csv(path)
    missing = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("Cannot evaluate an empty manifest")
    manifest = manifest.copy()
    numeric_folds = pd.to_numeric(manifest["fold"], errors="raise")
    if not np.equal(numeric_folds, np.floor(numeric_folds)).all():
        raise ValueError("Manifest folds must be integers")
    manifest["fold"] = numeric_folds.astype(int)
    numeric_labels = pd.to_numeric(manifest["label"], errors="raise")
    if not np.equal(numeric_labels, np.floor(numeric_labels)).all():
        raise ValueError("Manifest labels must be integers")
    manifest["label"] = numeric_labels.astype(int)
    if not manifest["label"].isin((0, 1)).all():
        raise ValueError("Manifest labels must be binary")
    observed_folds = tuple(sorted(int(value) for value in manifest["fold"].unique()))
    if observed_folds != EXPECTED_FOLDS:
        raise ValueError(
            f"Expected manifest folds {EXPECTED_FOLDS}, found {observed_folds}"
        )
    split_groups = (
        manifest.groupby("source_group", dropna=False)["fold"].nunique()
    )
    leaking_groups = split_groups[split_groups > 1]
    if not leaking_groups.empty:
        raise ValueError(
            "Source groups cross evaluation folds: "
            + ", ".join(str(value) for value in leaking_groups.index[:5])
        )
    keys = manifest[["example_id", "view_id"]].astype(str)
    if keys.duplicated().any():
        duplicate = keys[keys.duplicated(keep=False)].iloc[0].to_dict()
        raise ValueError(f"Duplicate manifest example/view key: {duplicate}")
    manifest["_manifest_order"] = np.arange(len(manifest), dtype=int)
    return manifest


def _default_checkpoint_loader(path: Path, device: str) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def load_oof_probabilities(
    config: TrainConfig,
    manifest: pd.DataFrame,
    *,
    view_loader: Callable[[Path, Path, bool], list[FeatureView]] = load_views,
    checkpoint_loader: Callable[[Path, str], dict[str, Any]] = (
        _default_checkpoint_loader
    ),
    model_factory: Callable[..., Any] = TemporalGRU,
    predictor: Callable[..., pd.DataFrame] = predict_views,
    device_resolver: Callable[[str], str] = get_device,
) -> OOFResult:
    """Load each row's outer-fold model and return one OOF probability.

    The rows are separated from that fold's parameter-training subset.  The
    existing trainer did use the same outer fold for best-epoch selection, so
    callers must not describe the resulting metrics as a pristine test estimate.
    """

    if int(config.folds) != len(EXPECTED_FOLDS):
        raise ValueError(
            f"Combined evaluation requires exactly {len(EXPECTED_FOLDS)} folds"
        )
    checkpoint_paths = {
        fold: config.checkpoints_dir / f"gru_fold{fold}_best.pt"
        for fold in EXPECTED_FOLDS
    }
    missing = [path for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        rendered = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "All five best checkpoints are required before evaluation. "
            f"Missing:\n{rendered}"
        )
    checkpoint_fingerprints = {
        str(fold): {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for fold, path in checkpoint_paths.items()
    }
    views = view_loader(config.manifest, config.features_dir, True)
    view_by_key: dict[tuple[str, str], FeatureView] = {}
    for view in views:
        key = (str(view.example_id), str(view.view_id))
        if key in view_by_key:
            raise ValueError(f"Duplicate cached feature view: {key}")
        view_by_key[key] = view
    manifest_keys = {
        (str(row["example_id"]), str(row["view_id"]))
        for _, row in manifest.iterrows()
    }
    missing_views = sorted(manifest_keys - set(view_by_key))
    extra_views = sorted(set(view_by_key) - manifest_keys)
    if missing_views or extra_views:
        raise ValueError(
            "Cached feature views do not match the manifest: "
            f"missing={len(missing_views)} extra={len(extra_views)}"
        )

    device = device_resolver(config.device)
    records: list[dict[str, Any]] = []
    for fold in EXPECTED_FOLDS:
        path = checkpoint_paths[fold]
        checkpoint = checkpoint_loader(path, device)
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(
                f"Checkpoint {path} records fold {checkpoint.get('fold')}, "
                f"expected {fold}"
            )
        if list(checkpoint.get("feature_names", [])) != FEATURE_NAMES:
            raise ValueError(f"Checkpoint feature schema mismatch: {path}")
        fold_views = [
            view
            for view in views
            if int(view.fold) == fold
        ]
        if not fold_views:
            raise ValueError(f"No cached feature views found for fold {fold}")
        model = model_factory(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        predictions = predictor(
            model,
            fold_views,
            checkpoint["mean"],
            checkpoint["std"],
            device,
            config.batch_size,
        )
        expected_keys = {
            (str(view.example_id), str(view.view_id)) for view in fold_views
        }
        predicted_keys: set[tuple[str, str]] = set()
        for _, prediction in predictions.iterrows():
            key = (
                str(prediction["example_id"]),
                str(prediction["view_id"]),
            )
            if key in predicted_keys:
                raise ValueError(f"Duplicate OOF prediction: {key}")
            predicted_keys.add(key)
            probability = float(prediction["probability"])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid OOF probability for {key}: {probability}")
            records.append(
                {
                    "example_id": key[0],
                    "view_id": key[1],
                    "oof_fold": fold,
                    "raw_handball_probability": probability,
                    "checkpoint_path": str(path),
                    "checkpoint_sha256": checkpoint_fingerprints[str(fold)][
                        "sha256"
                    ],
                }
            )
        if predicted_keys != expected_keys:
            raise ValueError(
                f"Fold {fold} predictions do not match its cached feature views"
            )
    predictions = pd.DataFrame(records)
    if len(predictions) != len(manifest):
        raise ValueError(
            f"Expected {len(manifest)} OOF probabilities, got {len(predictions)}"
        )
    merged = manifest[
        ["example_id", "view_id", "fold", "_manifest_order"]
    ].merge(
        predictions,
        on=["example_id", "view_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["raw_handball_probability"].isna().any():
        raise ValueError("At least one manifest row has no OOF probability")
    if not np.equal(merged["fold"], merged["oof_fold"]).all():
        raise ValueError("At least one row was scored by the wrong fold checkpoint")
    predictions = merged.sort_values("_manifest_order").drop(
        columns=["fold", "_manifest_order"]
    )
    return OOFResult(
        predictions=predictions.reset_index(drop=True),
        checkpoint_fingerprints=checkpoint_fingerprints,
        device=str(device),
    )


def load_role_inputs(
    row: pd.Series,
    features_dir: Path,
) -> RoleInputs:
    artifact = feature_path(features_dir, row)
    if not artifact.is_file():
        raise FileNotFoundError(f"Cached feature artifact not found: {artifact}")
    with np.load(artifact, allow_pickle=False) as loaded:
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
        raise ValueError(f"Feature schema mismatch in {artifact}")
    if len(selected) != len(features):
        raise ValueError(
            f"Feature/index length mismatch in {artifact}: "
            f"{len(features)} != {len(selected)}"
        )
    frames_dir = project_path(str(row["frames_dir"]))
    frame_paths = sorted_frames(frames_dir)
    if not frame_paths:
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
    source_fingerprint = prtreid_source_fingerprint(
        artifact,
        frame_paths,
        selected,
        metadata_path,
    )
    return RoleInputs(
        features=features,
        selected_indices=selected,
        frame_paths=frame_paths,
        metadata=metadata,
        source_fingerprint=source_fingerprint,
        feature_artifact=artifact,
    )


class JerseyGloveRoleRuntime:
    """Lazily create and reuse expensive role-analysis resources."""

    def __init__(
        self,
        config: JerseyGloveConfig,
        *,
        classifier: Callable[..., dict[str, Any]] = (
            classify_goalkeeper_after_handball
        ),
    ):
        self.config = config
        self.classifier = classifier
        self.tracker: YOLOPersonTracker | None = None
        self.worker: PRTReIDWorkerClient | None = None
        self.hand_extractor: MediaPipeActorHandExtractor | None = None
        self.glove_model: GloveClassifier | None = None

    def _ensure_resources(self) -> None:
        if self.tracker is not None:
            return
        self.tracker = YOLOPersonTracker(self.config.base_config)
        if self.config.use_prtreid_evidence:
            self.worker = PRTReIDWorkerClient(self.config.base_config)
        if self.config.glove_enabled:
            self.hand_extractor = MediaPipeActorHandExtractor(self.config)
            self.glove_model = GloveClassifier(
                self.config.glove_checkpoint,
                device=self.config.glove_device,
                batch_size=self.config.glove_batch_size,
            )

    def analyze(
        self,
        inputs: RoleInputs,
        raw_probability: float,
        threshold: float,
    ) -> dict[str, Any]:
        self._ensure_resources()
        return self.classifier(
            inputs.frame_paths,
            inputs.features,
            inputs.selected_indices,
            inputs.metadata,
            raw_probability,
            threshold,
            self.config,
            tracker=self.tracker,
            role_worker=self.worker,
            hand_extractor=self.hand_extractor,
            glove_model=self.glove_model,
            force_evaluation=True,
        )

    def close(self) -> None:
        if self.hand_extractor is not None:
            self.hand_extractor.close()
            self.hand_extractor = None
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def __enter__(self) -> "JerseyGloveRoleRuntime":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class SupervisedGoalkeeperRuntime:
    """Reuse one tracker and one trained image classifier for all clips."""

    def __init__(self, config: SupervisedGoalkeeperConfig):
        self.config = config
        self.tracker = YOLOPersonTracker(config.base_config)
        self.classifier = GoalkeeperClassifier(
            config.checkpoint,
            device=config.device,
            batch_size=config.batch_size,
        )

    def analyze(
        self,
        inputs: RoleInputs,
        _raw_probability: float,
        _threshold: float,
    ) -> dict[str, Any]:
        return classify_supervised_goalkeeper(
            inputs.frame_paths,
            inputs.features,
            inputs.selected_indices,
            inputs.metadata,
            self.config,
            tracker=self.tracker,
            classifier=self.classifier,
        )

    def __enter__(self) -> "SupervisedGoalkeeperRuntime":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class CachedRoleAnalyzer:
    """Per-row source/config/checkpoint-aware role-result cache."""

    def __init__(
        self,
        cache_dir: Path,
        config_fingerprint: str,
        input_loader: Callable[[pd.Series], RoleInputs],
        analyze_uncached: Callable[[RoleInputs, float, float], dict[str, Any]],
        *,
        overwrite: bool = False,
    ):
        self.cache_dir = cache_dir
        self.config_fingerprint = config_fingerprint
        self.input_loader = input_loader
        self.analyze_uncached = analyze_uncached
        self.overwrite = overwrite

    def cache_path(self, row: pd.Series) -> Path:
        return (
            self.cache_dir
            / _safe_component(row["domain"], maximum=40)
            / _safe_component(row["example_id"])
            / f"{_safe_component(row['view_id'], maximum=48)}.json"
        )

    def _is_current(
        self,
        result: dict[str, Any],
        *,
        source_fingerprint: str,
        raw_probability: float,
        threshold: float,
        checkpoint_sha256: str,
    ) -> bool:
        try:
            cached_probability = float(result["combined_oof_probability"])
            cached_threshold = float(result["combined_threshold"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            result.get("combined_cache_schema_version") == SCHEMA_VERSION
            and result.get("config_fingerprint") == self.config_fingerprint
            and result.get("source_fingerprint") == source_fingerprint
            and result.get("combined_oof_checkpoint_sha256")
            == checkpoint_sha256
            and math.isclose(
                cached_probability, raw_probability, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(
                cached_threshold, threshold, rel_tol=0.0, abs_tol=1e-12
            )
        )

    def _decorate(
        self, result: dict[str, Any], path: Path, cache_hit: bool
    ) -> dict[str, Any]:
        decorated = dict(result)
        decorated["_combined_cache_hit"] = cache_hit
        decorated["_combined_cache_path"] = str(path)
        return decorated

    def __call__(
        self,
        row: pd.Series,
        raw_probability: float,
        threshold: float,
        checkpoint_sha256: str,
    ) -> dict[str, Any]:
        destination = self.cache_path(row)
        try:
            inputs = self.input_loader(row)
        except Exception as exc:
            return self._decorate(
                {
                    "evaluated": False,
                    "execution_status": "error",
                    "status": "error",
                    "is_goalkeeper": None,
                    "goalkeeper_evidence_score": None,
                    "reason": "role_input_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "config_fingerprint": self.config_fingerprint,
                    "source_fingerprint": None,
                },
                destination,
                False,
            )
        if destination.is_file() and not self.overwrite:
            try:
                cached = json.loads(destination.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict) and self._is_current(
                cached,
                source_fingerprint=inputs.source_fingerprint,
                raw_probability=raw_probability,
                threshold=threshold,
                checkpoint_sha256=checkpoint_sha256,
            ):
                return self._decorate(cached, destination, True)
        try:
            result = self.analyze_uncached(
                inputs, raw_probability, threshold
            )
            if not isinstance(result, dict):
                raise TypeError("Goalkeeper analyzer must return a dictionary")
            result = dict(result)
            result["execution_status"] = "completed"
        except Exception as exc:
            result = {
                "schema_version": 1,
                "evaluated": False,
                "execution_status": "error",
                "status": "error",
                "is_goalkeeper": None,
                "goalkeeper_evidence_score": None,
                "reason": "goalkeeper_analysis_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        result.update(
            {
                "config_fingerprint": self.config_fingerprint,
                "source_fingerprint": inputs.source_fingerprint,
                "combined_cache_schema_version": SCHEMA_VERSION,
                "combined_oof_probability": float(raw_probability),
                "combined_threshold": float(threshold),
                "combined_oof_checkpoint_sha256": checkpoint_sha256,
                "feature_artifact": str(inputs.feature_artifact),
            }
        )
        try:
            save_jersey_glove_result(result, destination)
        except Exception as exc:
            result = {
                "evaluated": False,
                "execution_status": "error",
                "status": "error",
                "is_goalkeeper": None,
                "goalkeeper_evidence_score": None,
                "reason": "role_cache_write_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "config_fingerprint": self.config_fingerprint,
                "source_fingerprint": inputs.source_fingerprint,
                "combined_cache_schema_version": SCHEMA_VERSION,
                "combined_oof_probability": float(raw_probability),
                "combined_threshold": float(threshold),
                "combined_oof_checkpoint_sha256": checkpoint_sha256,
                "feature_artifact": str(inputs.feature_artifact),
            }
        return self._decorate(result, destination, False)


def canonical_role_status(result: Mapping[str, Any]) -> str:
    raw_status = str(result.get("status", "unknown")).strip().lower()
    if (
        str(result.get("execution_status", "")).lower() == "error"
        or raw_status == "error"
    ):
        return "error"
    is_goalkeeper = result.get("is_goalkeeper")
    is_boolean = isinstance(is_goalkeeper, (bool, np.bool_))
    if raw_status == "goalkeeper":
        return (
            "goalkeeper"
            if is_boolean and bool(is_goalkeeper)
            else "error"
        )
    if raw_status == "not_goalkeeper":
        return (
            "not_goalkeeper"
            if is_boolean and not bool(is_goalkeeper)
            else "error"
        )
    if raw_status == "not_evaluated":
        return "not_evaluated" if is_goalkeeper is None else "error"
    if raw_status in ("", "unknown", "unavailable"):
        return "unknown" if is_goalkeeper is None else "error"
    return "error"


def fuse_final_decision(
    raw_probability: float,
    threshold: float,
    role_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply positive-only role gating and the conservative veto policy."""

    probability = float(raw_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("raw_probability must be finite and between 0 and 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    raw_positive = probability >= threshold
    role_status = canonical_role_status(role_result)
    veto = raw_positive and role_status == "goalkeeper"
    fallback = raw_positive and role_status in ("unknown", "error")
    final_positive = raw_positive and role_status != "goalkeeper"
    if not raw_positive:
        rule = "raw_not_handball_goalkeeper_skipped"
        event_label = "not_handball"
        actor_role = None
    elif veto:
        rule = "confirmed_goalkeeper_veto"
        event_label = "handball_goalkeeper"
        actor_role = "goalkeeper"
    elif role_status == "not_goalkeeper":
        rule = "raw_handball_confirmed_outfield"
        event_label = "handball_outfield"
        actor_role = "outfield"
    elif role_status == "error":
        rule = "raw_handball_role_error_fallback"
        event_label = "handball_actor_unknown"
        actor_role = "unknown"
    else:
        rule = "raw_handball_role_unknown_fallback"
        event_label = "handball_actor_unknown"
        actor_role = "unknown"
    return {
        "raw_prediction": int(raw_positive),
        "final_prediction": int(final_positive),
        "combined_event_label": event_label,
        "handball_actor_role": actor_role,
        "goalkeeper_status": role_status,
        "goalkeeper_analysis_required": bool(raw_positive),
        "goalkeeper_analysis_invoked": bool(
            role_result.get("analysis_invoked", False)
        ),
        "goalkeeper_veto": bool(veto),
        "role_fallback": bool(fallback),
        "role_error": raw_positive and role_status == "error",
        "final_decision_rule": rule,
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _flatten_role_result(result: Mapping[str, Any]) -> dict[str, Any]:
    association = result.get("association")
    association = association if isinstance(association, dict) else {}
    jersey = result.get("jersey")
    jersey = jersey if isinstance(jersey, dict) else {}
    glove = result.get("glove")
    glove = glove if isinstance(glove, dict) else {}
    return {
        "goalkeeper_status_raw": str(result.get("status", "unknown")),
        "goalkeeper_evaluated": bool(result.get("evaluated", False)),
        "goalkeeper_execution_status": result.get("execution_status"),
        "is_goalkeeper": result.get("is_goalkeeper"),
        "goalkeeper_evidence_score": result.get(
            "goalkeeper_evidence_score"
        ),
        "goalkeeper_reason": result.get("reason"),
        "actor_track_id": result.get("actor_track_id"),
        "association_confident": association.get("confident"),
        "association_score": association.get("score"),
        "jersey_team_match": jersey.get("team_match_score"),
        "jersey_outlier": jersey.get("outlier_score"),
        "glove_probability": glove.get("glove_probability"),
        "role_error_type": result.get("error_type"),
        "role_error_message": result.get("error"),
        "role_cache_hit": bool(result.get("_combined_cache_hit", False)),
        "role_cache_path": result.get("_combined_cache_path"),
        "role_source_fingerprint": result.get("source_fingerprint"),
    }


def evaluate_combined_rows(
    rows: pd.DataFrame,
    role_analyzer: RoleAnalyzer,
    *,
    threshold: float = 0.5,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> pd.DataFrame:
    """Gate goalkeeper analysis on raw-positive clips, then fuse decisions."""

    required = {
        "example_id",
        "view_id",
        "label",
        "raw_handball_probability",
        "checkpoint_sha256",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Evaluation rows are missing columns: {missing}")
    if rows.empty:
        raise ValueError("Cannot evaluate zero rows")
    output: list[dict[str, Any]] = []
    total = len(rows)
    for number, (_, row) in enumerate(rows.iterrows(), start=1):
        probability = float(row["raw_handball_probability"])
        raw_positive = probability >= threshold
        if not raw_positive:
            role_result = {
                "evaluated": False,
                "execution_status": "skipped",
                "status": "not_evaluated",
                "is_goalkeeper": None,
                "goalkeeper_evidence_score": None,
                "reason": "raw_not_handball_gate",
                "analysis_invoked": False,
            }
        else:
            try:
                role_result = role_analyzer(
                    row,
                    probability,
                    threshold,
                    str(row["checkpoint_sha256"]),
                )
                if not isinstance(role_result, dict):
                    raise TypeError(
                        "Goalkeeper analyzer must return a dictionary"
                    )
                role_result = dict(role_result)
                role_result["analysis_invoked"] = True
            except Exception as exc:
                role_result = {
                    "evaluated": False,
                    "execution_status": "error",
                    "status": "error",
                    "is_goalkeeper": None,
                    "goalkeeper_evidence_score": None,
                    "reason": "goalkeeper_analysis_invocation_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "analysis_invoked": True,
                }
        decision = fuse_final_decision(probability, threshold, role_result)
        record = {
            str(column): _json_value(row[column])
            for column in rows.columns
            if not str(column).startswith("_")
        }
        record.update(_flatten_role_result(role_result))
        record.update(decision)
        label = int(row["label"])
        record.update(
            {
                "raw_predicted_label": (
                    "handball" if decision["raw_prediction"] else "not_handball"
                ),
                "final_predicted_label": (
                    "handball"
                    if decision["final_prediction"]
                    else "not_handball"
                ),
                "raw_correct": decision["raw_prediction"] == label,
                "final_correct": decision["final_prediction"] == label,
            }
        )
        output.append(record)
        if progress is not None:
            progress(number, total, record)
    return pd.DataFrame(output)


def hard_binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, Any]:
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    if labels_array.ndim != 1 or predictions_array.ndim != 1:
        raise ValueError("Labels and predictions must be one-dimensional")
    if len(labels_array) == 0 or len(labels_array) != len(predictions_array):
        raise ValueError("Labels and predictions must have equal non-zero length")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not np.isin(predictions_array, (0, 1)).all():
        raise ValueError("Predictions must be binary")
    true_positive = int(
        np.sum((labels_array == 1) & (predictions_array == 1))
    )
    false_positive = int(
        np.sum((labels_array == 0) & (predictions_array == 1))
    )
    false_negative = int(
        np.sum((labels_array == 1) & (predictions_array == 0))
    )
    true_negative = int(
        np.sum((labels_array == 0) & (predictions_array == 0))
    )

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)
    f1 = divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)
    accuracy = divide(true_positive + true_negative, len(labels_array))
    return {
        "examples": int(len(labels_array)),
        "positives": int(np.sum(labels_array == 1)),
        "negatives": int(np.sum(labels_array == 0)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


def summarize_evaluation(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    fingerprints: Mapping[str, Any],
    output_csv: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    baseline = hard_binary_metrics(
        predictions["label"], predictions["raw_prediction"]
    )
    combined = hard_binary_metrics(
        predictions["label"], predictions["final_prediction"]
    )
    metric_names = ("precision", "recall", "f1", "accuracy")
    deltas: dict[str, Any] = {
        name: float(combined[name]) - float(baseline[name])
        for name in metric_names
    }
    deltas.update(
        {
            name: int(combined[name]) - int(baseline[name])
            for name in ("tn", "fp", "fn", "tp")
        }
    )
    deltas["confusion_matrix"] = (
        np.asarray(combined["confusion_matrix"], dtype=int)
        - np.asarray(baseline["confusion_matrix"], dtype=int)
    ).tolist()
    status_counter = Counter(str(value) for value in predictions["goalkeeper_status"])
    raw_status_counter = Counter(
        str(value) for value in predictions["goalkeeper_status_raw"]
    )
    event_counter = Counter(
        str(value) for value in predictions["combined_event_label"]
    )
    total = len(predictions)
    invoked_mask = predictions["goalkeeper_analysis_invoked"].astype(bool)
    evaluated_mask = predictions["goalkeeper_evaluated"].astype(bool)
    error_mask = predictions["role_error"].astype(bool) & invoked_mask
    cache_hit_mask = predictions["role_cache_hit"].astype(bool) & invoked_mask
    invocation_count = int(invoked_mask.sum())
    role_completed = int((evaluated_mask & invoked_mask).sum())
    error_count = int(error_mask.sum())
    known_count = int(
        (
            invoked_mask
            & predictions["goalkeeper_status"].isin(
                ("goalkeeper", "not_goalkeeper")
            )
        ).sum()
    )

    def rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 1.0

    counts = {
        "examples": total,
        "raw_positive": int(predictions["raw_prediction"].sum()),
        "combined_positive": int(predictions["final_prediction"].sum()),
        "combined_event_label": dict(sorted(event_counter.items())),
        "goalkeeper_status": {
            status: int(status_counter.get(status, 0))
            for status in CANONICAL_ROLE_STATUSES
        },
        "raw_goalkeeper_status": dict(sorted(raw_status_counter.items())),
        "goalkeeper_vetoes": int(
            predictions["goalkeeper_veto"].astype(bool).sum()
        ),
        "role_fallbacks": int(
            predictions["role_fallback"].astype(bool).sum()
        ),
        "role_errors": error_count,
        "role_cache_hits": int(cache_hit_mask.sum()),
        "role_cache_misses": int(invocation_count - cache_hit_mask.sum()),
        "goalkeeper_analysis_skipped": int(total - invocation_count),
        "goalkeeper_known_decisions": known_count,
    }
    completion = {
        "expected_rows": total,
        "oof_probability_rows": int(
            predictions["raw_handball_probability"].notna().sum()
        ),
        "goalkeeper_analysis_eligible_rows": int(
            predictions["raw_prediction"].sum()
        ),
        "goalkeeper_analysis_invocations": invocation_count,
        "goalkeeper_analysis_skipped": int(total - invocation_count),
        "goalkeeper_analysis_completed": role_completed,
        "goalkeeper_analysis_errors": error_count,
        "goalkeeper_known_decisions": known_count,
        "final_prediction_rows": int(
            predictions["final_prediction"].notna().sum()
        ),
        "final_prediction_completion_rate": float(
            predictions["final_prediction"].notna().mean()
        ),
        "goalkeeper_analysis_completion_rate": rate(
            role_completed, invocation_count
        ),
        "goalkeeper_known_decision_rate": rate(
            known_count, invocation_count
        ),
        "all_rows_have_final_predictions": bool(
            predictions["final_prediction"].notna().all()
        ),
    }
    policy = decision_policy(threshold)
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation": "five_fold_outer_fold_combined_handball_goalkeeper",
        "policy": policy,
        "policy_fingerprint": _json_fingerprint(policy),
        "metrics": {
            "baseline_raw_gru": baseline,
            "combined_final": combined,
            "combined_minus_baseline": deltas,
        },
        "counts": counts,
        "completion": completion,
        "fingerprints": dict(fingerprints),
        "outputs": {
            "predictions_csv": str(output_csv) if output_csv else None,
            "metrics_json": str(output_json) if output_json else None,
        },
        "limitations": {
            "goalkeeper_metrics_available": False,
            "reason": "manifest_has_no_goalkeeper_ground_truth_labels",
            "glove_enabled": fingerprints.get("glove_enabled"),
            "outer_fold_used_for_epoch_selection": True,
            "estimate_note": (
                "Fold-separated predictions are mildly optimistic because "
                "each outer fold was also used to select the best epoch."
            ),
        },
    }


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def run_combined_evaluation(
    train_config_path: str | Path = "configs/temporal_classifier.yaml",
    role_config_path: str | Path = "configs/jersey_glove_goalkeeper.yaml",
    output_csv: str | Path = "artifacts/reports/combined_oof_predictions.csv",
    output_json: str | Path = "artifacts/reports/combined_oof_metrics.json",
    role_cache_dir: str | Path = "artifacts/roles_combined_oof",
    *,
    threshold: float = 0.5,
    overwrite_role_cache: bool = False,
    supervised_goalkeeper_config_path: str | Path | None = None,
) -> dict[str, Any]:
    train_config_file = project_path(train_config_path)
    output_csv_path = project_path(output_csv)
    output_json_path = project_path(output_json)
    cache_path = project_path(role_cache_dir)
    train_config = load_train_config(train_config_file)
    if supervised_goalkeeper_config_path is not None:
        role_config_file = project_path(supervised_goalkeeper_config_path)
        role_config = load_supervised_goalkeeper_config(role_config_file)
        role_config_fingerprint = supervised_config_fingerprint(role_config)
        runtime_factory: Callable[[], Any] = lambda: (
            SupervisedGoalkeeperRuntime(role_config)
        )
        backend = "supervised_player_crop_actor_track"
        glove_enabled: bool | None = None
        goalkeeper_checkpoint = role_config.checkpoint
    else:
        role_config_file = project_path(role_config_path)
        role_config = load_jersey_glove_config(role_config_file)
        role_config_fingerprint = jersey_glove_config_fingerprint(role_config)
        runtime_factory = lambda: JerseyGloveRoleRuntime(role_config)
        backend = "jersey_glove_actor_track"
        glove_enabled = bool(role_config.glove_enabled)
        goalkeeper_checkpoint = None
    manifest = load_evaluation_manifest(train_config.manifest)
    oof = load_oof_probabilities(train_config, manifest)
    rows = manifest.merge(
        oof.predictions,
        on=["example_id", "view_id"],
        how="left",
        validate="one_to_one",
    ).sort_values("_manifest_order")
    running = {"vetoes": 0, "errors": 0}

    def show_progress(
        number: int, total: int, record: Mapping[str, Any]
    ) -> None:
        running["vetoes"] += int(bool(record["goalkeeper_veto"]))
        running["errors"] += int(bool(record["role_error"]))
        cache_status = (
            "skipped"
            if not bool(record["goalkeeper_analysis_invoked"])
            else "hit"
            if bool(record["role_cache_hit"])
            else "miss"
        )
        print(
            (
                f"[{number}/{total}] fold={int(record['fold'])} "
                f"p={float(record['raw_handball_probability']):.6f} "
                f"raw={record['raw_predicted_label']} "
                f"gk={record['goalkeeper_status']} "
                f"final={record['final_predicted_label']} "
                f"cache={cache_status} "
                f"vetoes={running['vetoes']} errors={running['errors']} "
                f"{record['example_id']} {record['view_id']}"
            ),
            flush=True,
        )

    with runtime_factory() as runtime:
        analyzer = CachedRoleAnalyzer(
            cache_path,
            role_config_fingerprint,
            lambda row: load_role_inputs(row, train_config.features_dir),
            runtime.analyze,
            overwrite=overwrite_role_cache,
        )
        predictions = evaluate_combined_rows(
            rows,
            analyzer,
            threshold=threshold,
            progress=show_progress,
        )
    fingerprints: dict[str, Any] = {
        "manifest": {
            "path": str(train_config.manifest),
            "sha256": _sha256_file(train_config.manifest),
        },
        "train_config": {
            "path": str(train_config_file),
            "sha256": _sha256_file(train_config_file),
        },
        "goalkeeper_config": {
            "path": str(role_config_file),
            "sha256": _sha256_file(role_config_file),
            "runtime_fingerprint": role_config_fingerprint,
        },
        "goalkeeper_checkpoint": (
            {
                "path": str(goalkeeper_checkpoint),
                "sha256": _sha256_file(goalkeeper_checkpoint),
            }
            if goalkeeper_checkpoint is not None
            else None
        ),
        "checkpoints": oof.checkpoint_fingerprints,
        "evaluator_source_sha256": _sha256_file(Path(__file__)),
        "device": oof.device,
        "goalkeeper_backend": backend,
        "glove_enabled": glove_enabled,
    }
    policy_fingerprint = _json_fingerprint(decision_policy(threshold))
    predictions["train_config_sha256"] = fingerprints["train_config"]["sha256"]
    predictions["goalkeeper_config_fingerprint"] = role_config_fingerprint
    predictions["policy_fingerprint"] = policy_fingerprint
    _atomic_write_csv(predictions, output_csv_path)
    summary = summarize_evaluation(
        predictions,
        threshold=threshold,
        fingerprints=fingerprints,
        output_csv=output_csv_path,
        output_json=output_json_path,
    )
    if summary["policy_fingerprint"] != policy_fingerprint:
        raise RuntimeError("Internal policy fingerprint mismatch")
    _atomic_write_json(summary, output_json_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate five-fold handball probabilities with goalkeeper "
            "analysis gated to raw-positive clips."
        )
    )
    parser.add_argument(
        "--train-config", default="configs/temporal_classifier.yaml"
    )
    parser.add_argument(
        "--goalkeeper-config",
        default="configs/jersey_glove_goalkeeper.yaml",
    )
    parser.add_argument(
        "--supervised-goalkeeper-config",
        help=(
            "Use the trained full-player goalkeeper classifier instead of "
            "the jersey/glove experiment."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/reports/combined_oof_predictions.csv",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/reports/combined_oof_metrics.json",
    )
    parser.add_argument(
        "--role-cache-dir",
        default="artifacts/roles_combined_oof",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite-role-cache", action="store_true")
    args = parser.parse_args()
    summary = run_combined_evaluation(
        args.train_config,
        args.goalkeeper_config,
        args.output_csv,
        args.output_json,
        args.role_cache_dir,
        threshold=args.threshold,
        overwrite_role_cache=args.overwrite_role_cache,
        supervised_goalkeeper_config_path=(
            args.supervised_goalkeeper_config
        ),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
