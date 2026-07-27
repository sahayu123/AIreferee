import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from training.data import aggregate_sequence
from training.features import FEATURE_NAMES, FeatureExtractor
from training.gru import TemporalGRU
from training.inference import resolve_frames
from training.manifest import _imported_rows, sorted_frames
from training.role_audit import _load_selected_features
from training.role_detector import (
    RoleConfig,
    RoleDetection,
    aggregate_role_evidence,
    box_actor_coverage,
    box_iou,
    box_overlap_min_area,
    classify_selected_actor,
    match_selected_player,
    role_config_fingerprint,
    role_result_is_current,
    role_source_fingerprint,
    save_role_result,
    selected_player_box,
)


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))


def _role_config(tmp_path: Path, **overrides) -> RoleConfig:
    values = {
        "checkpoint": tmp_path / "role.pt",
        "device": "cpu",
        "confidence": 0.2,
        "image_size": 640,
        "minimum_actor_coverage": 0.2,
        "minimum_matches": 2,
        "minimum_coverage": 0.25,
        "role_vote_threshold": 0.55,
        "goalkeeper_vote_threshold": 0.7,
        "manifest": tmp_path / "manifest.csv",
        "features_dir": tmp_path / "features",
        "roles_dir": tmp_path / "roles",
        "audits_dir": tmp_path / "audits",
        "report": tmp_path / "report.csv",
        "logs_dir": tmp_path / "logs",
    }
    values.update(overrides)
    return RoleConfig(**values)


def test_imported_manifest_uses_only_selected_auxiliary_label(tmp_path: Path):
    root = tmp_path / "imported"
    rows = ["image_path,label,action_id,clip"]
    for action, label in ((0, "1"), (1, "0")):
        for index in range(12):
            frame = root / "test" / f"action_{action}" / "clip_0" / f"frame_{index}.jpg"
            _image(frame)
            rows.append(f"/old/computer/test/action_{action}/clip_0/frame_{index}.jpg,{label},{action},clip_0")
    (root / "test" / "labels.csv").write_text("\n".join(rows), encoding="utf-8")
    imported = _imported_rows(root, "1")
    assert len(imported) == 1
    assert imported[0]["example_id"] == "imported_action_0000"
    assert imported[0]["label"] == 0 and imported[0]["auxiliary_label"] == "1"


def test_unpadded_frames_are_sorted_numerically(tmp_path: Path):
    for index in (0, 1, 10, 11, 2):
        _image(tmp_path / f"frame_{index}.jpg")
    assert [path.name for path in sorted_frames(tmp_path)] == [
        "frame_0.jpg", "frame_1.jpg", "frame_2.jpg", "frame_10.jpg", "frame_11.jpg"
    ]


def test_temporal_selection_covers_start_and_end():
    class Item:
        quality = 1.0

    selected = FeatureExtractor._temporal_indices([Item() for _ in range(41)], 12)
    assert len(selected) == 12 and selected[0] <= 3 and selected[-1] >= 38


def test_gru_and_baseline_shapes():
    features = np.zeros((12, len(FEATURE_NAMES)), dtype=np.float32)
    assert aggregate_sequence(features).shape == (len(FEATURE_NAMES) * 5,)
    model = TemporalGRU(len(FEATURE_NAMES), hidden_size=8, layers=1, dropout=0.1)
    assert model(torch.zeros(2, 12, len(FEATURE_NAMES))).shape == (2,)


def test_resolve_frames_accepts_labeled_and_candidate_layouts(tmp_path: Path):
    labeled = tmp_path / "labeled"
    _image(labeled / "frames" / "frame_0000.jpg")
    candidate = tmp_path / "candidate"
    _image(candidate / "clean_frames" / "frame_0000.jpg")
    assert len(resolve_frames(labeled)) == 1
    assert len(resolve_frames(candidate)) == 1


def test_selected_player_box_and_iou_use_stored_normalized_features():
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    features[index["player_x"]] = 0.5
    features[index["player_y"]] = 0.5
    features[index["player_w"]] = 0.4
    features[index["player_h"]] = 0.6
    features[index["player_valid"]] = 1
    box = selected_player_box(features, 200, 100)
    assert box is not None
    assert np.allclose(box, (60, 20, 140, 80))
    assert box_iou(box, box) == 1.0


def test_role_matching_uses_actor_overlap_not_highest_confidence():
    actor = (10.0, 10.0, 50.0, 90.0)
    detections = [
        RoleDetection("player", 0.99, (100, 10, 150, 90)),
        RoleDetection("goalkeeper", 0.75, (12, 12, 52, 88)),
    ]
    matched, overlap = match_selected_player(
        actor, detections, minimum_actor_coverage=0.2
    )
    assert matched is detections[1]
    assert overlap > 0.8

    containing = RoleDetection("goalkeeper", 0.8, (0, 0, 80, 100))
    assert box_iou(actor, containing.box) < 0.5
    assert box_overlap_min_area(actor, containing.box) == 1.0
    assert box_actor_coverage(actor, containing.box) == 1.0
    matched, _ = match_selected_player(
        actor, [containing], minimum_actor_coverage=0.8
    )
    assert matched is containing

    tiny = RoleDetection("goalkeeper", 0.99, (20, 20, 25, 30))
    assert box_overlap_min_area(actor, tiny.box) == 1.0
    matched, _ = match_selected_player(
        actor, [tiny], minimum_actor_coverage=0.2
    )
    assert matched is None


def test_temporal_role_voting_requires_repeated_evidence(tmp_path: Path):
    config = _role_config(tmp_path, minimum_matches=3)
    evidence = [
        {
            "selected_box": [0, 0, 10, 20],
            "matched_role": role,
            "confidence": confidence,
            "actor_coverage": 0.9,
        }
        for role, confidence in (
            ("goalkeeper", 0.9),
            ("goalkeeper", 0.8),
            ("goalkeeper", 0.85),
            ("player", 0.2),
        )
    ]
    result = aggregate_role_evidence(evidence, config)
    assert result["predicted_role"] == "goalkeeper"
    assert result["is_goalkeeper"] is True
    assert result["goalkeeper_score"] > 0.9

    insufficient = aggregate_role_evidence(evidence[:1], config)
    assert insufficient["predicted_role"] == "unknown"
    assert insufficient["is_goalkeeper"] is None

    repeated = [{**evidence[0], "frame_index": 4} for _ in range(3)]
    duplicate_result = aggregate_role_evidence(repeated, config)
    assert duplicate_result["matched_frames"] == 1
    assert duplicate_result["is_goalkeeper"] is None

    ambiguous = [
        {
            "selected_box": [0, 0, 10, 20],
            "matched_role": role,
            "confidence": confidence,
            "actor_coverage": 1.0,
            "frame_index": frame_index,
        }
        for frame_index, (role, confidence) in enumerate((
            ("goalkeeper", 0.9),
            ("goalkeeper", 0.9),
            ("player", 1.0),
        ))
    ]
    ambiguous_result = aggregate_role_evidence(ambiguous, config)
    assert 0.55 < ambiguous_result["goalkeeper_score"] < 0.7
    assert ambiguous_result["predicted_role"] == "unknown"
    assert ambiguous_result["is_goalkeeper"] is None


def test_role_classifier_matches_existing_actor_across_selected_frames(tmp_path: Path):
    class FakeDetector:
        def detect(self, frame: np.ndarray) -> list[RoleDetection]:
            return [RoleDetection("goalkeeper", 0.9, (30, 20, 70, 80))]

    frames: list[Path] = []
    for index in range(2):
        path = tmp_path / f"frame_{index}.jpg"
        cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
        frames.append(path)
    features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    features[:, index["player_x"]] = 0.5
    features[:, index["player_y"]] = 0.5
    features[:, index["player_w"]] = 0.4
    features[:, index["player_h"]] = 0.6
    features[:, index["player_valid"]] = 1
    result, overlays = classify_selected_actor(
        FakeDetector(),
        frames,
        features,
        [0, 1],
        _role_config(tmp_path),
    )
    assert result["is_goalkeeper"] is True
    assert result["matched_frames"] == 2
    assert len(overlays) == 2


def test_role_classifier_skips_detector_without_selected_actor(tmp_path: Path):
    class CountingDetector:
        calls = 0

        def detect(self, frame: np.ndarray) -> list[RoleDetection]:
            self.calls += 1
            return []

    path = tmp_path / "frame_0.jpg"
    cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))
    detector = CountingDetector()
    result, _ = classify_selected_actor(
        detector,
        [path],
        np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        [0],
        _role_config(tmp_path),
    )
    assert detector.calls == 0
    assert result["is_goalkeeper"] is None


def test_role_audit_rejects_nonbaseline_feature_schema(tmp_path: Path):
    config = _role_config(tmp_path)
    destination = config.features_dir / "native" / "example" / "primary.npz"
    destination.parent.mkdir(parents=True)
    np.savez_compressed(
        destination,
        features=np.zeros((12, len(FEATURE_NAMES) + 1), dtype=np.float32),
        metadata=json.dumps({
            "feature_names": FEATURE_NAMES + ["goalkeeper_score"],
            "selected_frame_indices": list(range(12)),
        }),
    )
    row = pd.Series({
        "domain": "native",
        "example_id": "example",
        "view_id": "primary",
    })
    with pytest.raises(ValueError, match="Unexpected feature shape"):
        _load_selected_features(config, row)


def test_role_cache_tracks_feature_and_selected_frame_inputs(tmp_path: Path):
    config = _role_config(tmp_path)
    config.checkpoint.write_bytes(b"checkpoint")
    artifact = tmp_path / "features.npz"
    artifact.write_bytes(b"first features")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"first frame")
    fingerprint = role_source_fingerprint(artifact, [frame], [0])
    destination = tmp_path / "role.json"
    save_role_result(
        {
            "schema_version": 1,
            "config_fingerprint": role_config_fingerprint(config),
            "source_fingerprint": fingerprint,
        },
        destination,
    )
    assert role_result_is_current(destination, config, fingerprint)

    artifact.write_bytes(b"changed features")
    changed = role_source_fingerprint(artifact, [frame], [0])
    assert changed != fingerprint
    assert not role_result_is_current(destination, config, changed)
