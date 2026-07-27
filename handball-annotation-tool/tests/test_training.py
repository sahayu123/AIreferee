import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
from training.prtreid_role import (
    TrackObservation,
    YOLOPersonTracker,
    associate_handball_actor,
    classify_actor_role,
    load_prtreid_config,
    prtreid_source_fingerprint,
)
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


def _prtreid_config(tmp_path: Path, **overrides):
    config = load_prtreid_config("configs/prtreid_goalkeeper.yaml")
    values = {
        "person_detector": tmp_path / "yolo11n.pt",
        "prtreid_checkpoint": tmp_path / "prtreid.pth.tar",
        "worker_command": ("fake-worker",),
        "manifest": tmp_path / "manifest.csv",
        "features_dir": tmp_path / "features",
        "roles_dir": tmp_path / "roles",
        "audits_dir": tmp_path / "audits",
        "report": tmp_path / "report.csv",
        "logs_dir": tmp_path / "logs",
    }
    values.update(overrides)
    return replace(config, **values)


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


def test_prtreid_actor_association_prefers_original_arm_anchor(tmp_path: Path):
    frame = tmp_path / "frame_0.jpg"
    _image(frame)
    tracks = {
        7: [TrackObservation(0, frame, (10, 5, 60, 95), 0.7)],
        9: [TrackObservation(0, frame, (100, 5, 155, 95), 0.99)],
    }
    metadata = {
        "frames_before": 0,
        "closest_arm": {"start": [25, 30], "end": [35, 50]},
    }
    association = associate_handball_actor(
        tracks,
        [(100, 200)],
        np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        [0],
        metadata,
        _prtreid_config(tmp_path),
    )
    assert association.track_id == 7
    assert association.method == "closest_arm"
    assert association.confident


def test_prtreid_fallback_association_uses_saved_actor_boxes(tmp_path: Path):
    frames = []
    for frame_index in range(3):
        frame = tmp_path / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(frame), np.zeros((100, 200, 3), dtype=np.uint8))
        frames.append(frame)
    tracks = {
        3: [
            TrackObservation(index, frames[index], (60, 20, 140, 80), 0.6)
            for index in range(3)
        ],
        8: [
            TrackObservation(index, frames[index], (150, 20, 195, 80), 0.99)
            for index in range(3)
        ],
    }
    features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    feature_index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    features[:, feature_index["player_x"]] = 0.5
    features[:, feature_index["player_y"]] = 0.5
    features[:, feature_index["player_w"]] = 0.4
    features[:, feature_index["player_h"]] = 0.6
    features[:, feature_index["player_valid"]] = 1
    association = associate_handball_actor(
        tracks,
        [(100, 200)] * 3,
        features,
        [0, 1, 2],
        {},
        _prtreid_config(tmp_path),
    )
    assert association.track_id == 3
    assert association.confident


def test_prtreid_classifies_every_actor_track_frame_but_abstains_on_weak_link(
    tmp_path: Path,
):
    frame_paths = []
    observations = []
    for frame_index in range(4):
        frame = tmp_path / f"frame_{frame_index}.jpg"
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[10:90, 50:90] = (0, 0, 255)
        cv2.imwrite(str(frame), image)
        frame_paths.append(frame)
        observations.append(
            TrackObservation(frame_index, frame, (50, 10, 90, 90), 0.9)
        )

    class FakeTracker:
        def track(self, frame_paths):
            return {4: observations}

    class FakeWorker:
        def __init__(self):
            self.crops = []

        def predict(self, crops):
            self.crops.extend(crops)
            return [
                {
                    "frame_path": crop["frame_path"],
                    "bbox": crop["bbox"],
                    "role_probabilities": {
                        "ball": 0.01,
                        "goalkeeper": 0.95,
                        "other": 0.01,
                        "player": 0.02,
                        "referee": 0.01,
                    },
                    "predicted_role": "goalkeeper",
                    "confidence": 0.95,
                    "margin": 0.93,
                }
                for crop in crops
            ]

    features = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    feature_index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    features[:, feature_index["player_x"]] = 0.5
    features[:, feature_index["player_y"]] = 0.5
    features[:, feature_index["player_w"]] = 0.4
    features[:, feature_index["player_h"]] = 0.8
    features[:, feature_index["player_valid"]] = 1
    worker = FakeWorker()
    result = classify_actor_role(
        frame_paths,
        features,
        [0, 1],
        {},
        _prtreid_config(
            tmp_path,
            minimum_anchor_coverage=0.20,
            minimum_association_score=0.80,
        ),
        tracker=FakeTracker(),
        worker=worker,
    )
    assert len(worker.crops) == 4
    assert len({Path(crop["frame_path"]).name for crop in worker.crops}) == 4
    assert result["tracks"][0]["aggregate"]["is_goalkeeper"] is True
    assert result["association"]["confident"] is False
    assert result["predicted_role"] == "unknown"
    assert result["is_goalkeeper"] is None


def test_prtreid_cache_fingerprint_covers_unselected_frames(tmp_path: Path):
    artifact = tmp_path / "features.npz"
    artifact.write_bytes(b"base features")
    selected = tmp_path / "frame_0.jpg"
    unselected = tmp_path / "frame_1.jpg"
    selected.write_bytes(b"selected")
    unselected.write_bytes(b"AAAAAAAAAA")
    first = prtreid_source_fingerprint(
        artifact, [selected, unselected], [0]
    )
    original_stat = unselected.stat()
    unselected.write_bytes(b"BBBBBBBBBB")
    os.utime(
        unselected,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    second = prtreid_source_fingerprint(
        artifact, [selected, unselected], [0]
    )
    assert first != second


def test_prtreid_fallback_marks_equal_crossing_tracks_ambiguous(tmp_path: Path):
    frames = []
    for frame_index in range(3):
        frame = tmp_path / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(frame), np.zeros((100, 100, 3), dtype=np.uint8))
        frames.append(frame)
    tracks = {
        track_id: [
            TrackObservation(index, frames[index], (30, 10, 70, 90), 0.9)
            for index in range(3)
        ]
        for track_id in (2, 5)
    }
    features = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    feature_index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
    features[:, feature_index["player_x"]] = 0.5
    features[:, feature_index["player_y"]] = 0.5
    features[:, feature_index["player_w"]] = 0.4
    features[:, feature_index["player_h"]] = 0.8
    features[:, feature_index["player_valid"]] = 1
    association = associate_handball_actor(
        tracks,
        [(100, 100)] * 3,
        features,
        [0, 1, 2],
        {},
        _prtreid_config(tmp_path, minimum_anchor_votes=3),
    )
    assert association.margin == pytest.approx(0.0)
    assert association.conflicting
    assert not association.confident


def test_prtreid_three_frame_fragment_cannot_decide_41_frame_clip(tmp_path: Path):
    frame_paths = []
    for frame_index in range(41):
        frame = tmp_path / f"frame_{frame_index}.jpg"
        cv2.imwrite(str(frame), np.zeros((100, 100, 3), dtype=np.uint8))
        frame_paths.append(frame)
    observations = [
        TrackObservation(index, frame_paths[index], (30, 10, 70, 90), 0.9)
        for index in range(3)
    ]

    class FragmentTracker:
        def track(self, frame_paths):
            return {6: observations}

    class GoalkeeperWorker:
        def predict(self, crops):
            return [
                {
                    "frame_path": crop["frame_path"],
                    "bbox": crop["bbox"],
                    "role_probabilities": {
                        "ball": 0.01,
                        "goalkeeper": 0.95,
                        "other": 0.01,
                        "player": 0.02,
                        "referee": 0.01,
                    },
                    "predicted_role": "goalkeeper",
                    "confidence": 0.95,
                    "margin": 0.93,
                }
                for crop in crops
            ]

    features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    metadata = {
        "frames_before": 1,
        "closest_arm": {"start": [40, 40], "end": [45, 45]},
    }
    result = classify_actor_role(
        frame_paths,
        features,
        [1],
        metadata,
        _prtreid_config(tmp_path),
        tracker=FragmentTracker(),
        worker=GoalkeeperWorker(),
    )
    aggregate = result["tracks"][0]["aggregate"]
    assert aggregate["prediction_frames"] == 3
    assert aggregate["coverage"] == pytest.approx(3 / 41)
    assert aggregate["is_goalkeeper"] is None
    assert result["association"]["confident"] is True
    assert result["is_goalkeeper"] is None


def test_prtreid_tracker_reset_reuses_predictor_without_callback_growth():
    class FakeByteTracker:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    byte_trackers = [FakeByteTracker(), FakeByteTracker()]
    predictor = SimpleNamespace(trackers=byte_trackers, vid_path=["old", "old"])
    tracker = YOLOPersonTracker.__new__(YOLOPersonTracker)
    tracker.model = SimpleNamespace(predictor=predictor)
    original_predictor = tracker.model.predictor
    tracker._reset_for_clip()
    assert tracker.model.predictor is original_predictor
    assert predictor.vid_path == [None, None]
    assert [item.reset_calls for item in byte_trackers] == [1, 1]
