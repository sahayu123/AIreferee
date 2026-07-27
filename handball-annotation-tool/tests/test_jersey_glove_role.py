from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.jersey_glove_role import (
    classify_goalkeeper_after_handball,
    compute_glove_evidence,
    compute_jersey_team_evidence,
    decide_goalkeeper,
    hand_crop_from_landmarks,
    load_jersey_glove_config,
    _overlapping_actor_fragment_ids,
)
from training.prtreid_role import TrackObservation


def _config(**overrides):
    config = load_jersey_glove_config(
        "configs/jersey_glove_goalkeeper.yaml"
    )
    defaults = {
        "jersey_minimum_track_frames": 1,
        "jersey_maximum_samples_per_track": 1,
        "jersey_minimum_pixels": 20,
        "jersey_minimum_comparison_tracks": 4,
        "jersey_minimum_cluster_tracks": 2,
    }
    defaults.update(overrides)
    return replace(config, **defaults)


def _jersey_scene(
    tmp_path: Path, actor_color: tuple[int, int, int]
) -> tuple[list[Path], dict[int, list[TrackObservation]]]:
    colors = {
        1: actor_color,
        2: (0, 0, 255),
        3: (0, 0, 255),
        4: (255, 0, 0),
        5: (255, 0, 0),
    }
    frames: list[Path] = []
    tracks: dict[int, list[TrackObservation]] = {}
    for frame_index, (track_id, color) in enumerate(colors.items()):
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[10:110, 20:100] = color
        path = tmp_path / f"frame_{frame_index}.jpg"
        assert cv2.imwrite(str(path), image)
        frames.append(path)
        tracks[track_id] = [
            TrackObservation(
                frame_index,
                path,
                (20.0, 10.0, 100.0, 110.0),
                0.9,
            )
        ]
    return frames, tracks


def test_jersey_actor_matching_dominant_team_has_high_match(tmp_path: Path):
    frames, tracks = _jersey_scene(tmp_path, (0, 0, 255))
    evidence = compute_jersey_team_evidence(
        frames, tracks, 1, _config()
    )
    assert evidence["available"] is True
    assert evidence["team_match_score"] > 0.99
    assert evidence["outlier_score"] < 0.01
    assert sorted(evidence["team_cluster_sizes"]) == [2, 2]


def test_jersey_unique_actor_is_outlier(tmp_path: Path):
    frames, tracks = _jersey_scene(tmp_path, (0, 255, 0))
    evidence = compute_jersey_team_evidence(
        frames, tracks, 1, _config()
    )
    assert evidence["available"] is True
    assert evidence["team_match_score"] < 0.10
    assert evidence["outlier_score"] > 0.90


def test_jersey_insufficient_peers_is_missing_not_zero(tmp_path: Path):
    frames, tracks = _jersey_scene(tmp_path, (0, 255, 0))
    evidence = compute_jersey_team_evidence(
        frames, {key: tracks[key] for key in (1, 2, 3)}, 1, _config()
    )
    assert evidence["available"] is False
    assert evidence["reason"] == "insufficient_comparison_tracks"
    assert evidence["team_match_score"] is None
    assert evidence["outlier_score"] is None


def test_overlapping_actor_track_is_excluded_as_fragment(tmp_path: Path):
    frame = tmp_path / "frame.jpg"
    actor = TrackObservation(0, frame, (20.0, 10.0, 80.0, 110.0), 0.9)
    duplicate = TrackObservation(
        0, frame, (22.0, 12.0, 79.0, 112.0), 0.8
    )
    nearby_player = TrackObservation(
        0, frame, (70.0, 10.0, 120.0, 110.0), 0.9
    )
    fragments = _overlapping_actor_fragment_ids(
        {1: [actor], 2: [duplicate], 3: [nearby_player]}, 1
    )
    assert fragments == [2]


def test_hand_crop_uses_limb_geometry_and_native_resolution():
    grid = (np.indices((140, 140)).sum(axis=0) % 2 * 255).astype(np.uint8)
    frame = np.repeat(grid[:, :, None], 3, axis=2)
    config = _config(hand_minimum_blur_variance=1.0)
    hand, rejection = hand_crop_from_landmarks(
        frame,
        7,
        "right",
        (20, 10, 100, 110),
        np.asarray([55.0, 35.0]),
        np.asarray([60.0, 55.0]),
        np.asarray([70.0, 75.0]),
        0.9,
        0.9,
        0.9,
        config,
    )
    assert rejection is None
    assert hand is not None
    assert hand.frame_index == 7
    assert hand.side == "right"
    assert hand.native_side == config.hand_minimum_native_side
    assert hand.crop.shape[:2] == (
        config.hand_minimum_native_side,
        config.hand_minimum_native_side,
    )


def test_hand_crop_rejects_low_confidence_and_blur():
    config = _config(hand_minimum_blur_variance=15.0)
    frame = np.zeros((140, 140, 3), dtype=np.uint8)
    landmarks = (
        frame,
        1,
        "left",
        (20, 10, 100, 110),
        np.asarray([55.0, 35.0]),
        np.asarray([60.0, 55.0]),
        np.asarray([70.0, 75.0]),
    )
    hand, rejection = hand_crop_from_landmarks(
        *landmarks, 0.9, 0.9, 0.2, config
    )
    assert hand is None
    assert rejection == "low_wrist_confidence"
    hand, rejection = hand_crop_from_landmarks(
        *landmarks, 0.9, 0.9, 0.9, config
    )
    assert hand is None
    assert rejection == "blurry_crop"


def test_disabled_glove_stage_preserves_missingness():
    evidence = compute_glove_evidence(
        [],
        {
            "distinct_frames": 0,
            "sufficient_for_glove_inference": False,
        },
        _config(glove_enabled=False),
    )
    assert evidence["available"] is False
    assert evidence["reason"] == "classifier_disabled"
    assert evidence["glove_probability"] is None


def test_disabled_prtreid_uses_strong_team_jersey_for_field_player():
    decision = decide_goalkeeper(
        {"confident": True},
        {
            "available": True,
            "team_match_score": 0.95,
            "outlier_score": 0.05,
        },
        {"available": False, "scores": None},
        {"available": False, "glove_probability": None},
        _config(use_prtreid_evidence=False),
    )
    assert decision["status"] == "not_goalkeeper"
    assert decision["is_goalkeeper"] is False
    assert decision["reason"] == "field_team_jersey_match"


@pytest.mark.parametrize(
    ("association", "jersey", "prt", "glove", "expected"),
    [
        (
            {"confident": True},
            {
                "available": True,
                "team_match_score": 0.05,
                "outlier_score": 0.95,
            },
            {"available": True, "scores": {"player": 0.05}},
            {"available": True, "glove_probability": 0.90},
            "goalkeeper",
        ),
        (
            {"confident": True},
            {
                "available": True,
                "team_match_score": 0.95,
                "outlier_score": 0.05,
            },
            {"available": True, "scores": {"player": 0.90}},
            {"available": True, "glove_probability": 0.90},
            "not_goalkeeper",
        ),
        (
            {"confident": True},
            {
                "available": True,
                "team_match_score": 0.05,
                "outlier_score": 0.95,
            },
            {"available": True, "scores": {"player": 0.05}},
            {"available": False, "glove_probability": None},
            "unknown",
        ),
        (
            {"confident": False},
            {
                "available": True,
                "team_match_score": 0.05,
                "outlier_score": 0.95,
            },
            {"available": True, "scores": {"player": 0.05}},
            {"available": True, "glove_probability": 0.99},
            "unknown",
        ),
    ],
)
def test_three_state_decision_and_safety_guards(
    association, jersey, prt, glove, expected
):
    decision = decide_goalkeeper(
        association, jersey, prt, glove, _config()
    )
    assert decision["status"] == expected


def test_not_handball_skips_every_goalkeeper_model(tmp_path: Path):
    class ForbiddenTracker:
        def track(self, frame_paths):
            raise AssertionError("tracker must not run for not-handball")

    frame = tmp_path / "frame.jpg"
    assert cv2.imwrite(str(frame), np.zeros((20, 20, 3), dtype=np.uint8))
    result = classify_goalkeeper_after_handball(
        [frame],
        np.zeros((1, 56), dtype=np.float32),
        [0],
        {},
        0.49,
        0.50,
        _config(),
        tracker=ForbiddenTracker(),
    )
    assert result["evaluated"] is False
    assert result["status"] == "not_evaluated"
    assert result["is_goalkeeper"] is None
