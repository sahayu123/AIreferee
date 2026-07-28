import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from training.features import FEATURE_NAMES
from training.motion_features import (
    MOTION_FEATURE_NAMES,
    MOTION_GRU_FEATURE_NAMES,
    MotionFeatureConfig,
    augment_motion_features,
    closest_arm_geometry,
    motion_feature_path,
)
from training.motion_gru import (
    MotionGRUConfig,
    MotionRandomViewDataset,
    load_motion_views,
    motion_normalization,
)
from training.gru import TemporalGRU


def _base_sequence(
    ball_x: list[float],
    ball_y: list[float],
) -> np.ndarray:
    features = np.zeros(
        (len(ball_x), len(FEATURE_NAMES)), dtype=np.float32
    )
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    for row, (x_value, y_value) in zip(features, zip(ball_x, ball_y)):
        row[indices["ball_x"]] = x_value
        row[indices["ball_y"]] = y_value
        row[indices["ball_w"]] = 0.08
        row[indices["ball_h"]] = 0.08
        row[indices["ball_conf"]] = 0.9
        row[indices["ball_valid"]] = 1
        row[indices["player_x"]] = 0.5
        row[indices["player_y"]] = 0.55
        row[indices["player_w"]] = 0.4
        row[indices["player_h"]] = 0.8
        row[indices["player_conf"]] = 0.9
        row[indices["player_valid"]] = 1
        landmarks = {
            "left_shoulder": (0.45, 0.28),
            "left_elbow": (0.45, 0.48),
            "left_wrist": (0.45, 0.68),
            "right_shoulder": (0.62, 0.28),
            "right_elbow": (0.62, 0.48),
            "right_wrist": (0.62, 0.68),
        }
        for name, (x_point, y_point) in landmarks.items():
            row[indices[f"{name}_x"]] = x_point
            row[indices[f"{name}_y"]] = y_point
            row[indices[f"{name}_visibility"]] = 1
            row[indices[f"{name}_valid"]] = 1
    return features


def _frames(
    root: Path,
    ball_x: list[float],
    ball_y: list[float],
) -> list[Path]:
    paths: list[Path] = []
    width, height = 160, 120
    for index, (x_value, y_value) in enumerate(zip(ball_x, ball_y)):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.line(image, (72, 34), (72, 82), (180, 180, 180), 8)
        cv2.line(image, (99, 34), (99, 82), (180, 180, 180), 8)
        cv2.circle(
            image,
            (round(x_value * width), round(y_value * height)),
            6,
            (255, 255, 255),
            -1,
        )
        path = root / f"frame_{index:04d}.jpg"
        assert cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def _feature_config(tmp_path: Path) -> MotionFeatureConfig:
    return MotionFeatureConfig(
        manifest=tmp_path / "manifest.csv",
        base_features_dir=tmp_path / "base",
        motion_features_dir=tmp_path / "motion",
        logs_dir=tmp_path / "logs",
        flow_width=160,
        flow_preset="fast",
        arm_mask_fraction=0.1,
        contact_margin=0.05,
    )


def test_closest_arm_geometry_selects_left_forearm() -> None:
    row = _base_sequence([0.46], [0.58])[0]

    point, distance, side, valid = closest_arm_geometry(row, aspect=4 / 3)

    assert valid
    assert side == -1
    assert distance < 0.03
    assert point.shape == (2,)


def test_motion_augmentation_keeps_base_features_and_adds_flow(
    tmp_path: Path,
) -> None:
    ball_x = [0.30, 0.38, 0.46, 0.38, 0.30]
    ball_y = [0.45, 0.55, 0.65, 0.55, 0.45]
    base = _base_sequence(ball_x, ball_y)
    frame_paths = _frames(tmp_path, ball_x, ball_y)

    enhanced = augment_motion_features(
        base,
        list(range(5)),
        frame_paths,
        fps=5.0,
        config=_feature_config(tmp_path),
    )
    indices = {
        name: index for index, name in enumerate(MOTION_GRU_FEATURE_NAMES)
    }

    assert enhanced.shape == (5, len(MOTION_GRU_FEATURE_NAMES))
    assert np.isfinite(enhanced).all()
    np.testing.assert_array_equal(enhanced[:, : len(FEATURE_NAMES)], base)
    assert enhanced[:, indices["motion_ball_speed"]].max() > 0
    assert enhanced[:, indices["motion_ball_vertical_reversal"]].max() > 0
    assert enhanced[:, indices["motion_bounce_score"]].max() >= 0
    assert enhanced[:, indices["motion_ball_arm_overlap"]].max() > 0
    assert enhanced[:, indices["flow_ball_valid"]].max() == 1
    assert enhanced[:, indices["flow_left_arm_valid"]].max() == 1
    assert enhanced[:, indices["flow_optical_valid"]].max() == 1


def test_motion_augmentation_rejects_bad_selected_index(
    tmp_path: Path,
) -> None:
    base = _base_sequence([0.4], [0.5])
    frame_paths = _frames(tmp_path, [0.4], [0.5])

    with pytest.raises(ValueError, match="selected frame"):
        augment_motion_features(
            base,
            [4],
            frame_paths,
            fps=25,
            config=_feature_config(tmp_path),
        )


def _training_config(tmp_path: Path) -> MotionGRUConfig:
    return MotionGRUConfig(
        manifest=tmp_path / "manifest.csv",
        motion_features_dir=tmp_path / "motion",
        checkpoints_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        baseline_reports_dir=tmp_path / "baseline",
        device="cpu",
        seed=42,
        folds=2,
        fold=0,
        epochs=2,
        batch_size=2,
        hidden_size=8,
        layers=1,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
    )


def _write_motion_artifact(
    config: MotionGRUConfig,
    *,
    example_id: str,
    label: int,
    fold: int,
) -> None:
    row = pd.Series(
        {
            "example_id": example_id,
            "view_id": "primary",
            "label": label,
            "domain": "native",
            "fold": fold,
        }
    )
    path = motion_feature_path(config.motion_features_dir, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.zeros(
        (3, len(MOTION_GRU_FEATURE_NAMES)), dtype=np.float32
    )
    np.savez_compressed(
        path,
        features=features,
        metadata=json.dumps(
            {
                "schema": "ai_referee.motion_features",
                "schema_version": 2,
                "feature_names": MOTION_GRU_FEATURE_NAMES,
                "selected_frame_indices": [0, 1, 2],
            }
        ),
    )


def test_motion_views_and_original_gru_accept_expanded_input(
    tmp_path: Path,
) -> None:
    config = _training_config(tmp_path)
    pd.DataFrame(
        [
            {
                "example_id": "negative",
                "view_id": "primary",
                "label": 0,
                "domain": "native",
                "fold": 0,
            },
            {
                "example_id": "positive",
                "view_id": "primary",
                "label": 1,
                "domain": "native",
                "fold": 1,
            },
        ]
    ).to_csv(config.manifest, index=False)
    _write_motion_artifact(
        config, example_id="negative", label=0, fold=0
    )
    _write_motion_artifact(
        config, example_id="positive", label=1, fold=1
    )

    views = load_motion_views(config)
    mean, std = motion_normalization(views)
    dataset = MotionRandomViewDataset(
        views, mean, std, random_view=False
    )
    features, label, example_id = dataset[1]
    model = TemporalGRU(
        len(MOTION_GRU_FEATURE_NAMES),
        hidden_size=8,
        layers=1,
        dropout=0.1,
    )

    assert features.shape == (3, len(MOTION_GRU_FEATURE_NAMES))
    assert label.item() == 1
    assert example_id == "positive"
    assert model(features.unsqueeze(0)).shape == (1,)


def test_motion_feature_names_are_unique() -> None:
    assert len(MOTION_FEATURE_NAMES) == len(set(MOTION_FEATURE_NAMES))
    assert not (set(MOTION_FEATURE_NAMES) & set(FEATURE_NAMES))
