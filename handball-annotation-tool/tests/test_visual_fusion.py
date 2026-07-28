import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from training.features import FEATURE_NAMES, feature_path
from training.visual_features import (
    VISUAL_BACKBONE,
    VISUAL_EMBEDDING_SIZE,
    MobileNetVisualEncoder,
    context_crop_box,
    encode_crops,
    save_visual_artifact,
    visual_feature_path,
)
from training.visual_fusion import (
    FusionRandomViewDataset,
    VisualFusionConfig,
    VisualFusionGRU,
    fusion_normalization,
    load_fusion_views,
)


def feature_row() -> np.ndarray:
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    values[indices["player_x"]] = 0.5
    values[indices["player_y"]] = 0.5
    values[indices["player_w"]] = 0.2
    values[indices["player_h"]] = 0.4
    values[indices["player_valid"]] = 1
    values[indices["ball_x"]] = 0.7
    values[indices["ball_y"]] = 0.5
    values[indices["ball_w"]] = 0.05
    values[indices["ball_h"]] = 0.05
    values[indices["ball_valid"]] = 1
    return values


def test_context_crop_contains_player_and_ball() -> None:
    box = context_crop_box(
        feature_row(),
        frame_width=1000,
        frame_height=500,
        crop_margin=0,
    )

    left, top, right, bottom = box
    assert left <= 400 and top <= 150
    assert right >= 725 and bottom >= 350
    assert right < 1000 and bottom < 500


def test_context_crop_falls_back_to_full_frame_without_player() -> None:
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    assert context_crop_box(
        values,
        frame_width=640,
        frame_height=360,
        crop_margin=0.35,
    ) == (0, 0, 640, 360)


def test_visual_encoder_returns_one_embedding_per_crop() -> None:
    model = MobileNetVisualEncoder(pretrained=False)
    crops = [
        Image.new("RGB", (80, 120), "green"),
        Image.new("RGB", (120, 80), "blue"),
    ]

    embeddings = encode_crops(
        crops,
        model,
        device="cpu",
        batch_size=2,
    )

    assert embeddings.shape == (2, VISUAL_EMBEDDING_SIZE)
    assert np.isfinite(embeddings).all()


def test_visual_fusion_model_shapes() -> None:
    model = VisualFusionGRU(
        numerical_size=len(FEATURE_NAMES),
        visual_size=VISUAL_EMBEDDING_SIZE,
        visual_projection_size=8,
        hidden_size=16,
        layers=1,
        dropout=0.1,
    )

    output = model(
        torch.zeros(3, 12, len(FEATURE_NAMES)),
        torch.zeros(3, 12, VISUAL_EMBEDDING_SIZE),
    )

    assert output.shape == (3,)


def _config(tmp_path: Path) -> VisualFusionConfig:
    return VisualFusionConfig(
        manifest=tmp_path / "manifest.csv",
        base_features_dir=tmp_path / "base",
        visual_features_dir=tmp_path / "visual",
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
        visual_projection_size=4,
        dropout=0.1,
        learning_rate=1e-3,
        weight_decay=1e-3,
        patience=2,
    )


def _write_feature_pair(
    config: VisualFusionConfig,
    *,
    example_id: str,
    label: int,
    fold: int,
    visual_selected: list[int] | None = None,
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
    base_path = feature_path(config.base_features_dir, row)
    visual_path = visual_feature_path(config.visual_features_dir, row)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [0, 1]
    numerical = np.stack([feature_row(), feature_row()])
    np.savez_compressed(
        base_path,
        features=numerical,
        metadata=json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "selected_frame_indices": selected,
            }
        ),
    )
    save_visual_artifact(
        visual_path,
        np.zeros((2, VISUAL_EMBEDDING_SIZE), dtype=np.float32),
        {
            "backbone": VISUAL_BACKBONE,
            "selected_frame_indices": (
                selected if visual_selected is None else visual_selected
            ),
        },
    )


def test_load_fusion_views_and_dataset_keep_modalities_aligned(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = pd.DataFrame(
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
    )
    manifest.to_csv(config.manifest, index=False)
    _write_feature_pair(
        config,
        example_id="negative",
        label=0,
        fold=0,
    )
    _write_feature_pair(
        config,
        example_id="positive",
        label=1,
        fold=1,
    )

    views = load_fusion_views(config)
    normalization = fusion_normalization(views)
    dataset = FusionRandomViewDataset(
        views,
        *normalization,
        random_view=False,
    )
    numerical, visual, label, example_id = dataset[1]

    assert len(views) == 2
    assert numerical.shape == (2, len(FEATURE_NAMES))
    assert visual.shape == (2, VISUAL_EMBEDDING_SIZE)
    assert label.item() == 1
    assert example_id == "positive"


def test_load_fusion_views_rejects_temporal_misalignment(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    pd.DataFrame(
        [
            {
                "example_id": "broken",
                "view_id": "primary",
                "label": 1,
                "domain": "native",
                "fold": 0,
            }
        ]
    ).to_csv(config.manifest, index=False)
    _write_feature_pair(
        config,
        example_id="broken",
        label=1,
        fold=0,
        visual_selected=[1, 0],
    )

    with pytest.raises(ValueError, match="alignment"):
        load_fusion_views(config)


def test_save_visual_artifact_rejects_wrong_embedding_size(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Visual embeddings"):
        save_visual_artifact(
            tmp_path / "bad.npz",
            np.zeros((2, 10), dtype=np.float32),
            {},
        )
