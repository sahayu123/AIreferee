from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

import training.goalkeeper_classifier as classifier_module
import training.supervised_goalkeeper as supervised_module
from training.goalkeeper_classifier import (
    GoalkeeperClassifier,
    aggregate_track_probabilities,
    build_goalkeeper_model,
    save_checkpoint,
)
from training.goalkeeper_dataset import (
    assign_group_folds,
    collect_source_images,
    extract_player_candidates,
)
from training.prtreid_role import ActorAssociation, TrackObservation
from training.supervised_goalkeeper import (
    classify_supervised_goalkeeper,
    load_supervised_goalkeeper_config,
)


def test_track_probability_aggregation_is_conservative():
    goalkeeper = aggregate_track_probabilities(
        [0.91, 0.84, 0.88, 0.20],
        minimum_crops=3,
        goalkeeper_threshold=0.75,
        minimum_agreement=0.60,
    )
    assert goalkeeper["status"] == "goalkeeper"
    assert goalkeeper["is_goalkeeper"] is True

    field_player = aggregate_track_probabilities(
        [0.05, 0.12, 0.20, 0.80],
        minimum_crops=3,
        not_goalkeeper_threshold=0.25,
        minimum_agreement=0.60,
    )
    assert field_player["status"] == "not_goalkeeper"

    uncertain = aggregate_track_probabilities(
        [0.10, 0.90], minimum_crops=3
    )
    assert uncertain["status"] == "unknown"
    assert uncertain["reason"] == "insufficient_valid_player_crops"


def test_safe_checkpoint_round_trip_is_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = build_goalkeeper_model(pretrained=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.copy_(torch.tensor([0.0, 1.0]))
    checkpoint = save_checkpoint(
        tmp_path / "goalkeeper.pt",
        model,
        input_size=(32, 32),
        decision_thresholds=(0.2, 0.8),
    )
    observed_weights = []
    original_builder = classifier_module.mobilenet_v3_small

    def builder_spy(*args, **kwargs):
        observed_weights.append(kwargs.get("weights"))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(
        classifier_module, "mobilenet_v3_small", builder_spy
    )
    classifier = GoalkeeperClassifier(checkpoint, batch_size=1)
    probabilities = classifier.predict_goalkeeper_probability(
        [np.zeros((40, 20, 3), dtype=np.uint8)]
    )
    assert probabilities.shape == (1,)
    assert probabilities[0] == pytest.approx(0.7310586, abs=1e-5)
    assert classifier.not_goalkeeper_threshold == 0.2
    assert classifier.goalkeeper_threshold == 0.8
    assert observed_weights == [None]


def test_crop_extraction_requires_review_for_multiple_people(tmp_path: Path):
    goalkeeper_source = tmp_path / "goalkeeper" / "match_a"
    field_source = tmp_path / "not_goalkeeper" / "match_b"
    goalkeeper_source.mkdir(parents=True)
    field_source.mkdir(parents=True)
    image = np.full((240, 320, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(goalkeeper_source / "one.jpg"), image)
    assert cv2.imwrite(str(field_source / "many.jpg"), image)

    class Detector:
        def detect(self, value):
            if int(value[0, 0, 0]) == 127 and len(calls) == 0:
                calls.append(1)
                return [{"bbox": [80, 20, 180, 220], "confidence": 0.9}]
            return [
                {"bbox": [20, 20, 100, 220], "confidence": 0.9},
                {"bbox": [180, 20, 280, 220], "confidence": 0.8},
            ]

    calls: list[int] = []
    sources = collect_source_images(goalkeeper_source.parent, field_source.parent)
    manifest = extract_player_candidates(
        sources, Detector(), tmp_path / "output"
    )
    goalkeeper_rows = manifest[
        manifest["source_label"] == "goalkeeper"
    ]
    field_rows = manifest[
        manifest["source_label"] == "not_goalkeeper"
    ]
    assert goalkeeper_rows.iloc[0]["review_label"] == "goalkeeper"
    assert set(field_rows["status"]) == {"needs_review"}
    assert set(field_rows["review_label"]) == {""}


def test_group_folds_never_split_a_source_group():
    rows = []
    for label in ("goalkeeper", "not_goalkeeper"):
        for group in range(5):
            for image in range(2):
                rows.append(
                    {
                        "crop_path": f"/tmp/{label}-{group}-{image}.jpg",
                        "review_label": label,
                        "source_group": f"{label}-match-{group}",
                    }
                )
    result = assign_group_folds(pd.DataFrame(rows), folds=5, seed=7)
    assert set(result["fold"]) == set(range(5))
    assert result.groupby("source_group")["fold"].nunique().max() == 1


def test_actor_track_classification_uses_multiple_full_player_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame_paths = []
    observations = []
    for index in range(4):
        path = tmp_path / f"frame_{index:02d}.jpg"
        image = np.full((240, 320, 3), 100 + index, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        frame_paths.append(path)
        observations.append(
            TrackObservation(
                frame_index=index,
                frame_path=path,
                box=(80.0, 20.0, 200.0, 220.0),
                detection_confidence=0.9,
            )
        )
    config = replace(
        load_supervised_goalkeeper_config(
            "configs/supervised_goalkeeper.yaml"
        ),
        minimum_player_height=50,
        minimum_blur_variance=0.0,
        minimum_samples=3,
        maximum_samples=4,
    )
    association = ActorAssociation(
        track_id=7,
        method="test",
        score=0.9,
        margin=0.5,
        anchor_votes=4,
        confident=True,
        conflicting=False,
        evidence=(),
    )
    monkeypatch.setattr(
        supervised_module,
        "track_all_people",
        lambda *args, **kwargs: {7: observations},
    )
    monkeypatch.setattr(
        supervised_module,
        "associate_handball_actor",
        lambda *args, **kwargs: association,
    )

    class Classifier:
        not_goalkeeper_threshold = 0.25
        goalkeeper_threshold = 0.75

        def predict_goalkeeper_probability(self, crops):
            assert len(crops) == 4
            return np.asarray([0.90, 0.85, 0.80, 0.40])

    result = classify_supervised_goalkeeper(
        frame_paths,
        np.zeros((1, 56), dtype=np.float32),
        [0],
        {},
        config,
        classifier=Classifier(),
    )
    assert result["status"] == "goalkeeper"
    assert result["is_goalkeeper"] is True
    assert result["valid_crops"] == 4
    assert len(result["frame_probabilities"]) == 4
