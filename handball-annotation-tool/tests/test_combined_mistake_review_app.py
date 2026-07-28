import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from combined_mistake_review_app import (
    AUDIT_COLUMNS,
    actor_crops,
    filter_predictions,
    load_audit,
    load_feature_evidence,
    normalized_feature_box,
    prepare_predictions,
    render_frame,
    save_audit_record,
)
from training.features import FEATURE_NAMES, feature_path


def prediction_frame() -> pd.DataFrame:
    common = {
        "view_id": "primary",
        "domain": "native",
        "frames_dir": "dataset/example/frames",
        "frame_count": 41,
        "fps": 25.0,
        "fold": 0,
        "raw_handball_probability": 0.8,
        "raw_predicted_label": "handball",
        "final_predicted_label": "not_handball",
        "goalkeeper_status": "goalkeeper",
        "goalkeeper_evidence_score": 0.9,
        "goalkeeper_reason": "",
        "goalkeeper_analysis_invoked": True,
        "role_cache_path": "",
        "combined_event_label": "not_handball",
        "actor_track_id": 1,
        "association_score": 0.8,
        "final_decision_rule": "test",
        "source_name": "match.mp4",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "example_id": "veto-hurt",
                "label": 1,
                "raw_prediction": 1,
                "final_prediction": 0,
                "raw_correct": True,
                "final_correct": False,
                "goalkeeper_veto": True,
            },
            {
                **common,
                "example_id": "veto-fixed",
                "label": 0,
                "raw_prediction": 1,
                "final_prediction": 0,
                "raw_correct": False,
                "final_correct": True,
                "goalkeeper_veto": True,
            },
            {
                **common,
                "example_id": "base-miss",
                "label": 1,
                "raw_prediction": 0,
                "final_prediction": 0,
                "raw_correct": False,
                "final_correct": False,
                "goalkeeper_veto": False,
                "goalkeeper_analysis_invoked": False,
                "goalkeeper_status": "not_evaluated",
            },
            {
                **common,
                "example_id": "surviving-alarm",
                "label": 0,
                "raw_prediction": 1,
                "final_prediction": 1,
                "raw_correct": False,
                "final_correct": False,
                "goalkeeper_veto": False,
                "goalkeeper_status": "outfield",
                "final_predicted_label": "handball",
            },
        ]
    )


def test_prepare_and_filter_pipeline_failure_categories() -> None:
    prepared = prepare_predictions(prediction_frame())

    assert len(
        filter_predictions(
            prepared,
            category="Combined mistakes",
            audit_status="All",
        )
    ) == 3
    assert len(
        filter_predictions(
            prepared,
            category="Final false positives",
            audit_status="All",
        )
    ) == 1
    assert len(
        filter_predictions(
            prepared,
            category="Final false negatives",
            audit_status="All",
        )
    ) == 2
    assert len(
        filter_predictions(
            prepared,
            category="Base-model misses",
            audit_status="All",
        )
    ) == 1
    assert len(
        filter_predictions(
            prepared,
            category="All goalkeeper vetoes",
            audit_status="All",
        )
    ) == 2


def test_filter_search_is_safe_without_source_name() -> None:
    prepared = prepare_predictions(
        prediction_frame().drop(columns=["source_name"])
    )

    selected = filter_predictions(
        prepared,
        category="All clips",
        audit_status="All",
        search="base-miss",
    )

    assert selected["example_id"].tolist() == ["base-miss"]


def test_audit_round_trip_and_review_filter(tmp_path: Path) -> None:
    path = tmp_path / "audit.csv"
    audit = pd.DataFrame(columns=AUDIT_COLUMNS)
    record = {
        "example_id": "veto-hurt",
        "view_id": "primary",
        "reviewed": True,
        "actual_actor_role": "Outfield",
        "actor_track_correct": "Correct",
        "root_cause": "Goalkeeper classifier error",
        "notes": "Clearly an outfield player.",
        "updated_at_utc": "2026-07-28T00:00:00+00:00",
    }

    saved = save_audit_record(audit, path, record)
    loaded = load_audit(path)

    assert len(saved) == 1
    assert loaded.iloc[0]["notes"] == record["notes"]
    selected = filter_predictions(
        prepare_predictions(prediction_frame()),
        category="Combined mistakes",
        audit_status="Audited",
        audit=loaded,
    )
    assert selected["example_id"].tolist() == ["veto-hurt"]


def test_normalized_feature_box_and_visual_evidence(
    tmp_path: Path,
) -> None:
    frame_path = tmp_path / "frame_0000.jpg"
    Image.new("RGB", (100, 80), "black").save(frame_path)
    features_dir = tmp_path / "features"
    row = pd.Series(
        {
            "domain": "native",
            "example_id": "example",
            "view_id": "primary",
        }
    )
    artifact = feature_path(features_dir, row)
    artifact.parent.mkdir(parents=True)
    values = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    indices = {name: index for index, name in enumerate(FEATURE_NAMES)}
    for prefix, center, size in (
        ("player", (0.5, 0.5), (0.4, 0.5)),
        ("ball", (0.8, 0.2), (0.1, 0.1)),
    ):
        values[0, indices[f"{prefix}_x"]] = center[0]
        values[0, indices[f"{prefix}_y"]] = center[1]
        values[0, indices[f"{prefix}_w"]] = size[0]
        values[0, indices[f"{prefix}_h"]] = size[1]
        values[0, indices[f"{prefix}_valid"]] = 1
    metadata = {
        "feature_names": FEATURE_NAMES,
        "selected_frame_indices": [0],
    }
    np.savez_compressed(
        artifact,
        features=values,
        metadata=json.dumps(metadata),
    )

    assert normalized_feature_box(
        values[0],
        indices,
        prefix="player",
        frame_width=100,
        frame_height=80,
    ) == (30, 20, 70, 60)
    players, balls, resolved = load_feature_evidence(
        str(features_dir),
        "native",
        "example",
        "primary",
        (str(frame_path),),
    )

    assert Path(resolved) == artifact
    assert players[0]["bbox"] == [30, 20, 70, 60]
    assert balls[0]["bbox"] == [75, 12, 85, 20]
    rendered = render_frame(
        frame_path,
        observation=players[0],
        detected_ball=balls[0],
    )
    assert rendered.size == (100, 80)


def test_actor_crop_uses_cached_classifier_box(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame_0000.jpg"
    Image.new("RGB", (100, 80), "white").save(frame_path)
    payload = {
        "crop_metadata": [
            {
                "frame_index": 0,
                "bbox": [10, 20, 60, 70],
                "goalkeeper_probability": 0.75,
            }
        ]
    }

    crops = actor_crops([frame_path], payload)

    assert len(crops) == 1
    assert crops[0][0].size == (50, 50)
    assert crops[0][1]["goalkeeper_probability"] == 0.75


def test_prepare_predictions_rejects_duplicate_clip_keys() -> None:
    duplicated = pd.concat(
        [prediction_frame().iloc[[0]], prediction_frame().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepare_predictions(duplicated)
