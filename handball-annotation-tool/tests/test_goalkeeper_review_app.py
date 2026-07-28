from pathlib import Path

import pandas as pd
import pytest

from goalkeeper_review_app import (
    apply_source_labels,
    choose_next_source,
    save_manifest,
    source_summary,
    validate_manifest,
)


def candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_image": "/photos/a.jpg",
                "source_label": "goalkeeper",
                "source_group": "match-a",
                "crop_path": "/crops/a-0.jpg",
                "crop_id": "a-0",
                "detection_index": 0,
                "detection_count": 2,
                "bbox": "0,0,20,40",
                "status": "needs_review",
                "review_label": "",
            },
            {
                "source_image": "/photos/a.jpg",
                "source_label": "goalkeeper",
                "source_group": "match-a",
                "crop_path": "/crops/a-1.jpg",
                "crop_id": "a-1",
                "detection_index": 1,
                "detection_count": 2,
                "bbox": "20,0,40,40",
                "status": "needs_review",
                "review_label": "",
            },
            {
                "source_image": "/photos/b.jpg",
                "source_label": "not_goalkeeper",
                "source_group": "match-b",
                "crop_path": "/crops/b-0.jpg",
                "crop_id": "b-0",
                "detection_index": 0,
                "detection_count": 1,
                "bbox": "0,0,20,40",
                "status": "auto_accepted_single",
                "review_label": "not_goalkeeper",
            },
        ]
    )


def test_source_summary_tracks_completion_per_photo() -> None:
    summary = source_summary(candidate_frame()).set_index("source_image")

    assert summary.loc["/photos/a.jpg", "crop_count"] == 2
    assert summary.loc["/photos/a.jpg", "remaining"] == 2
    assert not bool(summary.loc["/photos/a.jpg", "completed"])
    assert bool(summary.loc["/photos/b.jpg", "completed"])


def test_apply_source_labels_does_not_change_other_photo() -> None:
    original = candidate_frame()
    updated = apply_source_labels(
        original,
        "/photos/a.jpg",
        {0: "goalkeeper", 1: "not_goalkeeper"},
    )

    assert original.loc[0, "review_label"] == ""
    assert updated.loc[0, "review_label"] == "goalkeeper"
    assert updated.loc[1, "review_label"] == "not_goalkeeper"
    assert updated.loc[0, "status"] == "reviewed"
    assert updated.loc[2].equals(original.loc[2])


def test_apply_source_labels_rejects_row_from_another_photo() -> None:
    with pytest.raises(ValueError, match="outside this source"):
        apply_source_labels(
            candidate_frame(),
            "/photos/a.jpg",
            {2: "goalkeeper"},
        )


def test_validate_manifest_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="crop_id"):
        validate_manifest(candidate_frame().drop(columns=["crop_id"]))


def test_choose_next_source_stops_at_queue_edges() -> None:
    sources = ["a", "b", "c"]

    assert choose_next_source(sources, "a") == "b"
    assert choose_next_source(sources, "c") is None
    assert choose_next_source(sources, "a", forward=False) is None
    assert choose_next_source(sources, "missing") == "a"


def test_save_manifest_creates_stable_backup_and_round_trips(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.csv"
    original = candidate_frame()
    original.to_csv(path, index=False)

    changed = original.copy()
    changed.loc[0, "review_label"] = "goalkeeper"
    save_manifest(changed, path)
    backup = path.with_suffix(".csv.before_manual_review")

    assert backup.is_file()
    assert pd.read_csv(path, keep_default_na=False).loc[
        0, "review_label"
    ] == "goalkeeper"
    assert pd.read_csv(backup, keep_default_na=False).loc[
        0, "review_label"
    ] == ""

    changed.loc[0, "review_label"] = "uncertain"
    save_manifest(changed, path)
    assert pd.read_csv(backup, keep_default_na=False).loc[
        0, "review_label"
    ] == ""
