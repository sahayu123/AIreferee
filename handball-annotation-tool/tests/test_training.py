import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from training.data import aggregate_sequence
from training.features import FEATURE_NAMES, FeatureExtractor
from training.gru import TemporalGRU
from training.inference import resolve_frames
from training.manifest import _imported_rows, sorted_frames


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((16, 16, 3), dtype=np.uint8))


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

