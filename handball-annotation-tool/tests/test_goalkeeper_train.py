from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms

from training.goalkeeper_train import (
    _save_gradcam,
    _save_mistakes,
    binary_metrics,
    load_config,
    select_abstaining_thresholds,
)


def test_training_config_has_separate_validation_and_test_folds():
    config = load_config("configs/goalkeeper_classifier.yaml")
    assert config.validation_fold != config.test_fold
    assert config.input_size == 224
    assert config.pretrained is True


def test_metrics_and_threshold_selection():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.02, 0.10, 0.30, 0.70, 0.90, 0.98])
    metrics = binary_metrics(labels, probabilities)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    low, high = select_abstaining_thresholds(
        labels, probabilities, target_precision=0.8
    )
    assert 0 <= low < high <= 1


def test_gradcam_writes_attention_overlay(tmp_path: Path):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(4, 2)

        def forward(self, value):
            value = self.features(value).mean(dim=(2, 3))
            return self.classifier(value)

    source = tmp_path / "player.jpg"
    assert cv2.imwrite(
        str(source), np.full((48, 32, 3), 127, dtype=np.uint8)
    )
    destination = tmp_path / "gradcam.jpg"
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    _save_gradcam(
        TinyModel(), source, transform, "cpu", destination
    )
    rendered = cv2.imread(str(destination))
    assert rendered is not None
    assert rendered.shape[:2] == (48, 32)


def test_save_mistakes_removes_files_from_previous_run(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "player.jpg"
    assert cv2.imwrite(
        str(source), np.full((32, 24, 3), 127, dtype=np.uint8)
    )
    destination = tmp_path / "reviews"
    destination.mkdir()
    stale = destination / "stale-old-run.jpg"
    stale.write_bytes(b"old")
    predictions = pd.DataFrame(
        [
            {
                "crop_path": str(source),
                "review_label": "goalkeeper",
                "goalkeeper_probability": 0.1,
                "correct": False,
            }
        ]
    )

    def fake_gradcam(_model, _source, _transform, _device, output):
        output.write_bytes(b"gradcam")

    monkeypatch.setattr(
        "training.goalkeeper_train._save_gradcam", fake_gradcam
    )
    _save_mistakes(
        predictions,
        destination,
        model=object(),
        transform=None,
        device="cpu",
    )

    assert not stale.exists()
    assert len(list(destination.glob("true-*.jpg"))) == 1
    assert len(list(destination.glob("gradcam_*.jpg"))) == 1
