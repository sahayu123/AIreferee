from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import training.inference as inference_module
from training.features import FEATURE_NAMES


def _prepare_fake_inference(
    monkeypatch,
    tmp_path: Path,
    probability: float,
) -> tuple[Path, Path]:
    candidate = tmp_path / "candidate"
    frames = candidate / "frames"
    frames.mkdir(parents=True)
    assert cv2.imwrite(
        str(frames / "frame_0000.jpg"),
        np.zeros((24, 32, 3), dtype=np.uint8),
    )
    checkpoint_path = tmp_path / "model.pt"
    checkpoint_path.write_bytes(b"test checkpoint")
    logit = math.log(probability / (1.0 - probability))

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def load_state_dict(self, state):
            pass

        def eval(self):
            return self

        def __call__(self, features):
            return torch.tensor([logit], dtype=torch.float32)

    class FakeExtractor:
        def __init__(self, config):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract(self, frame_paths):
            features = np.zeros((12, len(FEATURE_NAMES)), dtype=np.float32)
            overlays = [
                np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(12)
            ]
            return features, overlays, list(range(12))

    checkpoint = {
        "feature_names": FEATURE_NAMES,
        "model_config": {},
        "model": {},
        "mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        "std": np.ones(len(FEATURE_NAMES), dtype=np.float32),
    }
    monkeypatch.setattr(inference_module, "TemporalGRU", FakeModel)
    monkeypatch.setattr(inference_module, "FeatureExtractor", FakeExtractor)
    monkeypatch.setattr(
        inference_module, "load_feature_config", lambda path: object()
    )
    monkeypatch.setattr(
        inference_module,
        "load_train_config",
        lambda path: SimpleNamespace(device="cpu"),
    )
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        lambda *args, **kwargs: checkpoint,
    )
    return candidate, checkpoint_path


def test_goalkeeper_option_cannot_change_handball_result(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.8
    )
    observed: dict[str, float] = {}

    def fake_goalkeeper_stage(
        frames,
        features,
        selected,
        metadata,
        handball_probability,
        threshold,
        config,
    ):
        observed["probability"] = handball_probability
        return {
            "evaluated": True,
            "status": "unknown",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": 0.4,
            "reason": "test",
            "actor_observations": [],
        }

    monkeypatch.setattr(
        inference_module,
        "load_jersey_glove_config",
        lambda path: object(),
    )
    monkeypatch.setattr(
        inference_module,
        "classify_goalkeeper_after_handball",
        fake_goalkeeper_stage,
    )
    baseline = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "baseline.json",
    )
    augmented = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "augmented.json",
        jersey_glove_config_path="goalkeeper.yaml",
    )
    assert observed["probability"] == baseline["handball_probability"]
    assert augmented["handball_probability"] == baseline["handball_probability"]
    assert augmented["predicted_label"] == baseline["predicted_label"]
    assert augmented["threshold"] == baseline["threshold"]


def test_not_handball_skips_goalkeeper_config_and_models(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.2
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("goalkeeper config/models must not load")

    monkeypatch.setattr(
        inference_module, "load_jersey_glove_config", forbidden
    )
    monkeypatch.setattr(
        inference_module, "classify_goalkeeper_after_handball", forbidden
    )
    result = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "negative.json",
        jersey_glove_config_path="goalkeeper.yaml",
    )
    assert result["predicted_label"] == "not_handball"
    assert result["goalkeeper_status"] == "not_evaluated"
    assert result["goalkeeper_analysis"]["reason"] == "handball_below_threshold"
