from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
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


def test_confirmed_goalkeeper_vetoes_raw_handball_result(
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
        force_evaluation=False,
    ):
        observed["probability"] = handball_probability
        assert force_evaluation is True
        return {
            "evaluated": True,
            "status": "goalkeeper",
            "is_goalkeeper": True,
            "goalkeeper_evidence_score": 0.9,
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
    assert augmented["raw_predicted_label"] == baseline["predicted_label"]
    assert augmented["predicted_label"] == "not_handball"
    assert augmented["goalkeeper_veto_applied"] is True
    assert augmented["final_decision_reason"] == "confirmed_goalkeeper_veto"
    assert augmented["threshold"] == baseline["threshold"]


def test_raw_not_handball_skips_jersey_goalkeeper_analysis(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.2
    )

    observed = {"loaded": 0, "evaluated": 0}

    monkeypatch.setattr(
        inference_module,
        "load_jersey_glove_config",
        lambda path: observed.__setitem__("loaded", observed["loaded"] + 1)
        or object(),
    )

    def fake_goalkeeper_stage(*args, **kwargs):
        observed["evaluated"] += 1
        assert kwargs["force_evaluation"] is True
        return {
            "evaluated": True,
            "status": "not_goalkeeper",
            "is_goalkeeper": False,
            "goalkeeper_evidence_score": 0.1,
            "reason": "field_team_jersey_match",
            "actor_observations": [],
        }

    monkeypatch.setattr(
        inference_module,
        "classify_goalkeeper_after_handball",
        fake_goalkeeper_stage,
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
    assert result["raw_predicted_label"] == "not_handball"
    assert result["combined_event_label"] == "not_handball"
    assert result["goalkeeper_status"] == "not_evaluated"
    assert result["goalkeeper_analysis_required"] is False
    assert result["goalkeeper_analysis_invoked"] is False
    assert result["goalkeeper_analysis_completed"] is False
    assert result["goalkeeper_veto_applied"] is False
    assert result["final_decision_reason"] == (
        "raw_not_handball_goalkeeper_skipped"
    )
    assert observed == {"loaded": 0, "evaluated": 0}


def test_unknown_goalkeeper_status_preserves_raw_handball(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.8
    )
    monkeypatch.setattr(
        inference_module,
        "load_jersey_glove_config",
        lambda path: object(),
    )
    monkeypatch.setattr(
        inference_module,
        "classify_goalkeeper_after_handball",
        lambda *args, **kwargs: {
            "evaluated": True,
            "status": "unknown",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": 0.4,
            "reason": "uncertain",
            "actor_observations": [],
        },
    )
    result = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "unknown.json",
        jersey_glove_config_path="goalkeeper.yaml",
    )
    assert result["raw_predicted_label"] == "handball"
    assert result["predicted_label"] == "handball"
    assert result["combined_event_label"] == "handball_actor_unknown"
    assert result["handball_actor_role"] == "unknown"
    assert result["goalkeeper_veto_applied"] is False
    assert result["final_decision_reason"] == "raw_handball_preserved_unknown"


@pytest.mark.parametrize(
    "malformed_result",
    [
        None,
        {},
        {"status": "maybe", "is_goalkeeper": None},
        {"status": "goalkeeper", "is_goalkeeper": False},
        {
            "status": "goalkeeper",
            "is_goalkeeper": True,
            "actor_observations": "not-a-list",
        },
    ],
)
def test_malformed_goalkeeper_result_safely_preserves_raw_handball(
    monkeypatch,
    tmp_path: Path,
    malformed_result,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.8
    )
    monkeypatch.setattr(
        inference_module,
        "load_jersey_glove_config",
        lambda path: object(),
    )
    monkeypatch.setattr(
        inference_module,
        "classify_goalkeeper_after_handball",
        lambda *args, **kwargs: malformed_result,
    )
    output = tmp_path / "malformed.json"
    with pytest.warns(RuntimeWarning):
        result = inference_module.infer(
            candidate,
            checkpoint,
            "feature.yaml",
            "train.yaml",
            output,
            jersey_glove_config_path="goalkeeper.yaml",
        )
    assert output.is_file()
    assert result["raw_predicted_label"] == "handball"
    assert result["predicted_label"] == "handball"
    assert result["goalkeeper_status"] == "unavailable"
    assert result["goalkeeper_veto_applied"] is False


def test_supervised_goalkeeper_classifier_can_veto_raw_handball(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.8
    )
    observed = {"called": 0}
    monkeypatch.setattr(
        inference_module,
        "load_supervised_goalkeeper_config",
        lambda path: object(),
    )

    def fake_supervised_stage(*args, **kwargs):
        observed["called"] += 1
        return {
            "evaluated": True,
            "status": "goalkeeper",
            "is_goalkeeper": True,
            "goalkeeper_evidence_score": 0.92,
            "reason": "trained_player_crop_classifier",
            "actor_observations": [],
        }

    monkeypatch.setattr(
        inference_module,
        "classify_supervised_goalkeeper",
        fake_supervised_stage,
    )
    result = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "supervised.json",
        supervised_goalkeeper_config_path="supervised.yaml",
    )
    assert observed["called"] == 1
    assert result["raw_predicted_label"] == "handball"
    assert result["predicted_label"] == "not_handball"
    assert result["combined_event_label"] == "handball_goalkeeper"
    assert result["handball_actor_role"] == "goalkeeper"
    assert result["goalkeeper_analysis_required"] is True
    assert result["goalkeeper_analysis_invoked"] is True
    assert result["goalkeeper_analysis_completed"] is True
    assert result["goalkeeper_veto_applied"] is True
    assert (
        result["goalkeeper_detection_backend"]
        == "supervised_player_crop_actor_track"
    )


def test_raw_not_handball_skips_supervised_goalkeeper_analysis(
    monkeypatch,
    tmp_path: Path,
):
    candidate, checkpoint = _prepare_fake_inference(
        monkeypatch, tmp_path, probability=0.2
    )
    observed = {"loaded": 0, "evaluated": 0}

    def forbidden_loader(*args, **kwargs):
        observed["loaded"] += 1
        raise AssertionError("raw negatives must not load goalkeeper config")

    def forbidden_classifier(*args, **kwargs):
        observed["evaluated"] += 1
        raise AssertionError("raw negatives must not run goalkeeper model")

    monkeypatch.setattr(
        inference_module,
        "load_supervised_goalkeeper_config",
        forbidden_loader,
    )
    monkeypatch.setattr(
        inference_module,
        "classify_supervised_goalkeeper",
        forbidden_classifier,
    )
    result = inference_module.infer(
        candidate,
        checkpoint,
        "feature.yaml",
        "train.yaml",
        tmp_path / "supervised-negative.json",
        supervised_goalkeeper_config_path="supervised.yaml",
    )

    assert observed == {"loaded": 0, "evaluated": 0}
    assert result["raw_predicted_label"] == "not_handball"
    assert result["predicted_label"] == "not_handball"
    assert result["combined_event_label"] == "not_handball"
    assert result["handball_actor_role"] is None
    assert result["goalkeeper_status"] == "not_evaluated"
    assert result["goalkeeper_analysis_required"] is False
    assert result["goalkeeper_analysis_invoked"] is False
    assert result["goalkeeper_analysis_completed"] is False
