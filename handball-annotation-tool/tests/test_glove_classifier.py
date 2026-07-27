from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import training.glove_classifier as glove_module
from training.glove_classifier import (
    ARCHITECTURE,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    CLASS_NAMES,
    GloveClassifier,
    build_glove_model,
    create_glove_checkpoint,
    preprocess_bgr_crops,
    save_glove_checkpoint,
)


@pytest.fixture(scope="module")
def deterministic_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = build_glove_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier[-1].bias.copy_(
            torch.tensor([0.0, math.log(3.0)], dtype=torch.float32)
        )
    return save_glove_checkpoint(
        tmp_path_factory.mktemp("glove-model") / "glove.pt",
        model,
        input_size=(32, 32),
    )


def test_checkpoint_schema_is_explicit_and_versioned():
    model = build_glove_model()
    payload = create_glove_checkpoint(model, input_size=(32, 48))
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["architecture"] == ARCHITECTURE
    assert payload["class_names"] == list(CLASS_NAMES)
    assert payload["input_size"] == [32, 48]


def test_preprocess_converts_bgr_uint8_to_rgb():
    blue_bgr = np.zeros((3, 2, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255
    output = preprocess_bgr_crops(
        [blue_bgr],
        (16, 16),
        normalization_mean=(0.0, 0.0, 0.0),
        normalization_std=(1.0, 1.0, 1.0),
    )
    assert output.shape == (1, 3, 16, 16)
    assert torch.allclose(output[:, 0], torch.zeros((1, 16, 16)))
    assert torch.allclose(output[:, 1], torch.zeros((1, 16, 16)))
    assert torch.allclose(output[:, 2], torch.ones((1, 16, 16)))


def test_inference_is_batched_deterministic_and_offline(
    deterministic_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    load_arguments: list[dict[str, object]] = []
    model_arguments: list[object] = []
    original_load = glove_module.torch.load
    original_builder = glove_module.mobilenet_v3_small

    def load_spy(*args, **kwargs):
        load_arguments.append(dict(kwargs))
        return original_load(*args, **kwargs)

    def model_spy(*args, **kwargs):
        model_arguments.append(kwargs.get("weights"))
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(glove_module.torch, "load", load_spy)
    monkeypatch.setattr(glove_module, "mobilenet_v3_small", model_spy)
    classifier = GloveClassifier(
        deterministic_checkpoint,
        batch_size=2,
        torch_threads=1,
    )

    crops = [
        np.zeros((8, 10, 3), dtype=np.uint8),
        np.full((11, 7, 3), 255, dtype=np.uint8),
        np.full((5, 5, 3), 127, dtype=np.uint8),
    ]
    probabilities = classifier.predict_proba(crops)
    assert probabilities.shape == (3, 2)
    assert probabilities.dtype == np.float32
    assert np.allclose(probabilities, [[0.25, 0.75]] * 3, atol=1e-6)
    assert np.allclose(
        classifier.predict_glove_probability(crops),
        [0.75, 0.75, 0.75],
        atol=1e-6,
    )
    assert load_arguments == [{"map_location": "cpu", "weights_only": True}]
    assert model_arguments == [None]


def test_empty_batch_has_well_defined_probability_shape(
    deterministic_checkpoint: Path,
):
    classifier = GloveClassifier(deterministic_checkpoint, torch_threads=1)
    assert classifier.predict_proba([]).shape == (0, 2)
    assert classifier.predict_glove_probability([]).shape == (0,)


def test_missing_and_invalid_checkpoints_have_clear_errors(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Glove classifier checkpoint not found"):
        GloveClassifier(tmp_path / "missing.pt")

    invalid = tmp_path / "invalid.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture": ARCHITECTURE,
            "class_names": ["goalkeeper_glove", "bare_hand"],
            "input_size": [32, 32],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "state_dict": {"not_a_model": torch.zeros(1)},
        },
        invalid,
    )
    with pytest.raises(ValueError, match="class_names"):
        GloveClassifier(invalid)


@pytest.mark.parametrize(
    ("crop", "message"),
    [
        (np.zeros((4, 4, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((4, 4), dtype=np.uint8), r"shape \[height, width, 3\]"),
        (np.zeros((0, 4, 3), dtype=np.uint8), "must not be empty"),
    ],
)
def test_invalid_crop_inputs_raise_clear_errors(crop: np.ndarray, message: str):
    with pytest.raises(ValueError, match=message):
        preprocess_bgr_crops([crop], (32, 32))

    single_image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="wrap one image in a list"):
        preprocess_bgr_crops(single_image, (32, 32))
