from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import training.combined_evaluate as combined_module
from training.combined_evaluate import (
    CachedRoleAnalyzer,
    RoleInputs,
    SupervisedGoalkeeperRuntime,
    evaluate_combined_rows,
    fuse_final_decision,
    hard_binary_metrics,
    load_oof_probabilities,
    summarize_evaluation,
)
from training.config import TrainConfig
from training.data import FeatureView
from training.features import FEATURE_NAMES
from training.supervised_goalkeeper import (
    load_supervised_goalkeeper_config,
)


def test_supervised_runtime_reuses_tracker_and_classifier(monkeypatch):
    created = {"tracker": 0, "classifier": 0, "calls": 0}

    class Tracker:
        def __init__(self, config):
            created["tracker"] += 1

    class Classifier:
        def __init__(self, *args, **kwargs):
            created["classifier"] += 1

    def classify(*args, **kwargs):
        created["calls"] += 1
        assert isinstance(kwargs["tracker"], Tracker)
        assert isinstance(kwargs["classifier"], Classifier)
        return {
            "status": "unknown",
            "is_goalkeeper": None,
            "evaluated": True,
        }

    monkeypatch.setattr(combined_module, "YOLOPersonTracker", Tracker)
    monkeypatch.setattr(combined_module, "GoalkeeperClassifier", Classifier)
    monkeypatch.setattr(
        combined_module, "classify_supervised_goalkeeper", classify
    )
    runtime = SupervisedGoalkeeperRuntime(
        load_supervised_goalkeeper_config(
            "configs/supervised_goalkeeper.yaml"
        )
    )
    inputs = RoleInputs(
        features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        selected_indices=[0],
        frame_paths=[Path("frame.jpg")],
        metadata={},
        source_fingerprint="source",
        feature_artifact=Path("features.npz"),
    )
    runtime.analyze(inputs, 0.8, 0.5)
    runtime.analyze(inputs, 0.2, 0.5)
    assert created == {"tracker": 1, "classifier": 1, "calls": 2}


@pytest.mark.parametrize(
    (
        "probability",
        "role_result",
        "expected_final",
        "expected_veto",
        "expected_fallback",
    ),
    [
        (
            0.8,
            {
                "status": "goalkeeper",
                "is_goalkeeper": True,
                "execution_status": "completed",
            },
            0,
            True,
            False,
        ),
        (
            0.8,
            {
                "status": "not_goalkeeper",
                "is_goalkeeper": False,
                "execution_status": "completed",
            },
            1,
            False,
            False,
        ),
        (
            0.8,
            {
                "status": "unknown",
                "is_goalkeeper": None,
                "execution_status": "completed",
            },
            1,
            False,
            True,
        ),
        (
            0.8,
            {"status": "error", "execution_status": "error"},
            1,
            False,
            True,
        ),
        (
            0.2,
            {
                "status": "goalkeeper",
                "is_goalkeeper": True,
                "execution_status": "completed",
            },
            0,
            False,
            False,
        ),
        (
            0.2,
            {
                "status": "unknown",
                "is_goalkeeper": None,
                "execution_status": "completed",
            },
            0,
            False,
            False,
        ),
    ],
)
def test_final_decision_matrix(
    probability,
    role_result,
    expected_final,
    expected_veto,
    expected_fallback,
):
    decision = fuse_final_decision(probability, 0.5, role_result)
    assert decision["final_prediction"] == expected_final
    assert decision["goalkeeper_veto"] is expected_veto
    assert decision["role_fallback"] is expected_fallback


@pytest.mark.parametrize(
    "role_result",
    [
        {"status": "goalkeeper", "is_goalkeeper": None},
        {"status": "goalkeeper", "is_goalkeeper": False},
        {"status": "goalkeeper", "is_goalkeeper": "true"},
        {"status": "unexpected", "is_goalkeeper": True},
    ],
)
def test_malformed_goalkeeper_output_never_vetoes(role_result):
    decision = fuse_final_decision(0.8, 0.5, role_result)
    assert decision["goalkeeper_status"] == "error"
    assert decision["goalkeeper_veto"] is False
    assert decision["final_prediction"] == 1
    assert decision["role_fallback"] is True


def test_only_raw_positive_rows_invoke_goalkeeper_before_final_decision():
    rows = pd.DataFrame(
        [
            {
                "example_id": "raw-negative",
                "view_id": "primary",
                "label": 0,
                "domain": "native",
                "fold": 0,
                "raw_handball_probability": 0.2,
                "checkpoint_sha256": "fold0",
            },
            {
                "example_id": "keeper-veto",
                "view_id": "primary",
                "label": 0,
                "domain": "native",
                "fold": 1,
                "raw_handball_probability": 0.8,
                "checkpoint_sha256": "fold1",
            },
            {
                "example_id": "unknown-fallback",
                "view_id": "primary",
                "label": 1,
                "domain": "native",
                "fold": 2,
                "raw_handball_probability": 0.8,
                "checkpoint_sha256": "fold2",
            },
            {
                "example_id": "error-fallback",
                "view_id": "primary",
                "label": 1,
                "domain": "native",
                "fold": 3,
                "raw_handball_probability": 0.9,
                "checkpoint_sha256": "fold3",
            },
        ]
    )
    statuses = {
        "keeper-veto": "goalkeeper",
        "unknown-fallback": "unknown",
    }
    calls: list[tuple[str, float, float, str]] = []
    progress: list[str] = []

    def analyzer(row, probability, threshold, checkpoint_sha256):
        example_id = str(row["example_id"])
        calls.append(
            (example_id, probability, threshold, checkpoint_sha256)
        )
        if example_id == "error-fallback":
            raise RuntimeError("test role failure")
        return {
            "evaluated": True,
            "execution_status": "completed",
            "status": statuses[example_id],
            "is_goalkeeper": (
                True
                if statuses[example_id] == "goalkeeper"
                else False
                if statuses[example_id] == "not_goalkeeper"
                else None
            ),
        }

    predictions = evaluate_combined_rows(
        rows,
        analyzer,
        progress=lambda current, total, row: progress.append(
            str(row["example_id"])
        ),
    )
    assert [item[0] for item in calls] == [
        "keeper-veto",
        "unknown-fallback",
        "error-fallback",
    ]
    assert len(calls) == 3
    assert progress == rows["example_id"].tolist()
    assert predictions["raw_prediction"].tolist() == [0, 1, 1, 1]
    assert predictions["final_prediction"].tolist() == [0, 0, 1, 1]
    assert predictions["goalkeeper_status"].tolist() == [
        "not_evaluated",
        "goalkeeper",
        "unknown",
        "error",
    ]
    assert predictions["goalkeeper_veto"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert predictions["role_fallback"].tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert predictions["goalkeeper_analysis_invoked"].tolist() == [
        False,
        True,
        True,
        True,
    ]
    assert predictions["combined_event_label"].tolist() == [
        "not_handball",
        "handball_goalkeeper",
        "handball_actor_unknown",
        "handball_actor_unknown",
    ]
    assert predictions.loc[3, "role_error_type"] == "RuntimeError"


def test_handball_threshold_boundary_controls_goalkeeper_gate():
    rows = pd.DataFrame(
        [
            {
                "example_id": name,
                "view_id": "primary",
                "label": int(probability >= 0.5),
                "raw_handball_probability": probability,
                "checkpoint_sha256": "checkpoint",
            }
            for name, probability in (
                ("below", 0.4999),
                ("at", 0.5),
                ("above", 0.8),
            )
        ]
    )
    calls: list[str] = []

    def analyzer(row, *_args):
        calls.append(str(row["example_id"]))
        return {
            "evaluated": True,
            "execution_status": "completed",
            "status": "not_goalkeeper",
            "is_goalkeeper": False,
        }

    predictions = evaluate_combined_rows(rows, analyzer, threshold=0.5)

    assert calls == ["at", "above"]
    assert predictions["goalkeeper_analysis_invoked"].tolist() == [
        False,
        True,
        True,
    ]
    assert predictions["goalkeeper_status"].tolist() == [
        "not_evaluated",
        "not_goalkeeper",
        "not_goalkeeper",
    ]


def test_role_cache_reuses_only_matching_source_checkpoint_and_probability(
    tmp_path: Path,
):
    artifact = tmp_path / "features.npz"
    artifact.write_bytes(b"feature identity")
    inputs = RoleInputs(
        features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        selected_indices=[0],
        frame_paths=[],
        metadata={},
        source_fingerprint="source-one",
        feature_artifact=artifact,
    )
    analysis_calls: list[float] = []

    def analyze_uncached(role_inputs, probability, threshold):
        assert role_inputs is inputs
        analysis_calls.append(probability)
        return {
            "schema_version": 1,
            "evaluated": True,
            "status": "unknown",
            "is_goalkeeper": None,
            "goalkeeper_evidence_score": 0.4,
            "handball_probability_observed": probability,
            "handball_threshold_observed": threshold,
        }

    analyzer = CachedRoleAnalyzer(
        tmp_path / "cache",
        "role-config-fingerprint",
        lambda row: inputs,
        analyze_uncached,
    )
    row = pd.Series(
        {
            "domain": "native",
            "example_id": "example",
            "view_id": "primary",
        }
    )
    first = analyzer(row, 0.8, 0.5, "checkpoint-one")
    second = analyzer(row, 0.8, 0.5, "checkpoint-one")
    changed_checkpoint = analyzer(row, 0.8, 0.5, "checkpoint-two")
    assert analysis_calls == [0.8, 0.8]
    assert first["_combined_cache_hit"] is False
    assert second["_combined_cache_hit"] is True
    assert changed_checkpoint["_combined_cache_hit"] is False
    assert Path(str(first["_combined_cache_path"])).is_file()


def test_cache_write_failure_becomes_row_level_role_error(
    tmp_path: Path, monkeypatch
):
    artifact = tmp_path / "features.npz"
    artifact.write_bytes(b"feature identity")
    inputs = RoleInputs(
        features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        selected_indices=[0],
        frame_paths=[],
        metadata={},
        source_fingerprint="source-one",
        feature_artifact=artifact,
    )
    analyzer = CachedRoleAnalyzer(
        tmp_path / "cache",
        "role-config-fingerprint",
        lambda row: inputs,
        lambda role_inputs, probability, threshold: {
            "schema_version": 1,
            "evaluated": True,
            "status": "goalkeeper",
            "is_goalkeeper": True,
            "goalkeeper_evidence_score": 0.9,
        },
    )
    monkeypatch.setattr(
        "training.combined_evaluate.save_jersey_glove_result",
        lambda result, destination: (_ for _ in ()).throw(
            OSError("cache unavailable")
        ),
    )
    result = analyzer(
        pd.Series(
            {
                "domain": "native",
                "example_id": "example",
                "view_id": "primary",
            }
        ),
        0.8,
        0.5,
        "checkpoint-one",
    )
    decision = fuse_final_decision(0.8, 0.5, result)
    assert result["reason"] == "role_cache_write_error"
    assert result["error_type"] == "OSError"
    assert decision["goalkeeper_status"] == "error"
    assert decision["goalkeeper_veto"] is False
    assert decision["final_prediction"] == 1


def test_load_oof_probabilities_uses_all_five_matching_fold_checkpoints(
    tmp_path: Path,
):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    for fold in range(5):
        (checkpoints / f"gru_fold{fold}_best.pt").write_bytes(
            f"fold-{fold}".encode()
        )
    config = TrainConfig(
        manifest=tmp_path / "manifest.csv",
        features_dir=tmp_path / "features",
        checkpoints_dir=checkpoints,
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        device="cpu",
        seed=42,
        folds=5,
        fold=0,
        epochs=1,
        batch_size=2,
        hidden_size=2,
        layers=1,
        dropout=0.0,
        learning_rate=0.001,
        weight_decay=0.0,
        patience=1,
    )
    manifest = pd.DataFrame(
        [
            {
                "example_id": f"example-{fold}",
                "view_id": "primary",
                "fold": fold,
                "_manifest_order": fold,
            }
            for fold in range(5)
        ]
    )
    views = [
        FeatureView(
            example_id=f"example-{fold}",
            view_id="primary",
            label=fold % 2,
            domain="native",
            fold=fold,
            path=tmp_path / f"fold-{fold}.npz",
            features=np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32),
        )
        for fold in range(5)
    ]
    loaded_folds: list[int] = []

    def checkpoint_loader(path, device):
        fold = int(path.stem.split("fold")[1].split("_")[0])
        loaded_folds.append(fold)
        return {
            "fold": fold,
            "feature_names": FEATURE_NAMES,
            "model_config": {},
            "model": {},
            "mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            "std": np.ones(len(FEATURE_NAMES), dtype=np.float32),
        }

    class FakeModel:
        def to(self, device):
            return self

        def load_state_dict(self, state):
            return None

    def predictor(model, fold_views, mean, std, device, batch_size):
        return pd.DataFrame(
            [
                {
                    "example_id": view.example_id,
                    "view_id": view.view_id,
                    "label": view.label,
                    "domain": view.domain,
                    "probability": 0.1 + 0.1 * view.fold,
                }
                for view in fold_views
            ]
        )

    result = load_oof_probabilities(
        config,
        manifest,
        view_loader=lambda manifest_path, features_dir, require_all: views,
        checkpoint_loader=checkpoint_loader,
        model_factory=lambda **kwargs: FakeModel(),
        predictor=predictor,
        device_resolver=lambda requested: "cpu",
    )
    assert loaded_folds == [0, 1, 2, 3, 4]
    assert result.predictions["oof_fold"].tolist() == [0, 1, 2, 3, 4]
    assert result.predictions["raw_handball_probability"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    assert set(result.checkpoint_fingerprints) == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }


def test_metrics_include_accuracy_confusion_deltas_and_all_rows():
    metrics = hard_binary_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 0],
    )
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

    predictions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "raw_prediction": [0, 1, 1, 0],
            "final_prediction": [0, 0, 1, 0],
            "goalkeeper_status": [
                "not_evaluated",
                "goalkeeper",
                "unknown",
                "not_evaluated",
            ],
            "goalkeeper_status_raw": [
                "not_evaluated",
                "goalkeeper",
                "unknown",
                "not_evaluated",
            ],
            "goalkeeper_veto": [False, True, False, False],
            "role_fallback": [False, False, True, False],
            "role_error": [False, False, False, False],
            "role_cache_hit": [False, True, False, False],
            "goalkeeper_analysis_invoked": [False, True, True, False],
            "goalkeeper_evaluated": [False, True, True, False],
            "combined_event_label": [
                "not_handball",
                "handball_goalkeeper",
                "handball_actor_unknown",
                "not_handball",
            ],
            "raw_handball_probability": [0.1, 0.9, 0.8, 0.2],
        }
    )
    summary = summarize_evaluation(
        predictions,
        threshold=0.5,
        fingerprints={"glove_enabled": False},
    )
    combined = summary["metrics"]["combined_final"]
    delta = summary["metrics"]["combined_minus_baseline"]
    assert combined["accuracy"] == pytest.approx(0.75)
    assert combined["confusion_matrix"] == [[2, 0], [1, 1]]
    assert delta["accuracy"] == pytest.approx(0.25)
    assert delta["confusion_matrix"] == [[1, -1], [0, 0]]
    assert summary["counts"]["examples"] == 4
    assert summary["counts"]["goalkeeper_vetoes"] == 1
    assert summary["counts"]["role_fallbacks"] == 1
    assert summary["counts"]["role_errors"] == 0
    assert summary["counts"]["goalkeeper_analysis_skipped"] == 2
    assert summary["completion"]["goalkeeper_analysis_invocations"] == 2
    assert summary["completion"]["goalkeeper_analysis_skipped"] == 2
    assert summary["completion"]["goalkeeper_analysis_completed"] == 2
    assert summary["completion"]["final_prediction_rows"] == 4
