from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from combined_pipeline.decision import (
    DecisionThresholds,
    combine_decisions,
)
from combined_pipeline.foul_notebook import (
    _REQUIRED_MODULES,
    extract_notebook_package,
)
from combined_pipeline.handball import HandballSpecialistConfig
from combined_pipeline.foul_notebook import GeneralFoulSpecialistConfig
from combined_pipeline.pipeline import (
    ExecutionConfig,
    ParallelPipelineConfig,
    ParallelRefereePipeline,
)
from combined_pipeline.schemas import (
    FinalLabel,
    GeneralFoulResult,
    HandballResult,
    PreflightReport,
    SpecialistStatus,
    VideoContext,
)
from combined_pipeline.video import prepare_video


def handball(probability: float, quality: float = 1.0) -> HandballResult:
    return HandballResult(
        status=SpecialistStatus.COMPLETED,
        probability=probability,
        predicted_label=(
            "handball" if probability >= 0.5 else "not_handball"
        ),
        quality=quality,
    )


def foul(probability: float, quality: float = 1.0) -> GeneralFoulResult:
    return GeneralFoulResult(
        status=SpecialistStatus.COMPLETED,
        probability=probability,
        predicted_label="foul" if probability >= 0.5 else "not_foul",
        quality=quality,
    )


@pytest.mark.parametrize(
    ("handball_probability", "foul_probability", "expected"),
    [
        (0.90, 0.90, FinalLabel.HANDBALL),
        (0.20, 0.90, FinalLabel.OTHER_FOUL),
        (0.20, 0.10, FinalLabel.NO_FOUL),
        (0.50, 0.90, FinalLabel.NEEDS_REVIEW),
        (0.20, 0.50, FinalLabel.NEEDS_REVIEW),
    ],
)
def test_decision_table(
    handball_probability: float,
    foul_probability: float,
    expected: FinalLabel,
) -> None:
    label, _, _, partial = combine_decisions(
        handball(handball_probability),
        foul(foul_probability),
        DecisionThresholds(),
    )
    assert label == expected
    assert partial is False


def test_handball_high_survives_unavailable_general_foul() -> None:
    label, confidence, reason, partial = combine_decisions(
        handball(0.82),
        GeneralFoulResult.unavailable("weights absent"),
        DecisionThresholds(),
    )
    assert label == FinalLabel.HANDBALL
    assert confidence == pytest.approx(0.82)
    assert reason == "handball_specialist_high_confidence"
    assert partial is True


def test_low_quality_specialist_is_not_treated_as_decisive() -> None:
    label, _, reason, partial = combine_decisions(
        handball(0.99, quality=0.05),
        foul(0.15),
        DecisionThresholds(minimum_quality=0.20),
    )
    assert label == FinalLabel.NEEDS_REVIEW
    assert reason == "one_or_more_specialists_low_quality"
    assert partial is False


def test_general_foul_needs_review_cannot_be_promoted_to_other_foul() -> None:
    uncertain_foul = GeneralFoulResult(
        status=SpecialistStatus.COMPLETED,
        probability=0.95,
        predicted_label="needs_review",
        quality=1.0,
    )
    label, _, reason, partial = combine_decisions(
        handball(0.10),
        uncertain_foul,
        DecisionThresholds(),
    )
    assert label == FinalLabel.NEEDS_REVIEW
    assert reason == "general_foul_specialist_requested_review"
    assert partial is False


def _synthetic_notebook(path: Path, *, unsafe: bool = False) -> None:
    cells = []
    names = sorted(_REQUIRED_MODULES)
    for index, name in enumerate(names):
        declared = "../escape.py" if unsafe and index == 0 else f"airef/{name}"
        cells.append(
            {
                "cell_type": "code",
                "source": [
                    f"%%writefile {declared}\n",
                    f'MODULE_NAME = "{name}"\n',
                ],
            }
        )
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")


def test_notebook_modules_are_extracted_to_ignored_runtime(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "prototype.ipynb"
    runtime = tmp_path / "runtime"
    _synthetic_notebook(notebook)
    outputs = extract_notebook_package(notebook, runtime)
    assert set(outputs) == _REQUIRED_MODULES
    assert (runtime / "airef" / "__init__.py").is_file()
    assert (
        json.loads(
            (runtime / "airef" / ".source.json").read_text()
        )["modules"]
        == sorted(_REQUIRED_MODULES)
    )


def test_notebook_extractor_rejects_path_traversal(tmp_path: Path) -> None:
    notebook = tmp_path / "unsafe.ipynb"
    _synthetic_notebook(notebook, unsafe=True)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_notebook_package(notebook, tmp_path / "runtime")
    assert not (tmp_path / "escape.py").exists()


def _make_video(path: Path, *, frames: int = 100, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (80, 48),
    )
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((48, 80, 3), index % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()


def test_prepare_video_centers_same_41_frame_window(tmp_path: Path) -> None:
    source = tmp_path / "source.avi"
    _make_video(source)
    context = prepare_video(
        source,
        tmp_path / "prepared",
        max_frames=41,
        incident_time_seconds=5.0,
    )
    assert len(context.frame_paths) == 41
    assert context.source_start_frame == 30
    assert context.frame_paths[0].name == "frame_000030.jpg"
    assert context.frame_paths[-1].name == "frame_000070.jpg"
    assert context.source_video is not None
    assert context.source_video.is_file()


class FakeSpecialist:
    def __init__(self, name: str, result, barrier: threading.Barrier):
        self.name = name
        self.result = result
        self.barrier = barrier

    def preflight(self) -> PreflightReport:
        return PreflightReport(self.name, True)

    def predict(self, context, output_dir, progress=None):
        del context, output_dir
        if progress:
            progress(f"{self.name}: started")
        self.barrier.wait(timeout=2)
        return self.result


def _dummy_config(tmp_path: Path) -> ParallelPipelineConfig:
    placeholder = tmp_path / "placeholder"
    return ParallelPipelineConfig(
        execution=ExecutionConfig(
            mode="parallel",
            window_frames=41,
            output_root=tmp_path / "runs",
        ),
        decision=DecisionThresholds(),
        handball=HandballSpecialistConfig(
            feature_config=placeholder,
            train_config=placeholder,
            checkpoints=(placeholder,),
        ),
        general_foul=GeneralFoulSpecialistConfig(
            notebook=placeholder,
            runtime_root=tmp_path / "runtime",
            yolo_weights=str(placeholder),
            pose_model=placeholder,
            skeleton_weights="unused",
            require_roboflow_key=False,
        ),
    )


def test_pipeline_starts_independent_models_concurrently(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    handball_model = FakeSpecialist(
        "handball",
        handball(0.85),
        barrier,
    )
    foul_model = FakeSpecialist("foul", foul(0.80), barrier)

    def prepare(source, output, **kwargs):
        del output, kwargs
        path = Path(source)
        return VideoContext(
            source=path,
            source_video=path,
            frames_dir=tmp_path,
            frame_paths=tuple(tmp_path / f"{i}.jpg" for i in range(41)),
            fps=25.0,
            width=100,
            height=100,
            duration_seconds=41 / 25,
            source_start_frame=0,
        )

    pipeline = ParallelRefereePipeline(
        _dummy_config(tmp_path),
        handball_specialist=handball_model,
        general_foul_specialist=foul_model,
        video_preparer=prepare,
    )
    result = pipeline.run(
        tmp_path / "input.mp4",
        output_dir=tmp_path / "output",
    )
    assert result.final_label == FinalLabel.HANDBALL
    assert result.partial_result is False
    assert (tmp_path / "output" / "combined_result.json").is_file()
