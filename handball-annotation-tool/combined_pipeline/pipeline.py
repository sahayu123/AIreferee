from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import yaml

from .decision import DecisionThresholds, combine_decisions
from .foul_notebook import (
    GeneralFoulSpecialistConfig,
    NotebookGeneralFoulSpecialist,
)
from .handball import HandballSpecialist, HandballSpecialistConfig
from .schemas import (
    CombinedResult,
    GeneralFoulResult,
    HandballResult,
    PreflightReport,
    VideoContext,
)
from .video import prepare_video


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ProgressCallback = Callable[[str], None]


class Specialist(Protocol):
    name: str

    def preflight(self) -> PreflightReport: ...

    def predict(
        self,
        context: VideoContext,
        output_dir: Path,
        progress: ProgressCallback | None = None,
    ): ...


def _path(value: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str = "parallel"
    window_frames: int = 41
    incident_time_seconds: float | None = None
    output_root: Path = PROJECT_ROOT / "artifacts/parallel_runs"

    def __post_init__(self) -> None:
        if self.mode not in {"parallel", "sequential"}:
            raise ValueError("execution.mode must be parallel or sequential")
        if self.window_frames < 12:
            raise ValueError("execution.window_frames must be at least 12")
        if (
            self.incident_time_seconds is not None
            and self.incident_time_seconds < 0
        ):
            raise ValueError(
                "execution.incident_time_seconds cannot be negative"
            )


@dataclass(frozen=True)
class ParallelPipelineConfig:
    execution: ExecutionConfig
    decision: DecisionThresholds
    handball: HandballSpecialistConfig
    general_foul: GeneralFoulSpecialistConfig

    @classmethod
    def load(cls, path: str | Path) -> "ParallelPipelineConfig":
        config_path = _path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid pipeline configuration: {config_path}")
        execution = payload.get("execution", {})
        decision = payload.get("decision", {})
        handball = payload.get("handball", {})
        foul = payload.get("general_foul", {})
        if not all(
            isinstance(value, dict)
            for value in (execution, decision, handball, foul)
        ):
            raise ValueError("Every pipeline configuration section is an object")

        checkpoint_values = handball.get("checkpoints", [])
        if not isinstance(checkpoint_values, list):
            raise ValueError("handball.checkpoints must be a list")
        max_frames = foul.get("max_frames")
        return cls(
            execution=ExecutionConfig(
                mode=str(execution.get("mode", "parallel")),
                window_frames=int(execution.get("window_frames", 41)),
                incident_time_seconds=(
                    float(execution["incident_time_seconds"])
                    if execution.get("incident_time_seconds") is not None
                    else None
                ),
                output_root=_path(
                    execution.get(
                        "output_root",
                        "artifacts/parallel_runs",
                    )
                ),
            ),
            decision=DecisionThresholds(
                handball_high=float(
                    decision.get("handball_high", 0.70)
                ),
                handball_low=float(
                    decision.get("handball_low", 0.30)
                ),
                foul_high=float(decision.get("foul_high", 0.70)),
                foul_low=float(decision.get("foul_low", 0.30)),
                minimum_quality=float(
                    decision.get("minimum_quality", 0.20)
                ),
            ),
            handball=HandballSpecialistConfig(
                feature_config=_path(
                    handball.get(
                        "feature_config",
                        "configs/mediapipe_features.yaml",
                    )
                ),
                train_config=_path(
                    handball.get(
                        "train_config",
                        "configs/temporal_classifier.yaml",
                    )
                ),
                checkpoints=tuple(
                    _path(value) for value in checkpoint_values
                ),
                threshold=float(handball.get("threshold", 0.50)),
            ),
            general_foul=GeneralFoulSpecialistConfig(
                notebook=_path(foul["notebook"]),
                runtime_root=_path(
                    foul.get(
                        "runtime_root",
                        ".runtime/general_foul",
                    )
                ),
                yolo_weights=str(_path(foul["yolo_weights"])),
                pose_model=_path(foul["pose_model"]),
                skeleton_weights=str(foul["skeleton_weights"]),
                tracker=str(foul.get("tracker", "botsort.yaml")),
                frame_threshold=float(
                    foul.get("frame_threshold", 0.60)
                ),
                minimum_seconds=float(
                    foul.get("minimum_seconds", 0.12)
                ),
                high_confidence_override=float(
                    foul.get("high_confidence_override", 1.01)
                ),
                max_frames=(
                    int(max_frames) if max_frames is not None else None
                ),
                make_video=bool(foul.get("make_video", True)),
                require_roboflow_key=bool(
                    foul.get("require_roboflow_key", True)
                ),
            ),
        )


class ParallelRefereePipeline:
    def __init__(
        self,
        config: ParallelPipelineConfig,
        *,
        handball_specialist: Specialist | None = None,
        general_foul_specialist: Specialist | None = None,
        video_preparer: Callable[..., VideoContext] = prepare_video,
    ):
        self.config = config
        self.handball = handball_specialist or HandballSpecialist(
            config.handball
        )
        self.general_foul = (
            general_foul_specialist
            or NotebookGeneralFoulSpecialist(config.general_foul)
        )
        self.video_preparer = video_preparer
        self._run_lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        path: str | Path,
    ) -> "ParallelRefereePipeline":
        return cls(ParallelPipelineConfig.load(path))

    def preflight(self) -> dict[str, dict[str, object]]:
        return {
            "handball": self.handball.preflight().to_dict(),
            "general_foul": self.general_foul.preflight().to_dict(),
        }

    def _run_handball(
        self,
        context: VideoContext,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> HandballResult:
        try:
            return self.handball.predict(
                context,
                output_dir / "handball",
                progress,
            )
        except Exception as exc:
            progress(
                f"Handball specialist failed: {type(exc).__name__}: {exc}"
            )
            return HandballResult.failed(exc)

    def _run_foul(
        self,
        context: VideoContext,
        output_dir: Path,
        progress: ProgressCallback,
    ) -> GeneralFoulResult:
        try:
            return self.general_foul.predict(
                context,
                output_dir / "general_foul",
                progress,
            )
        except Exception as exc:
            progress(
                "General-foul specialist failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return GeneralFoulResult.failed(exc)

    def run(
        self,
        source: str | Path,
        *,
        output_dir: str | Path | None = None,
        incident_time_seconds: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> CombinedResult:
        # The notebook backend caches mutable global models and the handball
        # extractor owns tracker state. Keep separate UI requests serialized
        # while still running the two specialists concurrently within a run.
        with self._run_lock:
            return self._run_once(
                source,
                output_dir=output_dir,
                incident_time_seconds=incident_time_seconds,
                progress=progress,
            )

    def _run_once(
        self,
        source: str | Path,
        *,
        output_dir: str | Path | None = None,
        incident_time_seconds: float | None = None,
        progress: ProgressCallback | None = None,
    ) -> CombinedResult:
        started = time.monotonic()
        callback = progress or (lambda _message: None)
        callback_lock = threading.Lock()

        def notify(message: str) -> None:
            with callback_lock:
                callback(message)

        if output_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output = (
                self.config.execution.output_root
                / f"run_{stamp}_{uuid.uuid4().hex[:8]}"
            )
        else:
            output = _path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        chosen_time = (
            self.config.execution.incident_time_seconds
            if incident_time_seconds is None
            else incident_time_seconds
        )
        notify(
            "Preparing one shared "
            f"{self.config.execution.window_frames}-frame incident window "
            "for both models"
        )
        context = self.video_preparer(
            source,
            output / "input",
            max_frames=self.config.execution.window_frames,
            incident_time_seconds=chosen_time,
        )
        notify(
            f"Prepared {len(context.frame_paths)} frames at "
            f"{context.fps:.2f} FPS"
        )

        if self.config.execution.mode == "parallel":
            notify("Starting both independent specialists in parallel")
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="referee-specialist",
            ) as executor:
                handball_future = executor.submit(
                    self._run_handball,
                    context,
                    output,
                    notify,
                )
                foul_future = executor.submit(
                    self._run_foul,
                    context,
                    output,
                    notify,
                )
                handball = handball_future.result()
                foul = foul_future.result()
        else:
            notify("Starting specialists sequentially")
            handball = self._run_handball(
                context,
                output,
                notify,
            )
            foul = self._run_foul(
                context,
                output,
                notify,
            )

        label, confidence, reason, partial = combine_decisions(
            handball,
            foul,
            self.config.decision,
        )
        result = CombinedResult(
            final_label=label,
            confidence=confidence,
            decision_reason=reason,
            handball=handball,
            general_foul=foul,
            input_path=str(Path(source).expanduser().resolve()),
            output_dir=str(output),
            execution_mode=self.config.execution.mode,
            partial_result=partial,
            elapsed_seconds=time.monotonic() - started,
        )
        destination = output / "combined_result.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)
        notify(
            f"Final decision: {label.value} "
            f"({confidence:.1%}); partial={partial}"
        )
        return result
