from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SpecialistStatus(StrEnum):
    READY = "ready"
    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class FinalLabel(StrEnum):
    HANDBALL = "handball"
    OTHER_FOUL = "other_foul"
    NO_FOUL = "no_foul"
    NEEDS_REVIEW = "needs_review"


@dataclass(frozen=True)
class VideoContext:
    source: Path
    source_video: Path | None
    frames_dir: Path
    frame_paths: tuple[Path, ...]
    fps: float
    width: int
    height: int
    duration_seconds: float
    source_start_frame: int


@dataclass
class HandballResult:
    status: SpecialistStatus
    probability: float | None = None
    predicted_label: str = "unavailable"
    threshold: float = 0.5
    fold_probabilities: dict[str, float] = field(default_factory=dict)
    ensemble_standard_deviation: float = 0.0
    selected_frame_indices: list[int] = field(default_factory=list)
    peak_frame: int | None = None
    quality: float = 0.0
    ball_detection_rate: float | None = None
    player_detection_rate: float | None = None
    pose_valid_rate: float | None = None
    minimum_normalized_arm_distance: float | None = None
    low_confidence_warning: bool = False
    reason: str = ""
    overlay_path: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "HandballResult":
        return cls(
            status=SpecialistStatus.UNAVAILABLE,
            reason=reason,
            error=reason,
        )

    @classmethod
    def failed(cls, error: BaseException) -> "HandballResult":
        detail = f"{type(error).__name__}: {error}"
        return cls(
            status=SpecialistStatus.FAILED,
            reason="handball_specialist_failed",
            error=detail,
        )


@dataclass
class GeneralFoulResult:
    status: SpecialistStatus
    probability: float | None = None
    predicted_label: str = "unavailable"
    confidence: float = 0.0
    peak_frame: int | None = None
    peak_time: float | None = None
    quality: float = 0.0
    contact_type: str | None = None
    tackle_type: str | None = None
    reason: str = ""
    report: str = ""
    timeline: list[dict[str, Any]] = field(default_factory=list)
    annotated_video_path: str | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None

    @classmethod
    def unavailable(cls, reason: str) -> "GeneralFoulResult":
        return cls(
            status=SpecialistStatus.UNAVAILABLE,
            reason=reason,
            error=reason,
        )

    @classmethod
    def failed(cls, error: BaseException) -> "GeneralFoulResult":
        detail = f"{type(error).__name__}: {error}"
        return cls(
            status=SpecialistStatus.FAILED,
            reason="general_foul_specialist_failed",
            error=detail,
        )


@dataclass
class CombinedResult:
    final_label: FinalLabel
    confidence: float
    decision_reason: str
    handball: HandballResult
    general_foul: GeneralFoulResult
    input_path: str
    output_dir: str
    execution_mode: str
    partial_result: bool
    elapsed_seconds: float
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["final_label"] = str(self.final_label)
        value["handball"]["status"] = str(self.handball.status)
        value["general_foul"]["status"] = str(
            self.general_foul.status
        )
        return value


@dataclass(frozen=True)
class PreflightReport:
    name: str
    available: bool
    issues: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
