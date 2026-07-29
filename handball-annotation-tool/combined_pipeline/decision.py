from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    FinalLabel,
    GeneralFoulResult,
    HandballResult,
    SpecialistStatus,
)


@dataclass(frozen=True)
class DecisionThresholds:
    handball_high: float = 0.70
    handball_low: float = 0.30
    foul_high: float = 0.70
    foul_low: float = 0.30
    minimum_quality: float = 0.20

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.handball_low >= self.handball_high:
            raise ValueError("handball_low must be below handball_high")
        if self.foul_low >= self.foul_high:
            raise ValueError("foul_low must be below foul_high")


def _completed(status: SpecialistStatus) -> bool:
    return status == SpecialistStatus.COMPLETED


def combine_decisions(
    handball: HandballResult,
    foul: GeneralFoulResult,
    thresholds: DecisionThresholds,
) -> tuple[FinalLabel, float, str, bool]:
    """Fuse specialist decisions without mixing their internal features."""
    handball_completed = (
        _completed(handball.status) and handball.probability is not None
    )
    foul_completed = (
        _completed(foul.status) and foul.probability is not None
    )
    handball_ready = (
        handball_completed
        and handball.quality >= thresholds.minimum_quality
    )
    foul_ready = (
        foul_completed
        and foul.quality >= thresholds.minimum_quality
        and foul.predicted_label in {"foul", "not_foul"}
    )
    partial = not (handball_completed and foul_completed)

    if handball_ready and handball.probability >= thresholds.handball_high:
        return (
            FinalLabel.HANDBALL,
            float(handball.probability),
            "handball_specialist_high_confidence",
            partial,
        )

    if not (handball_completed and foul_completed):
        return (
            FinalLabel.NEEDS_REVIEW,
            float(
                max(
                    handball.probability or 0.0,
                    foul.probability or 0.0,
                )
            ),
            "one_or_more_specialists_unavailable",
            True,
        )

    if not handball_ready or foul.quality < thresholds.minimum_quality:
        return (
            FinalLabel.NEEDS_REVIEW,
            float(max(handball.probability, foul.probability)),
            "one_or_more_specialists_low_quality",
            False,
        )

    if not foul_ready:
        return (
            FinalLabel.NEEDS_REVIEW,
            float(max(handball.probability, foul.probability)),
            "general_foul_specialist_requested_review",
            False,
        )

    hp = float(handball.probability)
    fp = float(foul.probability)
    if hp <= thresholds.handball_low and fp >= thresholds.foul_high:
        return (
            FinalLabel.OTHER_FOUL,
            fp,
            "handball_rejected_general_foul_confirmed",
            False,
        )
    if hp <= thresholds.handball_low and fp <= thresholds.foul_low:
        return (
            FinalLabel.NO_FOUL,
            float(1 - max(hp, fp)),
            "both_specialists_low_confidence",
            False,
        )
    return (
        FinalLabel.NEEDS_REVIEW,
        float(max(hp, fp)),
        "specialists_uncertain_or_disagree",
        False,
    )
