"""Parallel handball and general-foul inference pipeline."""

from .decision import DecisionThresholds, combine_decisions
from .pipeline import ParallelPipelineConfig, ParallelRefereePipeline
from .schemas import (
    CombinedResult,
    GeneralFoulResult,
    HandballResult,
    SpecialistStatus,
)

__all__ = [
    "CombinedResult",
    "DecisionThresholds",
    "GeneralFoulResult",
    "HandballResult",
    "ParallelPipelineConfig",
    "ParallelRefereePipeline",
    "SpecialistStatus",
    "combine_decisions",
]
