from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from handball_annotator.runtime import get_device
from training.config import load_feature_config, load_train_config
from training.features import (
    ARM_SEGMENTS,
    FEATURE_NAMES,
    FeatureExtractor,
    _contact_sheet,
)
from training.gru import TemporalGRU

from .schemas import (
    HandballResult,
    PreflightReport,
    SpecialistStatus,
    VideoContext,
)


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class HandballSpecialistConfig:
    feature_config: Path
    train_config: Path
    checkpoints: tuple[Path, ...]
    threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("Handball threshold must be between 0 and 1")
        if not self.checkpoints:
            raise ValueError("At least one GRU checkpoint is required")


class HandballSpecialist:
    """Original 56-feature GRU, ensembled across independent fold models."""

    name = "handball_gru_56_feature_ensemble"

    def __init__(self, config: HandballSpecialistConfig):
        self.config = config

    def preflight(self) -> PreflightReport:
        issues: list[str] = []
        for path in (self.config.feature_config, self.config.train_config):
            if not path.is_file():
                issues.append(f"Missing configuration: {path}")
        missing = [
            str(path)
            for path in self.config.checkpoints
            if not path.is_file()
        ]
        if missing:
            issues.append(
                "Missing GRU checkpoints: " + ", ".join(missing)
            )
        return PreflightReport(
            name=self.name,
            available=not issues,
            issues=tuple(issues),
            details={
                "checkpoint_count": len(self.config.checkpoints),
                "feature_count": len(FEATURE_NAMES),
                "sequence_model": "TemporalGRU",
            },
        )

    @staticmethod
    def _quality(
        ball_rate: float,
        player_rate: float,
        pose_rate: float,
    ) -> float:
        return float(
            np.clip(
                0.40 * ball_rate
                + 0.30 * player_rate
                + 0.30 * pose_rate,
                0,
                1,
            )
        )

    @staticmethod
    def _peak_frame(
        features: np.ndarray,
        selected: Sequence[int],
    ) -> int | None:
        if not selected:
            return None
        feature_index = {
            name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES
        }
        candidates: list[tuple[float, int]] = []
        for row_index, row in enumerate(features):
            if (
                row[feature_index["ball_valid"]] < 0.5
                or row[feature_index["player_valid"]] < 0.5
            ):
                continue
            for segment, first, second in ARM_SEGMENTS:
                if (
                    row[feature_index[f"{first}_valid"]] >= 0.5
                    and row[feature_index[f"{second}_valid"]] >= 0.5
                ):
                    candidates.append(
                        (
                            float(
                                row[
                                    feature_index[
                                        f"{segment}_distance"
                                    ]
                                ]
                            ),
                            row_index,
                        )
                    )
        if not candidates:
            return None
        _, local = min(candidates)
        return int(selected[local])

    def predict(
        self,
        context: VideoContext,
        output_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> HandballResult:
        started = time.monotonic()
        report = self.preflight()
        if not report.available:
            return HandballResult.unavailable("; ".join(report.issues))
        progress = progress or (lambda _message: None)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_config = load_feature_config(self.config.feature_config)
        train_config = load_train_config(self.config.train_config)
        device = get_device(train_config.device)
        progress(
            "Handball: extracting one 12×56 YOLO/MediaPipe sequence"
        )
        with FeatureExtractor(feature_config) as extractor:
            features, overlays, selected = extractor.extract(
                list(context.frame_paths)
            )
        if features.shape != (
            feature_config.sequence_length,
            len(FEATURE_NAMES),
        ):
            raise ValueError(
                "Unexpected handball feature shape: "
                f"{features.shape}; expected "
                f"({feature_config.sequence_length}, {len(FEATURE_NAMES)})"
            )

        probabilities: dict[str, float] = {}
        seen_folds: set[int] = set()
        for checkpoint_path in self.config.checkpoints:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            if list(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
                raise ValueError(
                    f"Feature schema mismatch: {checkpoint_path}"
                )
            model_config = dict(checkpoint["model_config"])
            if int(model_config.get("input_size", -1)) != len(FEATURE_NAMES):
                raise ValueError(
                    f"GRU input size mismatch: {checkpoint_path}"
                )
            fold_number = int(checkpoint.get("fold", -1))
            if fold_number in seen_folds:
                raise ValueError(f"Duplicate GRU fold: {fold_number}")
            seen_folds.add(fold_number)
            model = TemporalGRU(**model_config).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            mean = np.asarray(checkpoint["mean"], dtype=np.float32)
            std = np.asarray(checkpoint["std"], dtype=np.float32)
            if (
                mean.shape != (len(FEATURE_NAMES),)
                or std.shape != (len(FEATURE_NAMES),)
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
            ):
                raise ValueError(
                    f"Invalid normalization arrays: {checkpoint_path}"
                )
            normalized = (features - mean) / np.maximum(std, 1e-6)
            with torch.inference_mode():
                tensor = torch.from_numpy(
                    normalized[None].astype(np.float32)
                ).to(device)
                probability = float(
                    torch.sigmoid(model(tensor))[0].cpu()
                )
            fold = str(fold_number)
            probabilities[f"fold_{fold}"] = probability
            progress(
                f"Handball: fold {fold} probability={probability:.3f}"
            )
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

        probability = float(np.mean(list(probabilities.values())))
        index = {name: FEATURE_NAMES.index(name) for name in FEATURE_NAMES}
        ball_rate = float(features[:, index["ball_valid"]].mean())
        player_rate = float(features[:, index["player_valid"]].mean())
        pose_rate = float(
            features[:, index["pose_valid_fraction"]].mean()
        )
        quality = self._quality(ball_rate, player_rate, pose_rate)
        low_confidence = (
            ball_rate < 0.25
            or player_rate < 0.5
            or pose_rate < 0.35
        )
        valid_distances = features[:, index["arm_min_distance"]]
        valid_distances = valid_distances[valid_distances > 0]
        minimum_distance = (
            float(valid_distances.min())
            if len(valid_distances)
            else None
        )
        overlay_path = output_dir / "handball_evidence.jpg"
        _contact_sheet(overlays, overlay_path)
        elapsed = time.monotonic() - started
        predicted = (
            "handball"
            if probability >= self.config.threshold
            else "not_handball"
        )
        local_peak = self._peak_frame(features, selected)
        progress(
            "Handball: ensemble "
            f"{probability:.3f} → {predicted} ({elapsed:.1f}s)"
        )
        return HandballResult(
            status=SpecialistStatus.COMPLETED,
            probability=probability,
            predicted_label=predicted,
            threshold=self.config.threshold,
            fold_probabilities=probabilities,
            ensemble_standard_deviation=float(
                np.std(list(probabilities.values()))
            ),
            selected_frame_indices=[
                context.source_start_frame + int(value)
                for value in selected
            ],
            peak_frame=(
                context.source_start_frame
                + int(local_peak)
                if local_peak is not None
                else None
            ),
            quality=quality,
            ball_detection_rate=ball_rate,
            player_detection_rate=player_rate,
            pose_valid_rate=pose_rate,
            minimum_normalized_arm_distance=minimum_distance,
            low_confidence_warning=low_confidence,
            reason="mean_probability_across_original_56_feature_gru_folds",
            overlay_path=str(overlay_path),
            elapsed_seconds=elapsed,
        )
