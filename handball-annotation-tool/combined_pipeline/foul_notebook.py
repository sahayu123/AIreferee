from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

import cv2
import numpy as np

from .schemas import (
    GeneralFoulResult,
    PreflightReport,
    SpecialistStatus,
    VideoContext,
)


ProgressCallback = Callable[[str], None]
_IMPORT_LOCK = threading.Lock()
_PACKAGE_PREFIX = "airef"
_REQUIRED_MODULES = {
    "image_mlp.py",
    "device.py",
    "roboflow_client.py",
    "tackle_classifier.py",
    "contact_classifier.py",
    "contact.py",
    "v411_fusion.py",
    "prototype_depth.py",
    "tracked_video.py",
}
_IMPORT_DEPENDENCIES = (
    "cv2",
    "mediapipe",
    "numpy",
    "onnxruntime",
    "PIL",
    "rtmlib",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
)


@dataclass(frozen=True)
class GeneralFoulSpecialistConfig:
    notebook: Path
    runtime_root: Path
    yolo_weights: str
    pose_model: Path
    skeleton_weights: str
    tracker: str = "botsort.yaml"
    frame_threshold: float = 0.60
    minimum_seconds: float = 0.12
    high_confidence_override: float = 1.01
    max_frames: int | None = None
    make_video: bool = True
    require_roboflow_key: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.frame_threshold < 1:
            raise ValueError("frame_threshold must be between 0 and 1")
        if self.minimum_seconds <= 0:
            raise ValueError("minimum_seconds must be positive")
        if self.high_confidence_override <= 0:
            raise ValueError("high_confidence_override must be positive")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive")


def _notebook_modules(notebook: Path) -> dict[str, str]:
    """Read only the notebook's explicitly declared ``airef`` modules."""
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook has no cells: {notebook}")
    modules: dict[str, str] = {}
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source_value = cell.get("source", [])
        source = (
            "".join(source_value)
            if isinstance(source_value, list)
            else str(source_value)
        )
        lines = source.splitlines(keepends=True)
        if not lines or not lines[0].startswith("%%writefile "):
            continue
        declared = lines[0].removeprefix("%%writefile ").strip()
        path = PurePosixPath(declared)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 2
            or path.parts[0] != _PACKAGE_PREFIX
        ):
            raise ValueError(
                f"Unsafe or unexpected notebook output path: {declared}"
            )
        modules[path.name] = "".join(lines[1:])
    missing = sorted(_REQUIRED_MODULES - modules.keys())
    if missing:
        raise ValueError(
            "Foul notebook is missing required modules: "
            + ", ".join(missing)
        )
    return modules


def extract_notebook_package(
    notebook: Path,
    runtime_root: Path,
) -> dict[str, str]:
    """Materialize the tracked notebook modules into an ignored runtime dir."""
    modules = _notebook_modules(notebook)
    digest = hashlib.sha256(
        notebook.read_bytes()
    ).hexdigest()
    package = runtime_root / _PACKAGE_PREFIX
    marker = package / ".source.json"
    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if (
            existing.get("notebook_sha256") == digest
            and all((package / name).is_file() for name in modules)
        ):
            return {
                name: str(package / name)
                for name in sorted(modules)
            }
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text(
        '"""Extracted general-foul prototype modules."""\n',
        encoding="utf-8",
    )
    for name, source in modules.items():
        (package / name).write_text(source, encoding="utf-8")
    marker.write_text(
        json.dumps(
            {
                "notebook": str(notebook),
                "notebook_sha256": digest,
                "modules": sorted(modules),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        name: str(package / name)
        for name in sorted(modules)
    }


class NotebookGeneralFoulSpecialist:
    """Adapter around the independent v2Prototype currently stored on main."""

    name = "main_v2_general_foul_prototype"

    def __init__(self, config: GeneralFoulSpecialistConfig):
        self.config = config

    @property
    def expected_checkpoints(self) -> tuple[Path, ...]:
        runs = self.config.runtime_root / "runs"
        return (
            runs / "image_foul_mlp_v408.pt",
            runs / "clean_tackle" / "resnet18_mixed.pt",
            runs / "contact" / "resnet_contact.pt",
        )

    def preflight(self) -> PreflightReport:
        issues: list[str] = []
        if not self.config.notebook.is_file():
            issues.append(f"Missing foul notebook: {self.config.notebook}")
        if not self.config.pose_model.is_file():
            issues.append(f"Missing pose model: {self.config.pose_model}")
        yolo_path = Path(self.config.yolo_weights)
        if (
            (yolo_path.is_absolute() or yolo_path.parent != Path("."))
            and not yolo_path.is_file()
        ):
            issues.append(f"Missing YOLO weights: {yolo_path}")
        missing_packages = [
            name
            for name in _IMPORT_DEPENDENCIES
            if importlib.util.find_spec(name) is None
        ]
        if importlib.util.find_spec("moge") is None:
            missing_packages.append("moge")
        if missing_packages:
            issues.append(
                "Missing Python packages: "
                + ", ".join(sorted(missing_packages))
            )
        missing_weights = [
            str(path)
            for path in self.expected_checkpoints
            if not path.is_file()
        ]
        if missing_weights:
            issues.append(
                "Missing general-foul checkpoints: "
                + ", ".join(missing_weights)
            )
        if (
            self.config.require_roboflow_key
            and not os.environ.get("ROBOFLOW_API_KEY")
            and not (self.config.runtime_root / ".env").is_file()
        ):
            issues.append(
                "ROBOFLOW_API_KEY is not set and runtime .env is absent"
            )
        return PreflightReport(
            name=self.name,
            available=not issues,
            issues=tuple(issues),
            details={
                "notebook": str(self.config.notebook),
                "runtime_root": str(self.config.runtime_root),
                "checkpoint_count": len(self.expected_checkpoints),
                "high_confidence_override": (
                    self.config.high_confidence_override
                ),
                "single_frame_override_enabled": (
                    self.config.high_confidence_override <= 1
                ),
                "uses_hosted_roboflow_per_frame": True,
            },
        )

    def _load_judge(self):
        extract_notebook_package(
            self.config.notebook,
            self.config.runtime_root,
        )
        runtime = str(self.config.runtime_root)
        with _IMPORT_LOCK:
            if runtime not in sys.path:
                sys.path.insert(0, runtime)
            existing = sys.modules.get("airef")
            if existing is not None:
                locations = list(
                    getattr(existing, "__path__", ())
                )
                if not any(
                    Path(location).resolve()
                    == (self.config.runtime_root / "airef").resolve()
                    for location in locations
                ):
                    for name in list(sys.modules):
                        if name == "airef" or name.startswith("airef."):
                            del sys.modules[name]
            module = importlib.import_module("airef.tracked_video")
        return module.judge_video

    @staticmethod
    def _probability(verdict: str, confidence: float) -> tuple[str, float]:
        normalized = verdict.strip().upper()
        confidence = float(np.clip(confidence, 0, 1))
        if normalized.startswith("FOUL"):
            return "foul", confidence
        if normalized.startswith("NO FOUL"):
            return "not_foul", 1.0 - confidence
        return "needs_review", confidence

    def predict(
        self,
        context: VideoContext,
        output_dir: Path,
        progress: ProgressCallback | None = None,
    ) -> GeneralFoulResult:
        started = time.monotonic()
        progress = progress or (lambda _message: None)
        report = self.preflight()
        if not report.available:
            return GeneralFoulResult.unavailable(
                "; ".join(report.issues)
            )
        if context.source_video is None:
            return GeneralFoulResult.unavailable(
                "General-foul specialist requires a source video"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        progress("General foul: loading the main-branch v2Prototype")
        judge_video = self._load_judge()
        result = judge_video(
            str(context.source_video),
            yolo_weights=self.config.yolo_weights,
            tracker=self.config.tracker,
            pose_model=str(self.config.pose_model),
            skeleton_weights=self.config.skeleton_weights,
            frame_threshold=self.config.frame_threshold,
            minimum_seconds=self.config.minimum_seconds,
            high_confidence_override=(
                self.config.high_confidence_override
            ),
            max_frames=self.config.max_frames,
            make_video=self.config.make_video,
            progress=lambda message: progress(
                f"General foul: {message}"
            ),
        )
        label, probability = self._probability(
            str(result.verdict),
            float(result.confidence),
        )
        peak_time = float(result.peak_time)
        peak_frame = int(round(peak_time * context.fps))
        timeline = [
            dict(row)
            for row in getattr(result, "timeline", [])
            if isinstance(row, dict)
        ]
        peak_row = (
            min(
                timeline,
                key=lambda row: abs(
                    float(row.get("time", 0.0)) - peak_time
                ),
            )
            if timeline
            else {}
        )
        peak_image_path: str | None = None
        if getattr(result, "peak_frame", None) is not None:
            destination = output_dir / "general_foul_peak.jpg"
            if cv2.imwrite(str(destination), result.peak_frame):
                peak_image_path = str(destination)
        annotated_path: str | None = None
        raw_annotated = getattr(result, "annotated_video", None)
        if raw_annotated and Path(raw_annotated).is_file():
            destination = output_dir / "general_foul_annotated.mp4"
            shutil.copy2(raw_annotated, destination)
            annotated_path = str(destination)
        elapsed = time.monotonic() - started
        progress(
            f"General foul: probability={probability:.3f}, "
            f"verdict={label} ({elapsed:.1f}s)"
        )
        reason = str(result.verdict)
        if peak_image_path:
            reason += f"; peak_image={peak_image_path}"
        return GeneralFoulResult(
            status=SpecialistStatus.COMPLETED,
            probability=probability,
            predicted_label=label,
            confidence=float(result.confidence),
            peak_frame=peak_frame,
            peak_time=peak_time,
            quality=1.0,
            contact_type=(
                str(peak_row["contact_type"])
                if peak_row.get("contact_type") is not None
                else None
            ),
            tackle_type=(
                str(peak_row["tackle_type"])
                if peak_row.get("tackle_type") is not None
                else None
            ),
            reason=reason,
            report=str(getattr(result, "report", "")),
            timeline=timeline,
            annotated_video_path=annotated_path,
            elapsed_seconds=elapsed,
        )
