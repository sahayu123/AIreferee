from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
NOTEBOOK_NAME = "AI Referee Foul Checker Prototype (1).ipynb"


def resolve_source_root() -> Path:
    configured = os.environ.get("VAIR_SOURCE_ROOT")
    candidates = [
        Path(configured).expanduser() if configured else None,
        APP_ROOT.parent,
        APP_ROOT.parent / "AIreferee_GitHub_GPU",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / NOTEBOOK_NAME).exists():
            return candidate.resolve()
    return APP_ROOT.parent.resolve()


SOURCE_ROOT = resolve_source_root()
CHECKPOINT_ROOT = Path(
    os.environ.get("VAIR_CHECKPOINT_ROOT", APP_ROOT / "models")
).expanduser().resolve()
NOTEBOOK = SOURCE_ROOT / NOTEBOOK_NAME
HANDBALL_PROJECT_ADAPTER = HERE / "handball_project_detector.py"
RUNTIME_ROOT = HERE / ".runtime"
ARTIFACT_ROOT = HERE / ".artifacts"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
WEIGHTS = {
    "image_foul_mlp_v408.pt": CHECKPOINT_ROOT / "image_foul_mlp_v408.pt",
    "resnet18_mixed.pt": CHECKPOINT_ROOT / "resnet18_mixed.pt",
    "resnet_contact.pt": CHECKPOINT_ROOT / "resnet_contact.pt",
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
MODEL_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)
RUNTIME_READY = False


def gpu_patch_module(relative: str, source: str) -> str:
    """Route the notebook's compatible inference paths through Apple GPU APIs."""
    mps_choice = (
        '"cuda" if torch.cuda.is_available() else '
        '"mps" if torch.backends.mps.is_available() else "cpu"'
    )
    if relative == "airef/image_mlp.py":
        source = source.replace(
            'return torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            f"return torch.device({mps_choice})",
        )
    elif relative == "airef/prototype_depth.py":
        source = source.replace(
            'torch.device("cuda" if torch.cuda.is_available() else "cpu")',
            f"torch.device({mps_choice})",
        )
    elif relative == "airef/tracked_video.py":
        source = source.replace(
            'device="cuda" if torch.cuda.is_available() else "cpu",',
            f"device=({mps_choice}),",
        )
        source = source.replace(
            'device = 0 if torch.cuda.is_available() else "cpu"',
            'device = 0 if torch.cuda.is_available() else '
            '"mps" if torch.backends.mps.is_available() else "cpu"',
        )
    elif relative == "airef/contact.py":
        source = source.replace("import numpy as np\n", "import numpy as np\nimport torch\n")
        source = source.replace(
            'result = _pose_model()(img_bgr, conf=0.20, verbose=False)[0]',
            'result = _pose_model()(img_bgr, conf=0.20, verbose=False, '
            'device=("mps" if torch.backends.mps.is_available() else "cpu"))[0]',
        )
        source = source.replace(
            'res = _ball_model()(im, verbose=False, conf=conf_thr)[0]',
            'res = _ball_model()(im, verbose=False, conf=conf_thr, '
            'device=("mps" if torch.backends.mps.is_available() else "cpu"))[0]',
        )
    return source


def prepare_runtime() -> None:
    """Extract the exact embedded modules from the source v2Prototype notebook."""
    global RUNTIME_READY
    if RUNTIME_READY:
        return
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"v2Prototype notebook not found: {NOTEBOOK}")
    missing = [name for name, path in WEIGHTS.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing v2Prototype checkpoints: " + ", ".join(missing))

    package = RUNTIME_ROOT / "airef"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    extracted = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        first, _, rest = source.partition("\n")
        if not first.startswith("%%writefile airef/"):
            continue
        relative = first.removeprefix("%%writefile ").strip()
        destination = RUNTIME_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(gpu_patch_module(relative, rest), encoding="utf-8")
        extracted += 1
    if extracted < 8:
        raise RuntimeError(f"Only extracted {extracted} v2Prototype modules")

    run_root = RUNTIME_ROOT / "runs"
    (run_root / "clean_tackle").mkdir(parents=True, exist_ok=True)
    (run_root / "contact").mkdir(parents=True, exist_ok=True)
    links = {
        run_root / "image_foul_mlp_v408.pt": WEIGHTS["image_foul_mlp_v408.pt"],
        run_root / "clean_tackle/resnet18_mixed.pt": WEIGHTS["resnet18_mixed.pt"],
        run_root / "contact/resnet_contact.pt": WEIGHTS["resnet_contact.pt"],
    }
    for destination, source in links.items():
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source)

    if str(RUNTIME_ROOT) not in sys.path:
        sys.path.insert(0, str(RUNTIME_ROOT))
    RUNTIME_READY = True


def update_job(job_id: str, **values: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(values)


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def progress_callback(job_id: str):
    stages = [
        ("loading", 8),
        ("tracked", 24),
        ("trained image detector", 37),
        ("supporting MLP", 47),
        ("Roboflow", 61),
        ("fusion", 76),
        ("SAM 2", 84),
    ]

    def report(message: str) -> None:
        lowered = message.lower()
        progress = 12
        for marker, percentage in stages:
            if marker.lower() in lowered:
                progress = percentage
        update_job(job_id, stage=message, progress=progress)

    return report


def handball_progress_callback(job_id: str):
    def report(message: str) -> None:
        update_job(job_id, stage=message, progress=91)

    return report


def artifact_url(job_id: str, path: Path | None) -> str | None:
    return None if path is None else f"/api/artifacts/{job_id}/{path.name}"


def load_classifiers() -> None:
    from airef.image_mlp import load_model
    from airef.tackle_classifier import load_classifier
    from airef.contact_classifier import load_contact

    load_model(WEIGHTS["image_foul_mlp_v408.pt"])
    load_classifier(WEIGHTS["resnet18_mixed.pt"])
    load_contact(WEIGHTS["resnet_contact.pt"])


def run_handball_video(
    job_id: str,
    media_path: Path,
    output_dir: Path,
):
    from .handball_project_detector import analyze_handball_project

    update_job(
        job_id,
        stage="Running Handball Detection Project ball/arm geometry",
        progress=87,
    )
    result = analyze_handball_project(
        media_path,
        output_dir / "handball-project",
        handball_progress_callback(job_id),
    )
    return result


def combine_video_verdict(general_result: Any, handball_result: Any):
    normalized = str(general_result.verdict).strip().upper()
    if normalized.startswith("FOUL"):
        general_label = "foul"
        general_probability = float(general_result.confidence)
    elif normalized.startswith("NO FOUL"):
        general_label = "not_foul"
        general_probability = 1.0 - float(general_result.confidence)
    else:
        general_label = "needs_review"
        general_probability = float(general_result.confidence)
    handball_probability = float(handball_result.probability)
    if handball_result.predicted_label == "handball" and handball_probability >= 0.70:
        return "handball", handball_probability, "ball-arm collision and arm-angle rule confirmed handball", False
    if general_label == "foul" and general_probability >= 0.70:
        return "other_foul", general_probability, "general-foul specialist exceeded threshold", False
    if (
        general_label == "not_foul"
        and general_probability <= 0.30
        and handball_probability <= 0.30
    ):
        confidence = max(1.0 - general_probability, 1.0 - handball_probability)
        return "no_foul", confidence, "both specialists were below threshold", False
    confidence = max(
        handball_probability,
        general_probability,
        1.0 - handball_probability,
        1.0 - general_probability,
    )
    return "needs_review", confidence, "handball geometry and general-foul evidence were uncertain", False


def image_analysis(job_id: str, media_path: Path, api_key: str) -> dict[str, Any]:
    prepare_runtime()
    os.environ["ROBOFLOW_API_KEY"] = api_key
    from airef.image_mlp import predict_bgr
    from airef.v411_fusion import run_image

    update_job(job_id, stage="Loading the three v2Prototype checkpoints", progress=12)
    load_classifiers()
    image = cv2.imread(str(media_path))
    if image is None:
        raise ValueError("The uploaded image could not be decoded.")

    update_job(job_id, stage="Running the trained image MLP", progress=34)
    probabilities, classes = predict_bgr(
        [image], weights=WEIGHTS["image_foul_mlp_v408.pt"], batch_size=1
    )
    mlp_foul = float(probabilities[0, classes.index("foul")])
    update_job(job_id, stage="Running Roboflow and V4.11 image fusion", progress=61)
    verdict = run_image(image, mlp_foul=mlp_foul)

    output_dir = ARTIFACT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "reviewed-frame.jpg"
    cv2.imwrite(str(frame_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    evidence = (
        verdict.confidence if verdict.outcome == "FOUL"
        else 1.0 - verdict.confidence if verdict.outcome == "NO FOUL"
        else 0.5
    )
    return {
        "verdict": verdict.outcome,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "frames_analyzed": 1,
        "peak_time": 0.0,
        "metrics": {
            "mlp_foul": mlp_foul,
            "roboflow_foul": verdict.roboflow_foul,
            "limb_gap": verdict.pose_gap,
            "contact_type": verdict.contact_type,
            "contact_probability": verdict.contact_probability,
            "tackle_type": verdict.tackle_type,
            "tackle_probability": verdict.tackle_probability,
            "metric_contact": "NOT RUN",
            "handball_status": "video_only",
        },
        "peak_frame_url": artifact_url(job_id, frame_path),
        "timeline": [{
            "time": 0.0,
            "scene_foul": mlp_foul,
            "pair_crop_foul": mlp_foul,
            "evidence": evidence,
        }],
    }


def video_analysis(job_id: str, media_path: Path, api_key: str) -> dict[str, Any]:
    prepare_runtime()
    os.environ["ROBOFLOW_API_KEY"] = api_key
    from airef.tracked_video import judge_video

    update_job(job_id, stage="Loading the three v2Prototype checkpoints", progress=7)
    load_classifiers()
    result = judge_video(
        str(media_path),
        yolo_weights="yolo11m.pt",
        tracker="botsort.yaml",
        skeleton_weights=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip"
        ),
        frame_threshold=0.60,
        minimum_seconds=0.12,
        high_confidence_override=0.97,
        max_frames=None,
        make_video=True,
        progress=progress_callback(job_id),
    )

    output_dir = ARTIFACT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "peak-frame.jpg"
    video_path = output_dir / "annotated-review.mp4"
    if result.peak_frame is not None:
        cv2.imwrite(str(frame_path), result.peak_frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if result.annotated_video:
        shutil.copy2(result.annotated_video, video_path)

    peak_row = min(
        result.timeline,
        key=lambda row: abs(float(row["time"]) - float(result.peak_time)),
    ) if result.timeline else {}
    reason = (
        result.verdict.split("—", 1)[1].strip()
        if "—" in result.verdict else result.verdict
    )
    handball = run_handball_video(
        job_id,
        media_path,
        output_dir,
    )
    final_label, final_confidence, decision_reason, partial = combine_video_verdict(
        result,
        handball,
    )
    handball_overlay = (
        Path(handball.overlay_path)
        if getattr(handball, "overlay_path", None)
        else None
    )
    handball_overlay_path = output_dir / "handball-evidence.jpg"
    if handball_overlay is not None and handball_overlay.is_file():
        shutil.copy2(handball_overlay, handball_overlay_path)

    if final_label == "handball":
        final_verdict = "FOUL — HANDBALL"
        if getattr(handball, "proximity_override", False):
            distance = float(handball.minimum_normalized_arm_distance or 0.0)
            threshold = float(handball.proximity_threshold)
            reason = (
                f"automatic handball: ball-to-{handball.handball_part}-arm "
                f"gap was {distance:.2%}, below the {threshold:.2%} "
                "high-confidence proximity threshold"
            )
        else:
            reason = (
                f"ball intersected the {handball.handball_part} arm at "
                f"{float(handball.handball_angle or 0):.0f}°; the supplied "
                "Handball Detection Project angle rule awards handball"
            )
    elif final_label == "other_foul":
        final_verdict = result.verdict
    elif final_label == "no_foul":
        final_verdict = "NO FOUL"
        reason = "both the handball and general-foul specialists were below threshold"
    else:
        final_verdict = "NEEDS REVIEW"
        reason = (
            "handball and general-foul specialists were uncertain or disagreed"
            if not partial
            else "one specialist could not complete"
        )

    handball_probability = (
        float(handball.probability)
        if getattr(handball, "probability", None) is not None
        else None
    )
    return {
        "verdict": final_verdict,
        "confidence": final_confidence,
        "reason": reason,
        "frames_analyzed": result.frames_analyzed,
        "player_tracks": result.player_tracks,
        "ball_tracks": result.ball_tracks,
        "peak_time": result.peak_time,
        "peak_frame_url": artifact_url(job_id, frame_path if frame_path.exists() else None),
        "annotated_video_url": artifact_url(job_id, video_path if video_path.exists() else None),
        "handball_overlay_url": artifact_url(
            job_id,
            handball_overlay_path if handball_overlay_path.exists() else None,
        ),
        "timeline": result.timeline,
        "metrics": {
            "mlp_foul": peak_row.get("scene_foul"),
            "roboflow_foul": peak_row.get("roboflow_foul"),
            "limb_gap": peak_row.get("limb_gap"),
            "contact_type": peak_row.get("contact_type"),
            "tackle_type": peak_row.get("tackle_type"),
            "metric_contact": peak_row.get("metric_contact", "NOT_RUN"),
            "handball_probability": handball_probability,
            "handball_quality": float(getattr(handball, "quality", 0.0)),
            "handball_status": str(handball.status),
            "handball_prediction": str(handball.predicted_label),
            "handball_ball_rate": getattr(handball, "ball_detection_rate", None),
            "handball_player_rate": getattr(handball, "player_detection_rate", None),
            "handball_pose_rate": getattr(handball, "pose_valid_rate", None),
            "handball_min_arm_distance": getattr(
                handball,
                "minimum_normalized_arm_distance",
                None,
            ),
            "handball_fold_probabilities": dict(
                getattr(handball, "fold_probabilities", {})
            ),
            "handball_frames": int(getattr(handball, "candidate_frames", 0)),
            "handball_hit_hand": bool(getattr(handball, "hit_hand", False)),
            "handball_part": getattr(handball, "handball_part", None),
            "handball_angle": getattr(handball, "handball_angle", None),
            "handball_proximity_override": bool(
                getattr(handball, "proximity_override", False)
            ),
            "handball_proximity_threshold": float(
                getattr(handball, "proximity_threshold", 0.04)
            ),
            "combined_decision_reason": decision_reason,
        },
        "report": (
            f"{result.report}\n\nHANDBALL DETECTION PROJECT\n"
            f"prediction: {getattr(handball, 'predicted_label', 'unavailable')}\n"
            f"probability: {handball_probability}\n"
            f"quality: {getattr(handball, 'quality', 0.0)}\n"
            f"ball hit arm: {getattr(handball, 'hit_hand', False)}\n"
            f"arm: {getattr(handball, 'handball_part', None)}\n"
            f"arm angle: {getattr(handball, 'handball_angle', None)}\n"
            f"proximity override: {getattr(handball, 'proximity_override', False)}\n"
            f"normalized arm gap: "
            f"{getattr(handball, 'minimum_normalized_arm_distance', None)}\n"
            f"direction-change frames: {getattr(handball, 'candidate_frames', 0)}\n"
            f"combined decision: {decision_reason}"
        ),
    }


def run_job(job_id: str, mode: str, media_path: Path, api_key: str) -> None:
    update_job(job_id, status="running", stage="Preparing local inference", progress=5)
    try:
        with MODEL_LOCK:
            result = (
                image_analysis(job_id, media_path, api_key)
                if mode == "image"
                else video_analysis(job_id, media_path, api_key)
            )
        update_job(
            job_id,
            status="complete",
            stage="Decision complete",
            progress=100,
            result=sanitize(result),
        )
    except Exception as error:
        traceback.print_exc()
        update_job(
            job_id,
            status="error",
            stage="Analysis failed",
            error=f"{type(error).__name__}: {error}",
        )
    finally:
        os.environ.pop("ROBOFLOW_API_KEY", None)
        media_path.unlink(missing_ok=True)


app = FastAPI(title="VAIR GitHub GPU Referee", version="github-main-gpu")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3200", "http://127.0.0.1:3200"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    mps = bool(torch.backends.mps.is_available())
    providers = ort.get_available_providers()
    return {
        "status": "ready" if mps else "gpu_unavailable",
        "notebook": NOTEBOOK.exists(),
        "source_repository": str(SOURCE_ROOT),
        "source_notebook": NOTEBOOK.name,
        "checkpoints": {name: path.exists() for name, path in WEIGHTS.items()},
        "handball": {
            "available": HANDBALL_PROJECT_ADAPTER.exists(),
            "source": "nadimra/handball-detection algorithm, modern MPS adapter",
            "method": "ball direction change + arm collision + arm angle",
            "detector": "YOLO11m on MPS",
            "pose": "YOLOv8m-Pose on MPS (modern HRNet replacement)",
            "proximity_override": {
                "normalized_arm_gap": 0.04,
                "minimum_ball_confidence": 0.35,
                "maximum_ball_radius_multiple": 1.5,
            },
            "archive_weights_included": False,
            "archive_training_dataset_included": False,
        },
        "acceleration": {
            "pytorch": "mps" if mps else "cpu",
            "rtmw": (
                "CoreMLExecutionProvider"
                if "CoreMLExecutionProvider" in providers
                else "CPUExecutionProvider"
            ),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        },
    }


@app.post("/api/analyze/{mode}")
async def begin_analysis(
    mode: str,
    file: UploadFile = File(...),
    api_key: str = Form(...),
) -> dict[str, str]:
    if mode not in {"image", "video"}:
        raise HTTPException(404, "Unknown analysis type")
    if not api_key.strip():
        raise HTTPException(400, "A private Roboflow API key is required.")

    job_id = uuid.uuid4().hex
    fallback = "incident.jpg" if mode == "image" else "incident.mp4"
    suffix = Path(file.filename or fallback).suffix
    upload = Path(tempfile.gettempdir()) / f"vair-{job_id}{suffix}"
    with upload.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            destination.write(chunk)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "stage": "Queued for local analysis",
            "progress": 1,
        }
    EXECUTOR.submit(run_job, job_id, mode, upload, api_key.strip())
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown analysis job")
        return sanitize(dict(job))


@app.get("/api/artifacts/{job_id}/{filename}")
def artifact(job_id: str, filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid artifact name")
    path = ARTIFACT_ROOT / job_id / filename
    if not path.exists():
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path)
