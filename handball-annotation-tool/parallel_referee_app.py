from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from combined_pipeline.pipeline import (
    PROJECT_ROOT,
    ParallelRefereePipeline,
)


CONFIG = PROJECT_ROOT / "configs/parallel_pipeline.yaml"
UPLOADS = PROJECT_ROOT / "artifacts/parallel_runs/uploads"


st.set_page_config(
    page_title="Parallel AI Referee",
    page_icon="⚽",
    layout="wide",
)


@st.cache_resource
def load_pipeline(config_path: str, modified_ns: int):
    del modified_ns
    return ParallelRefereePipeline.from_config(config_path)


def save_upload(name: str, content: bytes) -> Path:
    digest = hashlib.sha256(content).hexdigest()[:16]
    suffix = Path(name).suffix.lower() or ".mp4"
    destination = UPLOADS / f"{digest}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        destination.write_bytes(content)
    return destination


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return frames / max(fps, 1e-6)


def availability_badge(section: dict[str, object]) -> str:
    return "✅ Ready" if section["available"] else "⚠️ Unavailable"


st.title("Parallel handball + general-foul referee")
st.caption(
    "The original 12-frame handball GRU and the general-foul prototype run "
    "as independent specialists. Their internal features are never mixed."
)

if not CONFIG.is_file():
    st.error(f"Missing configuration: {CONFIG}")
    st.stop()

pipeline = load_pipeline(str(CONFIG), CONFIG.stat().st_mtime_ns)
preflight = pipeline.preflight()

with st.sidebar:
    st.header("Model readiness")
    for key, title in (
        ("handball", "Handball GRU"),
        ("general_foul", "General foul"),
    ):
        section = preflight[key]
        st.markdown(f"**{title}:** {availability_badge(section)}")
        for issue in section.get("issues", []):
            st.caption(str(issue))
uploaded = st.file_uploader(
    "Upload a short incident video",
    type=["mp4", "mov", "m4v", "avi", "mkv"],
)
if uploaded is None:
    st.info("Upload a video to run both specialists.")
    st.stop()

source = save_upload(uploaded.name, uploaded.getvalue())
duration = video_duration(source)
st.video(str(source))
if duration > 0:
    default_time = duration / 2
    incident_time = st.slider(
        "Center of the incident (seconds)",
        min_value=0.0,
        max_value=float(duration),
        value=float(default_time),
        step=max(0.01, min(0.10, duration / 100)),
        help=(
            "Both models receive the same 41-frame window centered here. "
            "For an already trimmed clip, leave this near the contact."
        ),
    )
else:
    incident_time = None

if not st.button(
    "Run both models",
    type="primary",
    use_container_width=True,
):
    st.stop()

messages: list[str] = []
with st.status("Running independent specialists…", expanded=True) as status:
    log = st.empty()
    events: queue.Queue[str] = queue.Queue()
    outcome: dict[str, object] = {}

    def update(message: str) -> None:
        events.put(message)

    def run_pipeline() -> None:
        try:
            outcome["result"] = pipeline.run(
                source,
                incident_time_seconds=incident_time,
                progress=update,
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(
        target=run_pipeline,
        name="parallel-referee-ui-run",
        daemon=True,
    )
    worker.start()
    while worker.is_alive() or not events.empty():
        while True:
            try:
                messages.append(events.get_nowait())
            except queue.Empty:
                break
        log.code("\n".join(messages[-18:]), language=None)
        if worker.is_alive():
            time.sleep(0.20)
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    result = outcome["result"]
    status.update(
        label=f"Finished: {result.final_label.value.replace('_', ' ').title()}",
        state="complete",
        expanded=True,
    )

label_column, handball_column, foul_column, mode_column = st.columns(4)
label_column.metric(
    "Final decision",
    result.final_label.value.replace("_", " ").upper(),
    f"{result.confidence:.1%}",
)
handball_column.metric(
    "Handball probability",
    (
        f"{result.handball.probability:.1%}"
        if result.handball.probability is not None
        else "Unavailable"
    ),
)
foul_column.metric(
    "General-foul probability",
    (
        f"{result.general_foul.probability:.1%}"
        if result.general_foul.probability is not None
        else "Unavailable"
    ),
)
mode_column.metric(
    "Execution",
    result.execution_mode.title(),
    "Partial" if result.partial_result else "Both completed",
)

if result.partial_result:
    st.warning(
        "This is a partial-system result. Inspect the model readiness panel "
        "and specialist errors before relying on it."
    )
st.write("Decision rule:", result.decision_reason)

handball_tab, foul_tab, evidence_tab = st.tabs(
    ["Handball specialist", "General-foul specialist", "Raw result"]
)
with handball_tab:
    st.write(
        {
            "status": result.handball.status,
            "prediction": result.handball.predicted_label,
            "probability": result.handball.probability,
            "quality": result.handball.quality,
            "fold_probabilities": result.handball.fold_probabilities,
            "peak_frame": result.handball.peak_frame,
        }
    )
    if (
        result.handball.overlay_path
        and Path(result.handball.overlay_path).is_file()
    ):
        st.image(
            result.handball.overlay_path,
            caption="The 12 frames used by the original 56-feature GRU",
            width="stretch",
        )
    if result.handball.error:
        st.error(result.handball.error)

with foul_tab:
    st.write(
        {
            "status": result.general_foul.status,
            "prediction": result.general_foul.predicted_label,
            "probability": result.general_foul.probability,
            "contact_type": result.general_foul.contact_type,
            "tackle_type": result.general_foul.tackle_type,
            "peak_time": result.general_foul.peak_time,
        }
    )
    if (
        result.general_foul.annotated_video_path
        and Path(result.general_foul.annotated_video_path).is_file()
    ):
        st.video(result.general_foul.annotated_video_path)
    if result.general_foul.timeline:
        timeline = pd.DataFrame(result.general_foul.timeline)
        chart_columns = [
            name
            for name in ("scene_foul", "pair_crop_foul", "evidence")
            if name in timeline.columns
        ]
        if chart_columns and "time" in timeline:
            st.line_chart(
                timeline.set_index("time")[chart_columns],
                y=chart_columns,
            )
    if result.general_foul.error:
        st.error(result.general_foul.error)

with evidence_tab:
    encoded = json.dumps(result.to_dict(), indent=2)
    st.json(result.to_dict())
    st.download_button(
        "Download combined_result.json",
        data=encoded,
        file_name="combined_result.json",
        mime="application/json",
    )
