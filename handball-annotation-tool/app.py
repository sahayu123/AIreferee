from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from handball_annotator.config import load_config
from handball_annotator.gallery import frame_gallery_html
from handball_annotator.miner import mine_video
from handball_annotator.storage import AnnotationStore

st.set_page_config(page_title="Handball Annotation Tool", page_icon="⚽", layout="wide")


@st.cache_resource
def resources():
    config = load_config()
    return config, AnnotationStore(config)


def candidate_directories(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/metadata.json"))


def save_upload(upload, directory: Path) -> Path:
    safe_name = Path(upload.name).name
    digest = hashlib.sha1(f"{safe_name}:{upload.size}".encode("utf-8")).hexdigest()[:10]
    destination = directory / f"{Path(safe_name).stem}_{digest}{Path(safe_name).suffix.lower()}"
    if not destination.exists() or destination.stat().st_size != upload.size:
        destination.write_bytes(upload.getbuffer())
    return destination


def video_title(source_name: str) -> str:
    stem = Path(source_name).stem
    return re.sub(r"_[0-9a-f]{10}$", "", stem)


config, store = resources()
st.title("⚽ Handball candidate annotation")
st.caption("Mine possible ball-to-arm events from long soccer videos, then label the clean clips.")

with st.sidebar:
    st.header("1 · Add a video")
    uploaded = st.file_uploader("Upload soccer footage", type=["mp4", "mov", "mkv", "avi", "m4v"])
    source = None
    if uploaded is not None:
        source = save_upload(uploaded, config.uploads_dir)
        st.success(f"Ready: {source.name}")
        if st.button("Find contact candidates", type="primary", use_container_width=True):
            bar = st.progress(0.0)
            status = st.empty()

            def update_progress(value: float, message: str) -> None:
                bar.progress(value)
                status.caption(message)

            try:
                with st.spinner("Loading models and analyzing the video…"):
                    found = mine_video(source, config, update_progress)
                st.success(f"Found {len(found)} candidate events.")
                st.session_state.active_video = source.name
                st.session_state[f"candidate_index_{source.name}"] = 0
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.header("Review settings")
    show_evidence = st.toggle("Show detection evidence", value=True,
                              help="Only changes the preview. Saved training data is always clean.")
    show_frames = st.toggle("Show frame viewer", value=True)
    st.caption("Labels and progress are saved automatically.")

candidates = candidate_directories(config.candidates_dir)
labels = store.labels()
unlabeled = sum(path.name not in labels for path in candidates)
counts = {label: sum(value == label for value in labels.values()) for label in ("handball", "not_handball", "uncertain")}

metric_columns = st.columns(4)
metric_columns[0].metric("Unlabeled", unlabeled)
metric_columns[1].metric("Handball", counts["handball"])
metric_columns[2].metric("Not handball", counts["not_handball"])
metric_columns[3].metric("Uncertain", counts["uncertain"])

if not candidates:
    st.info("Upload a soccer video and select **Find contact candidates** to begin.")
    st.stop()

candidate_metadata = {
    path: json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in candidates
}
videos = list(dict.fromkeys(metadata["source_name"] for metadata in candidate_metadata.values()))
if st.session_state.get("active_video") not in videos:
    st.session_state.active_video = videos[-1]
active_video = st.segmented_control(
    "Video tabs", videos, format_func=video_title, key="active_video", selection_mode="single", required=True
)
video_candidates = [path for path in candidates if candidate_metadata[path]["source_name"] == active_video]
index_key = f"candidate_index_{active_video}"
if index_key not in st.session_state:
    st.session_state[index_key] = next((i for i, path in enumerate(video_candidates) if path.name not in labels), 0)
st.session_state[index_key] = min(st.session_state[index_key], len(video_candidates) - 1)
index = st.session_state[index_key]
candidate = video_candidates[index]
metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))

navigation_left, navigation_center, navigation_right = st.columns([1, 3, 1])
if navigation_left.button("← Previous candidate", disabled=index == 0, use_container_width=True):
    st.session_state[index_key] -= 1
    st.rerun()
navigation_center.markdown(
    f"<div style='text-align:center'><b>Candidate {index + 1} of {len(video_candidates)}</b><br>"
    f"{metadata['source_name']} · {metadata['center_time_seconds']:.2f} seconds</div>", unsafe_allow_html=True
)
if navigation_right.button("Next candidate →", disabled=index == len(video_candidates) - 1, use_container_width=True):
    st.session_state[index_key] += 1
    st.rerun()

current_label = labels.get(candidate.name)
if current_label:
    st.info(f"Current label: **{current_label.replace('_', ' ').title()}**. You can change it below.")

video_path = candidate / ("evidence.mp4" if show_evidence else "clean.mp4")
st.video(str(video_path))

if show_frames:
    frame_root = candidate / ("evidence_frames" if show_evidence else "clean_frames")
    frames = sorted(frame_root.glob("*.jpg"))
    st.subheader("Frames around the possible contact")
    st.caption("Use the side arrows, keyboard arrow keys, thumbnails, or the fullscreen button.")
    components.html(frame_gallery_html(frames, int(metadata["frames_before"])), height=710, scrolling=False)

with st.expander("Detection details"):
    st.json(metadata)

st.subheader("Is this a handball?")
handball_col, no_col, uncertain_col = st.columns(3)


def apply_label(value: str) -> None:
    store.label(candidate, value)
    store.export_jsonl()
    if st.session_state[index_key] < len(video_candidates) - 1:
        st.session_state[index_key] += 1
    st.rerun()


if handball_col.button("✓ Handball", type="primary", use_container_width=True):
    apply_label("handball")
if no_col.button("✕ Not handball", use_container_width=True):
    apply_label("not_handball")
if uncertain_col.button("? Uncertain", use_container_width=True):
    apply_label("uncertain")

st.caption("The labeled dataset contains clean clips and clean frames only. Boxes, skeletons, paths, and distances are never copied into it.")
