from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from handball_annotator.config import load_config
from handball_annotator.gallery import frame_gallery_html
from handball_annotator.miner import create_manual_candidate_from_video, mine_video
from handball_annotator.similarity import find_handball_duplicates
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


@st.cache_data(show_spinner=False)
def video_properties(path: str, modified_time: float) -> tuple[int, float]:
    del modified_time
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Video cannot be opened: {path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    capture.release()
    return total, fps


@st.cache_data(show_spinner=False, max_entries=64)
def preview_frame(path: str, frame_index: int, modified_time: float):
    del modified_time
    capture = cv2.VideoCapture(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


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


def render_manual_selector(source_path: Path, *, expanded: bool = False) -> None:
    """Let the annotator create a candidate without relying on a detection."""
    with st.expander("Manually choose a 41-frame window", expanded=expanded):
        st.write("Choose the center frame anywhere in this video. The saved candidate contains 20 frames before and 20 after it.")
        if not source_path.is_file():
            st.error(f"The original uploaded video is missing: {source_path}")
            return
        try:
            total_frames, source_fps = video_properties(str(source_path), source_path.stat().st_mtime)
            first_center = config.frames_before
            last_center = total_frames - config.frames_after - 1
            if last_center < first_center:
                st.error(f"This video has only {total_frames} frames; at least {config.frames_before + config.frames_after + 1} are required.")
                return
            selected_center = st.slider(
                "Center frame", first_center, last_center, first_center, 1,
                key=f"manual_center_{source_path.name}",
                help="Frame numbering below is shown starting from 1.",
            )
            st.caption(
                f"Selected frame {selected_center + 1:,} of {total_frames:,} "
                f"({selected_center / source_fps:.2f} s) · saved window: "
                f"frames {selected_center - config.frames_before + 1:,}–{selected_center + config.frames_after + 1:,}"
            )
            st.image(
                preview_frame(str(source_path), selected_center, source_path.stat().st_mtime),
                caption=f"Selected center frame {selected_center + 1:,}", use_container_width=True,
            )
            if st.button("Add this 41-frame window", key=f"manual_add_{source_path.name}", use_container_width=True):
                created = create_manual_candidate_from_video(source_path, selected_center, config)
                st.session_state.active_video = source_path.name
                st.session_state.focus_candidate = created.name
                st.success("Manual candidate added. Opening it now…")
                st.rerun()
        except Exception as exc:
            st.error(str(exc))


if source is not None:
    st.subheader(f"Manually select from: {video_title(source.name)}")
    st.caption("Use this when detection misses the contact, or when you want to choose the exact moment yourself.")
    render_manual_selector(source, expanded=True)

if not candidates:
    if source is None:
        st.info("Upload a soccer video to find candidates or select a frame manually.")
    else:
        st.info("No detected candidates are available. Use the manual selector above.")
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
focus_candidate = st.session_state.pop("focus_candidate", None)
if focus_candidate:
    focus_index = next((i for i, path in enumerate(video_candidates) if path.name == focus_candidate), None)
    if focus_index is not None:
        st.session_state[index_key] = focus_index
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

duplicate_key = f"duplicate_matches_{candidate.name}"
with st.sidebar:
    st.divider()
    st.header("Already used?")
    st.caption("Compare this candidate with clips already labeled Handball.")
    if st.button("Check handball dataset", key=f"duplicate_button_{candidate.name}", use_container_width=True):
        with st.spinner("Comparing frame sequences…"):
            st.session_state[duplicate_key] = [
                match.to_dict() for match in find_handball_duplicates(candidate, config.dataset_dir)
            ]

if duplicate_key in st.session_state:
    duplicate_matches = st.session_state[duplicate_key]
    if duplicate_matches:
        st.warning(f"This may have been used before. Found {len(duplicate_matches)} possible handball-dataset match(es).")
        st.dataframe([
            {
                "Possible existing candidate": match["candidate_id"],
                "Source video": video_title(str(match["source_name"])),
                "Time": f"{float(match['center_time_seconds']):.2f} s",
                "Similarity": f"{float(match['similarity']):.1%}",
                "Reason": match["reason"],
            }
            for match in duplicate_matches
        ], hide_index=True, use_container_width=True)
    else:
        st.success("No likely duplicate was found among clips currently labeled Handball.")

video_path = candidate / ("evidence.mp4" if show_evidence else "clean.mp4")
st.video(str(video_path))

if show_frames:
    frame_root = candidate / ("evidence_frames" if show_evidence else "clean_frames")
    frames = sorted(frame_root.glob("*.jpg"))
    st.subheader("Frames around the possible contact")
    st.caption("Use the side arrows, keyboard arrow keys, thumbnails, or the fullscreen button.")
    components.html(frame_gallery_html(frames, int(metadata["frames_before"])), height=710, scrolling=False)

candidate_source = Path(metadata["source_path"])
if source is None or candidate_source.resolve() != source.resolve():
    render_manual_selector(candidate_source)

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
