from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import streamlit as st
import streamlit.components.v1 as components

from handball_annotator.config import load_config
from handball_annotator.gallery import frame_gallery_html
from handball_annotator.negative_sampler import NegativeReviewStore, create_sample
from handball_annotator.storage import AnnotationStore

st.set_page_config(page_title="Non-handball sampler", page_icon="🎞️", layout="wide")


@st.cache_resource
def resources():
    config = load_config()
    review_root = config.state_dir.parent / "negative_sampler"
    candidate_root = review_root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    return config, AnnotationStore(config), NegativeReviewStore(review_root), candidate_root


def save_upload(upload, directory: Path) -> Path:
    safe_name = Path(upload.name).name
    digest = hashlib.sha1(f"{safe_name}:{upload.size}".encode("utf-8")).hexdigest()[:10]
    destination = directory / f"{Path(safe_name).stem}_{digest}{Path(safe_name).suffix.lower()}"
    if not destination.exists() or destination.stat().st_size != upload.size:
        destination.write_bytes(upload.getbuffer())
    return destination


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


config, annotation_store, review_store, candidate_root = resources()
st.title("🎞️ Non-handball clip sampler")
st.caption("Review spaced 41-frame windows from full matches and add clean legal-play examples to the existing dataset.")

with st.sidebar:
    st.header("1 · Full-match video")
    uploaded = st.file_uploader("Upload soccer footage", type=["mp4", "mov", "mkv", "avi", "m4v"])
    spacing_seconds = st.number_input("Seconds between samples", min_value=2.0, max_value=120.0,
                                      value=5.0, step=1.0)
    target = st.number_input("Target total not-handball clips", min_value=1, value=200, step=10)
    st.caption("The next window is extracted only after you review the current one.")

labels = annotation_store.labels()
not_handball_count = sum(label == "not_handball" for label in labels.values())
target_value = int(target)
metric_left, metric_right = st.columns(2)
metric_left.metric("Not-handball clips", not_handball_count)
metric_right.metric("Remaining to target", max(target_value - not_handball_count, 0))
st.progress(min(not_handball_count / target_value, 1.0))

if uploaded is None:
    st.info("Upload a full match to begin sampling. Existing not-handball data will remain unchanged.")
    st.stop()

source = save_upload(uploaded, config.uploads_dir)
total_frames, fps = video_properties(str(source), source.stat().st_mtime)
first_center = config.frames_before
last_center = total_frames - config.frames_after - 1
if last_center < first_center:
    st.error(f"This video contains {total_frames} frames; at least 41 are required.")
    st.stop()

source_key = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
cursor_key = f"negative_cursor_{source_key}"
pause_key = f"negative_paused_{source_key}"
decisions = review_store.decisions(source)
spacing_frames = max(1, round(float(spacing_seconds) * fps))
if cursor_key not in st.session_state:
    reviewed_centers = [int(item["center_frame"]) for item in decisions]
    st.session_state[cursor_key] = min((max(reviewed_centers) + spacing_frames) if reviewed_centers else first_center,
                                       last_center)
if pause_key not in st.session_state:
    st.session_state[pause_key] = False

st.subheader(source.name)
st.caption(
    f"{total_frames:,} frames · {fps:.2f} FPS · approximately {total_frames / fps / 60:.1f} minutes · "
    f"{len(decisions)} windows reviewed from this video"
)

control_left, control_middle, control_right = st.columns([1, 1, 2])
if control_left.button("⏹ Stop parsing", disabled=st.session_state[pause_key], use_container_width=True):
    st.session_state[pause_key] = True
    st.rerun()
if control_middle.button("▶ Resume", disabled=not st.session_state[pause_key], use_container_width=True):
    st.session_state[pause_key] = False
    st.rerun()
control_right.caption("Stopping pauses this video immediately. Your position and all decisions are saved.")

if st.session_state[pause_key]:
    st.warning("Parsing is stopped. Select Resume when you want another sample.")
elif int(st.session_state[cursor_key]) > last_center:
    st.success("You reached the end of this video. All complete 41-frame windows at the selected spacing were reviewed.")
else:
    center = max(int(st.session_state[cursor_key]), first_center)
    with st.spinner("Extracting the current 41-frame window…"):
        candidate = create_sample(source, center, config, candidate_root)
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    st.subheader(f"Current sample · {metadata['center_time_seconds']:.2f} seconds")
    st.caption("The 41-frame clip plays automatically and loops back to the beginning.")
    st.video(str(candidate / "clean.mp4"), autoplay=True, loop=True, muted=True)
    frames = sorted((candidate / "clean_frames").glob("*.jpg"))
    manual_preview_key = f"manual_frame_preview_{source_key}"
    if manual_preview_key not in st.session_state:
        st.session_state[manual_preview_key] = False
    preview_label = "Hide manual frame preview" if st.session_state[manual_preview_key] else "Manually preview individual frames"
    if st.button(preview_label, key=f"manual_preview_button_{source_key}"):
        st.session_state[manual_preview_key] = not st.session_state[manual_preview_key]
        st.rerun()
    if st.session_state[manual_preview_key]:
        components.html(frame_gallery_html(frames, config.frames_before), height=710, scrolling=False)

    add_col, reject_col, jump_col = st.columns([1, 1, 2])
    if add_col.button("✓ Add to not handball", type="primary", use_container_width=True):
        annotation_store.label(candidate, "not_handball")
        annotation_store.export_jsonl()
        review_store.set_decision(source, center, candidate.name, "accepted")
        st.session_state[cursor_key] = center + spacing_frames
        st.rerun()
    if reject_col.button("✕ Don’t add", use_container_width=True):
        previous = next((item for item in decisions if int(item["center_frame"]) == center), None)
        if previous and previous["decision"] == "accepted":
            annotation_store.unlabel(candidate.name)
            annotation_store.export_jsonl()
        review_store.set_decision(source, center, candidate.name, "rejected")
        st.session_state[cursor_key] = center + spacing_frames
        st.rerun()
    chosen_time = jump_col.number_input(
        "Jump to time (seconds)", min_value=0.0, max_value=max(total_frames / fps, 0.0),
        value=float(center / fps), step=5.0,
    )
    if jump_col.button("Jump", use_container_width=True):
        st.session_state[cursor_key] = min(max(round(float(chosen_time) * fps), first_center), last_center)
        st.rerun()

    if center >= last_center:
        st.info("This is the final complete 41-frame window in the video.")

decisions = review_store.decisions(source)
if decisions:
    st.divider()
    st.subheader("Review history")
    st.caption("Change any previous decision. Removing an accepted sample also removes it from dataset/not_handball.")
    selected_history = st.selectbox(
        "Previous sample",
        list(reversed(decisions)),
        format_func=lambda item: f"{int(item['center_frame']) / fps:.2f} s — {str(item['decision']).title()}",
        key=f"history_choice_{source_key}",
    )
    history_center = int(selected_history["center_frame"])
    history_id = str(selected_history["candidate_id"])
    status = str(selected_history["decision"])
    replacement = "rejected" if status == "accepted" else "accepted"
    history_change, history_open, _ = st.columns([1, 1, 2])
    button_text = "Remove from dataset" if status == "accepted" else "Add to not handball"
    if history_change.button(button_text, key=f"change_{source_key}_{history_center}", use_container_width=True):
        history_candidate = candidate_root / history_id
        if replacement == "accepted":
            annotation_store.label(history_candidate, "not_handball")
        else:
            annotation_store.unlabel(history_id)
        annotation_store.export_jsonl()
        review_store.set_decision(source, history_center, history_id, replacement)
        st.rerun()
    if history_open.button("Open this sample", key=f"review_{source_key}_{history_center}", use_container_width=True):
        st.session_state[cursor_key] = history_center
        st.session_state[pause_key] = False
        st.rerun()
