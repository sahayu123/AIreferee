"""
AI Referee -- upload-and-classify web app.

Lets you drag-and-drop any clip(s) of an action and get the trained foul
classifier's prediction (plus foul type and predicted card, if those heads
were trained), and an annotated video (player boxes, pose skeleton,
"possible contact" highlighting, and a prediction banner) rendered inline in
the browser.

Reuses every heavy-lifting piece from predict_foul.py (feature extraction,
fusion/prediction math, video annotation) -- this file is just the upload UI
wrapped around that existing pipeline, so training/inference math can never
drift between the CLI tool and the app.

Run (from .venv-mediapipe, the same env predict_foul.py needs since MediaPipe
has no wheels for the main project's Python):
    .\\.venv-mediapipe\\Scripts\\python -m streamlit run app.py
Then open the URL Streamlit prints (usually http://localhost:8501).
"""

import time
import tempfile
import uuid
from pathlib import Path

import cv2
import streamlit as st

from predict_foul import (
    MODEL_PATH, FOUL_COLOR, NO_FOUL_COLOR, INFO_COLOR, RENDERED_VIDEOS_DIR,
    build_extractor, build_player_detector, fuse_and_predict, predict_card_and_type,
    load_model_bundle, render_annotated_video,
)
from extract_pose_features import build_pose_landmarker

st.set_page_config(page_title="AI Referee", page_icon="🟥", layout="centered")

UPLOAD_DIR = Path(tempfile.gettempdir()) / "ai_referee_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CLEANUP_MAX_AGE_SECONDS = 24 * 60 * 60


@st.cache_resource(show_spinner=False)
def cleanup_old_files():
    """Sweeps RENDERED_VIDEOS_DIR + UPLOAD_DIR of anything older than 24h --
    both directories otherwise grow by one file per classify click forever.
    Guarded by st.cache_resource so this only runs once per app process, not
    on every Streamlit rerun (the whole script reruns on every interaction)."""
    now = time.time()
    for directory in (RENDERED_VIDEOS_DIR, UPLOAD_DIR):
        if not directory.exists():
            continue
        for f in directory.glob("*"):
            try:
                if f.is_file() and (now - f.stat().st_mtime) > CLEANUP_MAX_AGE_SECONDS:
                    f.unlink()
            except OSError:
                pass   # best-effort cleanup -- not worth failing the app over
    return True


@st.cache_resource(show_spinner="Loading model + CNN backbone (first run only)...")
def load_pipeline():
    """Returns (pipeline_dict, load_error). load_error is a user-facing
    string when model.pkl exists but fails the feature-schema check (Track
    1.1's load_model_bundle() catches a stale model the instant it loads,
    instead of a confusing crash deep inside feature extraction)."""
    if not MODEL_PATH.exists():
        return None, None
    try:
        bundle = load_model_bundle()
    except RuntimeError as e:
        return None, str(e)
    cnn_model, preprocess, device = build_extractor()
    bundle["cnn_model"] = cnn_model
    bundle["preprocess"] = preprocess
    bundle["device"] = device
    return bundle, None


@st.cache_resource(show_spinner=False)
def load_detector(min_score: float):
    return build_player_detector(min_score)


cleanup_old_files()

st.title("🟥 AI Referee")
st.caption("Upload one action's camera-angle clip(s) and get the trained model's foul/no-foul call.")

pipeline, load_error = load_pipeline()
if load_error:
    st.error(f"Couldn't load the trained model: {load_error}")
    st.stop()
if pipeline is None:
    st.error(f"No trained model found at `{MODEL_PATH}`. Run train_foul_rf.py first.")
    st.stop()

with st.sidebar:
    st.header("Options")
    min_score = st.slider(
        "Player-detection confidence", 0.05, 0.9, 0.3, 0.05,
        help="MediaPipe person-detector threshold used for the cosmetic player boxes/contact "
             "highlighting in the rendered video (does not affect the prediction itself).",
    )
    st.markdown("---")
    st.markdown(
        "**Tip:** the model expects fused multi-view input -- for the real prediction, upload "
        "every available camera-angle clip of the same action (2-4 views). A single clip is "
        "accepted too, as a degraded single-view approximation."
    )

uploaded_files = st.file_uploader(
    "Drop clip(s) here (same action, different camera angles)",
    type=["mp4", "mov", "avi", "mkv", "webm"],
    accept_multiple_files=True,
)

classify_clicked = st.button("Classify", type="primary", disabled=not uploaded_files)

if classify_clicked and uploaded_files:
    tmp_paths = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        dest.write_bytes(f.getbuffer())
        tmp_paths.append(dest)

    try:
        # Fail fast on a corrupt/non-video upload before running the full
        # CNN+pose pipeline on it.
        for tmp_path in tmp_paths:
            cap = cv2.VideoCapture(str(tmp_path))
            ok = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
            cap.release()
            if not ok:
                st.error(f"Couldn't read **{tmp_path.name}** as a video -- is this a valid, "
                         f"non-corrupt video file?")
                st.stop()

        try:
            with st.spinner(f"Extracting features from {len(tmp_paths)} clip(s)..."):
                with build_pose_landmarker() as pose_landmarker:
                    pred, prob, label, X_fused = fuse_and_predict(
                        tmp_paths, pipeline["cnn_model"], pipeline["preprocess"], pipeline["device"],
                        pose_landmarker, pipeline, warn=False,
                    )
                    foul_type_label = foul_type_prob = card_label = card_prob = None
                    if pred:
                        foul_type_label, foul_type_prob, card_label, card_prob = predict_card_and_type(
                            X_fused, pipeline)

            if pred:
                st.error(f"## Prediction: FOUL  ({prob:.0%} foul probability)")
            else:
                st.success(f"## Prediction: NO FOUL  ({prob:.0%} foul probability)")
            st.progress(prob)

            if foul_type_label is not None:
                st.markdown(f"**Foul type:** {foul_type_label} ({foul_type_prob:.0%} confidence)")
            if card_label is not None:
                st.markdown(f"**Predicted card:** {card_label} ({card_prob:.0%} confidence)")

            if len(tmp_paths) == 1:
                st.warning(
                    "Single-view input -- this is a degraded approximation of the model's true "
                    "multi-view fusion. For the real prediction, upload every camera angle of the "
                    "same action."
                )

            st.markdown(f"*Threshold: {pipeline['threshold']:.2%} · fused from {len(tmp_paths)} view(s)*")

            box_detector = load_detector(min_score)
            banner = [(f"Predicted: {label} ({prob:.0%})", FOUL_COLOR if pred else NO_FOUL_COLOR)]
            if foul_type_label is not None:
                banner.append((f"Foul type: {foul_type_label} ({foul_type_prob:.0%})", INFO_COLOR))
            if card_label is not None:
                banner.append((f"Predicted card: {card_label} ({card_prob:.0%})", INFO_COLOR))

            st.markdown("---")
            st.subheader("Annotated playback")
            for tmp_path in tmp_paths:
                with st.spinner(f"Rendering annotated video for {tmp_path.name}..."):
                    out_path = render_annotated_video(tmp_path, box_detector, banner)
                st.video(str(out_path))
        except Exception as e:
            st.error(f"Something went wrong processing this clip: {e}")
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)
