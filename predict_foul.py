"""
Run the trained foul classifier.

Two modes:

1. Evaluate on a random slice of unseen data (default, no args):
   python predict_foul.py
   Each run randomly samples a fresh chunk of the held-out clips that
   train_foul_rf.py never trained on, and reports accuracy on it. The exact
   clips (and therefore the exact accuracy) change run to run, but since
   they're all drawn from the same held-out pool the numbers should stay
   close together -- that consistency is what shows the model is precise.

   Optional: python predict_foul.py --fraction 0.4   (default 0.5, i.e. half
   of the held-out set each run)
   Optional: python predict_foul.py --show 8   (how many of the sampled
   clips to render+play as annotated videos, default 5; use --show 0 to disable)

2. Predict one action from its camera-angle clip(s):
   python predict_foul.py "C:\\path\\to\\action_5\\clip_0.mp4" "C:\\path\\to\\action_5\\clip_1.mp4"
   The model expects a fused multi-view input (mean+max pooled across an
   action's 2-4 camera-angle clips, matching train_foul_rf.py) -- pass every
   available view of the same action for the real prediction. A single path
   is also accepted as a degraded single-view approximation (that one view's
   features are duplicated as both "mean" and "max").

   Optional: python predict_foul.py --min-score 0.3   (MediaPipe player-detection
   confidence threshold, default 0.3 -- frames are decoded at native video
   resolution now, not 224x224 crops, so the usual COCO EfficientDet range works)

Every shown clip is rendered as a full annotated MP4 -- every native frame of
the source clip (not just the handful sampled for classification), at its
original fps -- with MediaPipe's person-detection boxes, pose skeletons, and
a "possible contact" tag drawn on every frame, plus a burned-in banner
showing this model's prediction (and ground truth + correct/wrong, color
coded, in holdout-preview mode where it's known). The video is written to
rendered_videos/ and opened automatically in the default video player.

MediaPipe pose-landmark features (limb/contact proximity, not just the
cosmetic boxes above) are also real classifier input now: build_feature_vector()
below concatenates them onto the CNN embedding in the same order
train_foul_rf.py does, via the shared math in extract_pose_features.py.

This script needs its own environment: MediaPipe has no Windows wheels past
Python 3.12, so it can't share the main project's (3.14) environment. Set up
and run it via:
    py -3.12 -m venv .venv-mediapipe
    .\\.venv-mediapipe\\Scripts\\pip install -r requirements-test.txt
    .\\.venv-mediapipe\\Scripts\\python predict_foul.py
train_foul_rf.py is unaffected and keeps using the main/training environment.
"""

import os
import sys
import time
import argparse
import urllib.request
import cv2
import joblib
import numpy as np
import torch
import torchvision.models.video as tvv

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    classification_report, confusion_matrix,
)

from extract_pose_features import build_pose_landmarker, compute_pose_features_from_frames
from pose_feature_config import NUM_POSE_FEATURES

OUT_DIR = Path(os.environ.get("FOUL_OUT_DIR", str(Path(__file__).resolve().parent)))
MODEL_PATH = OUT_DIR / "model.pkl"
HOLDOUT_PATH = OUT_DIR / "holdout_test.pkl"
BACKBONE_CKPT = OUT_DIR / "finetuned_backbone.pt"
MAX_FRAMES_PER_ACTION = 8   # must match train_foul_rf.py
RENDERED_VIDEOS_DIR = OUT_DIR / "rendered_videos"

DETECTOR_MODEL_DIR = OUT_DIR / "mediapipe_models"
DETECTOR_MODEL_PATH = DETECTOR_MODEL_DIR / "efficientdet_lite0.tflite"
DETECTOR_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/object_detector/"
                       "efficientdet_lite0/int8/latest/efficientdet_lite0.tflite")

# Standard 33-landmark BlazePose skeleton connections (index pairs) -- used
# only to draw the pose overlay in rendered videos. This mediapipe build has
# no mp.solutions legacy API to pull the constant from, so it's hardcoded.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# ── Presentation color palette (RGB / RGBA for translucent fills) ──────────
PLAYER_BOX_COLOR = (235, 235, 235)         # clean near-white -- normal detection
CONTACT_BOX_COLOR = (255, 99, 71)           # alert orange-red -- box in a contact pair
CHIP_BG_PLAYER = (24, 27, 34, 195)          # translucent charcoal label backing
CHIP_BG_CONTACT = (255, 99, 71, 220)        # translucent alert-color label backing
SKELETON_COLOR = (86, 220, 255)             # cyan-blue "tech" pose overlay
BANNER_BG = (14, 16, 22, 215)               # translucent near-black bar
FOUL_COLOR = (255, 92, 92)                  # traffic-light red
NO_FOUL_COLOR = (100, 220, 130)             # traffic-light green
CORRECT_COLOR = (100, 220, 130)
WRONG_COLOR = (255, 92, 92)
WATERMARK_COLOR = (255, 255, 255, 130)
INFO_COLOR = (235, 200, 90)                 # amber -- neutral info lines (foul type / card), not a verdict


def build_extractor():
    """r3d_18 with its weights swapped for the fully fine-tuned backbone
    train_foul_rf.py trained end-to-end on our own foul/no-foul clips."""
    weights = tvv.R3D_18_Weights.KINETICS400_V1
    model = tvv.r3d_18(weights=weights)
    if BACKBONE_CKPT.exists():
        state = torch.load(BACKBONE_CKPT, map_location="cpu")
        backbone_state = {k: v for k, v in state.items()
                           if k.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))}
        model.load_state_dict(backbone_state, strict=False)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), weights.transforms(), device


def ensure_detector_model():
    """Download MediaPipe's EfficientDet-Lite0 object-detection model on first run
    (cached under mediapipe_models/ afterwards)."""
    DETECTOR_MODEL_DIR.mkdir(exist_ok=True)
    if not DETECTOR_MODEL_PATH.exists():
        print(f"[*] Downloading person-detection model -> {DETECTOR_MODEL_PATH}")
        try:
            urllib.request.urlretrieve(DETECTOR_MODEL_URL, DETECTOR_MODEL_PATH)
        except Exception as e:
            print(f"[!] Download failed ({e}). Manually download this file and save it to "
                  f"{DETECTOR_MODEL_PATH}:\n    {DETECTOR_MODEL_URL}")
            sys.exit(1)
    return DETECTOR_MODEL_PATH


def build_player_detector(min_score: float):
    options = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_detector_model())),
        max_results=10,
        score_threshold=min_score,
        category_allowlist=["person"],
    )
    return vision.ObjectDetector.create_from_options(options)


def boxes_overlap(box_a, box_b, iou_threshold: float = 0.15) -> bool:
    """IoU-based overlap check between two (x0,y0,x1,y1) boxes -- a simple
    heuristic for flagging possible player contact, not a foul classifier."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return False
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return (inter / union if union > 0 else 0.0) >= iou_threshold


def detect_players_and_contact(frame_rgb: np.ndarray, detector):
    """Pure detection, no drawing: returns (boxes: list[(x0,y0,x1,y1,score)],
    box_in_contact: list[bool] -- per-box, True if that box overlaps any
    other detected box, contact: bool -- True if any pair overlaps at all)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    boxes = []
    for detection in result.detections:
        box = detection.bounding_box
        x0, y0 = box.origin_x, box.origin_y
        x1, y1 = x0 + box.width, y0 + box.height
        score = detection.categories[0].score
        boxes.append((x0, y0, x1, y1, score))
    n = len(boxes)
    box_in_contact = [False] * n
    contact = False
    for i in range(n):
        for j in range(i + 1, n):
            if boxes_overlap(boxes[i][:4], boxes[j][:4]):
                contact = True
                box_in_contact[i] = True
                box_in_contact[j] = True
    return boxes, box_in_contact, contact


_font_cache = {}


def _get_font(kind: str, size: int):
    """kind: "bold" or "regular". Cached, with graceful fallback down to
    PIL's built-in bitmap font if no system fonts are found (e.g. non-Windows)."""
    key = (kind, size)
    if key not in _font_cache:
        candidates = {
            "bold": ["seguisb.ttf", "arialbd.ttf", "arial.ttf"],
            "regular": ["segoeui.ttf", "arial.ttf"],
        }[kind]
        font = None
        for name in candidates:
            try:
                font = ImageFont.truetype(name, size)
                break
            except Exception:
                continue
        _font_cache[key] = font or ImageFont.load_default()
    return _font_cache[key]


def draw_player_overlay(overlay: Image.Image, boxes, box_in_contact) -> Image.Image:
    """Draw a rounded box around every detected player -- white for a normal
    detection, alert orange-red (with a "CONTACT" label chip instead of
    "PLAYER") for any box overlapping another, so contact is visually
    obvious at a glance rather than a small corner caption. Draws onto a
    transparent RGBA overlay (composited by the caller) so label chips can
    be genuinely translucent instead of a flat opaque block."""
    draw = ImageDraw.Draw(overlay)
    font = _get_font("bold", 14)
    for (x0, y0, x1, y1, _score), in_contact in zip(boxes, box_in_contact):
        color = CONTACT_BOX_COLOR if in_contact else PLAYER_BOX_COLOR
        draw.rounded_rectangle([x0, y0, x1, y1], radius=6, outline=color, width=4 if in_contact else 3)

        label = "CONTACT" if in_contact else "PLAYER"
        chip_bg = CHIP_BG_CONTACT if in_contact else CHIP_BG_PLAYER
        text_w = draw.textlength(label, font=font)
        pad = 5
        chip_x0, chip_y1 = x0, max(18, y0)
        chip_y0 = chip_y1 - 20
        chip_x1 = chip_x0 + text_w + 2 * pad
        draw.rounded_rectangle([chip_x0, chip_y0, chip_x1, chip_y1], radius=5, fill=chip_bg)
        draw.text((chip_x0 + pad, chip_y0 + 2), label, font=font, fill=(255, 255, 255, 255))
    return overlay


def draw_pose_skeleton(overlay: Image.Image, pose_result) -> Image.Image:
    """Draw MediaPipe's pose landmarks + skeleton connections for every
    detected person, in cyan -- a distinct visual layer from the player
    boxes, so you can see exactly what the pose-contact features (used by
    the classifier) are derived from."""
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    for landmarks in pose_result.pose_landmarks:
        pts = [(lm.x * w, lm.y * h) for lm in landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                draw.line([pts[a], pts[b]], fill=SKELETON_COLOR, width=2)
        for x, y in pts:
            draw.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=SKELETON_COLOR)
    return overlay


def draw_banner(overlay: Image.Image, banner_lines) -> Image.Image:
    """Persistent, semi-transparent bottom banner burned into every frame,
    with rounded top corners and a small color-coded status dot per line
    (e.g. red for FOUL / green for NO FOUL, or correct/wrong on the
    ground-truth line). banner_lines is a list of (text, (r,g,b)) tuples,
    one per line -- callers decide content and colors (this function is
    display-only, mode-agnostic). Draws onto a transparent RGBA overlay
    (composited by the caller) for a real translucent look rather than a
    flat opaque bar."""
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    line_height = 32
    padding_v = 12
    padding_h = 18
    dot_radius = 6
    bar_height = line_height * len(banner_lines) + 2 * padding_v
    bar_top = h - bar_height
    # Extend well past the bottom edge so only the top corners read as rounded.
    draw.rounded_rectangle([0, bar_top, w, h + 20], radius=16, fill=BANNER_BG)
    for i, (text, color) in enumerate(banner_lines):
        font = _get_font("bold" if i == 0 else "regular", 23 if i == 0 else 18)
        line_top = bar_top + padding_v + i * line_height
        cy = line_top + line_height / 2 - 3
        cx = padding_h + dot_radius
        draw.ellipse([cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius], fill=color)
        draw.text((padding_h + 2 * dot_radius + 12, line_top), text, font=font, fill=(255, 255, 255, 255))
    return overlay


def draw_watermark(overlay: Image.Image) -> Image.Image:
    """Small, unobtrusive top-left branding mark -- keeps rendered clips
    identifiable when shown standalone in a presentation."""
    draw = ImageDraw.Draw(overlay)
    font = _get_font("bold", 15)
    draw.text((14, 12), "AI REFEREE", font=font, fill=WATERMARK_COLOR)
    return overlay


def render_annotated_video(video_path: Path, box_detector, banner_lines, out_dir=None) -> Path:
    """Decode EVERY native frame of video_path (not the handful sampled for
    classification), draw player boxes + pose skeleton + "possible contact"
    + the persistent prediction/ground-truth banner on each, and write an
    MP4 at the clip's original fps. Returns the output path.

    Builds its own VIDEO-mode PoseLandmarker internally (rather than
    accepting one from the caller) since MediaPipe's VIDEO mode requires
    strictly increasing frame timestamps per landmarker instance -- a fresh
    instance per clip avoids cross-clip timestamp ordering issues when
    rendering several clips in a row."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:   # guards <=0 and NaN
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = out_dir or RENDERED_VIDEOS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_{int(time.time() * 1000)}.mp4"
    # Prefer H.264 (browser-playable, needed for app.py's inline <video> preview) and
    # fall back to MPEG-4 Part 2 if no H.264 encoder is available on this machine --
    # both play fine in a desktop player (predict_foul.py's os.startfile path), but
    # only avc1 is HTML5-<video>-compatible for the upload app.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with build_pose_landmarker(running_mode=vision.RunningMode.VIDEO) as pose_landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, box_in_contact, _contact = detect_players_and_contact(frame_rgb, box_detector)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Draw onto a transparent RGBA layer, then alpha-composite onto
            # the base frame -- PIL doesn't alpha-blend fills drawn directly
            # on an RGBA image, so this two-layer approach is what actually
            # makes the label chips/banner read as translucent instead of a
            # flat opaque block.
            base = Image.fromarray(frame_rgb).convert("RGBA")
            overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
            overlay = draw_pose_skeleton(overlay, pose_result)
            overlay = draw_player_overlay(overlay, boxes, box_in_contact)
            overlay = draw_banner(overlay, banner_lines)
            overlay = draw_watermark(overlay)
            composited = Image.alpha_composite(base, overlay).convert("RGB")

            writer.write(cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR))
            frame_idx += 1

    writer.release()
    cap.release()
    return out_path


def load_clip_frames(video_path: Path, num_frames: int = MAX_FRAMES_PER_ACTION):
    """Decode num_frames evenly-spaced RGB frames from a clip video -- the
    same sparse sample train_foul_rf.py's CNN embeds and
    extract_pose_features.py computes pose features from."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")
    targets = set(np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist())

    frames = []
    i = 0
    while cap.isOpened() and len(frames) < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        if i in targets:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"Could not decode any frames from: {video_path}")
    return frames


def embed_clip(frames, model, transform, device):
    """CNN-embed already-decoded frames (kept separate from load_clip_frames
    so build_feature_vector() decodes the 8 sampled frames once and reuses
    them for both the CNN embedding and the pose features)."""
    tensors = [torch.from_numpy(f).permute(2, 0, 1).contiguous() for f in frames]
    with torch.no_grad():
        vid = torch.stack(tensors, dim=0)               # (T, C, H, W) uint8
        clip = transform(vid).unsqueeze(0).to(device)    # (1, C, T, H, W)
        emb = model(clip).cpu().numpy()
    return emb.reshape(1, -1)   # (1, 512)


def build_feature_vector(video_path: Path, cnn_model, cnn_transform, device, pose_landmarker):
    """Returns (X: (1, 512+NUM_POSE_FEATURES), frames: the 8 decoded RGB
    frames) -- combines the CNN embedding with MediaPipe pose-contact
    features in the exact [CNN | pose] column order train_foul_rf.py
    produces (both sides import the shared math from
    extract_pose_features.py, so the two can't silently disagree)."""
    frames = load_clip_frames(video_path)
    cnn_emb = embed_clip(frames, cnn_model, cnn_transform, device)
    pose_feat = compute_pose_features_from_frames(frames, pose_landmarker)
    X = np.concatenate([cnn_emb, pose_feat.reshape(1, -1)], axis=1)
    return X, frames


def _check_feature_shape(clf, X):
    """The single highest-risk failure mode here is a silently
    shape-mismatched or column-order-mismatched feature vector producing
    meaningless predictions -- fail loudly instead."""
    if hasattr(clf, "n_features_in_") and X.shape[1] != clf.n_features_in_:
        raise RuntimeError(
            f"model.pkl expects {clf.n_features_in_} features but got {X.shape[1]} -- "
            f"did training/inference get out of sync (retrained without updating "
            f"predict_foul.py, or vice versa)?"
        )


def load_model_bundle():
    """Loads model.pkl and validates its feature-schema metadata against the
    CURRENT pose-feature code, failing fast with an actionable message the
    instant the model loads -- instead of a confusing crash deep inside
    feature extraction. This is exactly the failure mode that broke this
    app once already (pose_feature_config.py grew 9->14 features without a
    matching retrain): _check_feature_shape() below still catches that case
    too, but only after CNN+pose extraction already ran on the user's
    upload -- this check fires first, before any video is touched.

    Bundles saved before this check existed won't have "pose_dim" -- that's
    backward compatible (warn, don't crash), so an old model.pkl still
    loads; only a genuine mismatch against a *present* pose_dim is fatal."""
    if not MODEL_PATH.exists():
        print(f"[!] {MODEL_PATH} not found. Run train_foul_rf.py first.")
        sys.exit(1)
    bundle = joblib.load(MODEL_PATH)
    pose_dim = bundle.get("pose_dim")
    if pose_dim is None:
        print(f"[!] {MODEL_PATH} predates feature-schema metadata -- can't pre-validate the "
              f"pose feature count; falling back to the end-of-pipeline shape check.")
    elif pose_dim != NUM_POSE_FEATURES:
        raise RuntimeError(
            f"{MODEL_PATH} was trained with {pose_dim} pose features but the current "
            f"pose_feature_config.py computes {NUM_POSE_FEATURES} -- the model is stale "
            f"relative to the feature-extraction code. Retrain via:\n"
            f"    py -3.14 train_foul_rf.py"
        )
    return bundle


def fuse_and_predict(video_paths, cnn_model, cnn_transform, device, pose_landmarker, bundle, warn=True):
    """video_paths: list of one-or-more Path objects -- ideally every
    available camera-angle clip for the SAME action, since the model
    expects fused multi-view input (mean+max pooled, matching
    train_foul_rf.py). A single path is accepted as a degraded
    approximation: that one view's features are duplicated as both "mean"
    and "max", which is mathematically identical to what real fusion
    produces for a 1-view group -- not the model's true intended input, but
    a graceful, well-defined fallback rather than a hard requirement.

    Single source of truth for the fusion + prediction math -- both
    predict_action() (CLI) and app.py (upload UI) call this, so they can't
    silently disagree on how multi-view features get combined.

    Returns (pred: 0/1, prob: float, label: str, X_fused: the (1, fusion_dim)
    per-action vector BEFORE the action-class stacking columns -- reusable by
    predict_card_and_type() below without re-running feature extraction)."""
    per_view = [build_feature_vector(p, cnn_model, cnn_transform, device, pose_landmarker)[0]
                for p in video_paths]   # each (1, 521)
    view_feats = np.concatenate(per_view, axis=0)   # (n_views, 521)

    if len(video_paths) == 1:
        if warn:
            print("[!] Single-view input -- approximating the model's expected multi-view fusion "
                  "by duplicating this view as both the 'mean' and 'max' pooled vector. This is "
                  "NOT the model's true intended multi-view input; for the real fused prediction, "
                  "pass every available camera-angle clip for this action.")
        X_fused = np.concatenate([view_feats[0], view_feats[0]]).reshape(1, -1)
    else:
        X_fused = np.concatenate([view_feats.mean(axis=0), view_feats.max(axis=0)]).reshape(1, -1)

    # If this model.pkl has the Track-2 auxiliary Action-class classifier,
    # the main classifier was trained on [X_fused | action-class probs] --
    # reproduce that exact transform here, matching train_foul_rf.py. Older
    # bundles (no action_class_clf) fall back to plain X_fused, unaffected.
    action_class_clf = bundle.get("action_class_clf")
    if action_class_clf is not None:
        X_main = np.concatenate([X_fused, action_class_clf.predict_proba(X_fused)], axis=1)
    else:
        X_main = X_fused

    clf, threshold = bundle["clf"], bundle["threshold"]
    _check_feature_shape(clf, X_main)

    prob = float(clf.predict_proba(X_main)[0][1])
    pred = int(prob >= threshold)
    label = "FOUL" if pred == 1 else "NO FOUL"
    return pred, prob, label, X_fused


def predict_card_and_type(X_fused, bundle):
    """Only called when the main prediction is FOUL. Runs the auxiliary
    foul-type classifier and (if trained) the card-outcome classifier on the
    SAME fused vector fuse_and_predict() already computed -- no extra
    feature extraction. Returns (foul_type_label, foul_type_prob,
    card_label, card_prob); any pair is (None, None) if that classifier
    isn't present in this model.pkl (e.g. card_clf skipped at training time
    for lack of labeled-severity Foul actions)."""
    foul_type_label = foul_type_prob = None
    card_label = card_prob = None

    action_class_clf = bundle.get("action_class_clf")
    foul_type_classes = bundle.get("foul_type_classes")
    if action_class_clf is not None and foul_type_classes:
        probs = action_class_clf.predict_proba(X_fused)[0]
        best = int(np.argmax(probs))
        foul_type_label = foul_type_classes[int(action_class_clf.classes_[best])]
        foul_type_prob = float(probs[best])

    card_clf = bundle.get("card_clf")
    card_classes = bundle.get("card_classes")
    if card_clf is not None and card_classes:
        probs = card_clf.predict_proba(X_fused)[0]
        best = int(np.argmax(probs))
        card_label = card_classes[int(card_clf.classes_[best])]
        card_prob = float(probs[best])

    return foul_type_label, foul_type_prob, card_label, card_prob


def predict_action(video_paths, min_score: float):
    bundle = load_model_bundle()
    cnn_model, preprocess, device = build_extractor()
    box_detector = build_player_detector(min_score)

    with build_pose_landmarker() as pose_landmarker:   # IMAGE mode, for feature extraction
        pred, prob, label, X_fused = fuse_and_predict(
            video_paths, cnn_model, preprocess, device, pose_landmarker, bundle)
        foul_type_label = foul_type_prob = card_label = card_prob = None
        if pred:
            foul_type_label, foul_type_prob, card_label, card_prob = predict_card_and_type(X_fused, bundle)

    print(f"Prediction: {label}  (foul probability: {prob:.2%}, threshold: {bundle['threshold']:.2%}, "
          f"fused from {len(video_paths)} view(s))")
    if foul_type_label is not None:
        print(f"  Foul type: {foul_type_label} ({foul_type_prob:.0%})")
    if card_label is not None:
        print(f"  Predicted card: {card_label} ({card_prob:.0%})")

    banner = [(f"Predicted: {label} ({prob:.0%})", FOUL_COLOR if pred else NO_FOUL_COLOR)]
    if foul_type_label is not None:
        banner.append((f"Foul type: {foul_type_label} ({foul_type_prob:.0%})", INFO_COLOR))
    if card_label is not None:
        banner.append((f"Predicted card: {card_label} ({card_prob:.0%})", INFO_COLOR))
    for video_path in video_paths:
        out_path = render_annotated_video(video_path, box_detector, banner)
        print(f"[OK] Rendered annotated video -> {out_path}")
        os.startfile(str(out_path))


def evaluate_random_holdout_slice(fraction: float, show: int, min_score: float):
    if not HOLDOUT_PATH.exists():
        print(f"[!] {HOLDOUT_PATH} not found. Re-run train_foul_rf.py to regenerate it.")
        sys.exit(1)

    bundle = load_model_bundle()
    clf, threshold = bundle["clf"], bundle["threshold"]
    holdout = joblib.load(HOLDOUT_PATH)
    X_test, y_test = holdout["X_test"], holdout["y_test"]
    paths_test = holdout.get("paths_test")

    # No fixed seed here on purpose -- a fresh random slice every run.
    n = len(y_test)
    sample_size = max(1, int(round(n * fraction)))
    idx = np.random.choice(n, size=sample_size, replace=False)
    X_sample, y_sample = X_test[idx], y_test[idx]
    _check_feature_shape(clf, X_sample)

    y_pred = (clf.predict_proba(X_sample)[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_sample, y_pred)
    bal_acc = balanced_accuracy_score(y_sample, y_pred)
    prec = precision_score(y_sample, y_pred, zero_division=0)

    print(f"[->] Evaluating on {sample_size}/{n} randomly sampled unseen actions "
          f"({fraction*100:.0f}% of the held-out set, threshold={threshold:.2f})")
    print(f"\n=== Accuracy: {acc:.4f} ({acc*100:.1f}%) "
          f"| Balanced accuracy: {bal_acc:.4f} ({bal_acc*100:.1f}%) "
          f"| Precision (Foul): {prec:.4f} ({prec*100:.1f}%) ===")
    print(classification_report(y_sample, y_pred, target_names=["No Foul", "Foul"], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_sample, y_pred))

    if show > 0:
        if not paths_test:
            print("\n[!] No view paths stored in holdout_test.pkl -- re-run train_foul_rf.py "
                  "to regenerate it with video previews enabled.")
        else:
            box_detector = build_player_detector(min_score)
            show_n = min(show, sample_size)
            print(f"\n[->] Rendering {show_n} sampled action(s) (all views each)...")
            for i in idx[:show_n]:
                view_paths = paths_test[i]   # list of 1-4 view paths for this action
                prob_i = clf.predict_proba(X_test[i:i + 1])[0][1]
                pred_i = int(prob_i >= threshold)
                true_i = int(y_test[i])
                correct = pred_i == true_i
                pred_label = "FOUL" if pred_i == 1 else "NO FOUL"
                true_label = "FOUL" if true_i == 1 else "NO FOUL"
                banner = [
                    (f"Predicted: {pred_label} ({prob_i:.0%})  [{len(view_paths)} view(s) fused]",
                     FOUL_COLOR if pred_i else NO_FOUL_COLOR),
                    (f"Ground truth: {true_label}  [{'CORRECT' if correct else 'WRONG'}]",
                     CORRECT_COLOR if correct else WRONG_COLOR),
                ]
                print(f"=== action true={true_label}  predicted={pred_label} (prob={prob_i:.2%})  "
                      f"[{'CORRECT' if correct else 'WRONG'}]  {len(view_paths)} view(s) ===")
                for vp in view_paths:
                    out_path = render_annotated_video(Path(vp), box_detector, banner)
                    print(f"    -> {out_path}")
                    os.startfile(str(out_path))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("videos", nargs="*", default=None,
                         help="Path(s) to one or more clip_N.mp4 view files for one-action "
                              "prediction (pass every available camera-angle clip for the "
                              "action -- the model expects fused multi-view input; a single "
                              "path is accepted as a degraded single-view approximation). "
                              "Omit entirely to instead evaluate on a random unseen slice of "
                              "held-out actions.")
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of the held-out set to sample each run (default 0.5).")
    parser.add_argument("--show", type=int, default=5,
                         help="How many of the sampled actions to render+play as annotated "
                              "videos (all views each), with prediction vs. true label "
                              "(default 5; 0 to disable).")
    parser.add_argument("--min-score", type=float, default=0.3,
                         help="MediaPipe player-detection confidence threshold (default 0.3 -- "
                              "frames are native-resolution decoded video now, not 224x224 crops).")
    args = parser.parse_args()

    if not args.videos:
        evaluate_random_holdout_slice(args.fraction, args.show, args.min_score)
    else:
        predict_action([Path(v) for v in args.videos], args.min_score)


if __name__ == "__main__":
    main()
