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

2. Predict a single clip:
   python predict_foul.py "C:\\path\\to\\clip_0.mp4"
   Point it at one raw SN-MVFouls-2025 clip video file.

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
    contact: bool)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    boxes = []
    for detection in result.detections:
        box = detection.bounding_box
        x0, y0 = box.origin_x, box.origin_y
        x1, y1 = x0 + box.width, y0 + box.height
        score = detection.categories[0].score
        boxes.append((x0, y0, x1, y1, score))
    contact = any(boxes_overlap(boxes[i][:4], boxes[j][:4])
                  for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
    return boxes, contact


def draw_player_overlay(img: Image.Image, boxes, contact: bool) -> Image.Image:
    """Draw a box + confidence score around every detected player, and flag
    "possible contact" when two player boxes significantly overlap."""
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1, score in boxes:
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        draw.text((x0, max(0, y0 - 12)), f"person {score:.2f}", fill=(255, 0, 0))
    if contact:
        draw.text((5, 5), "possible contact", fill=(255, 255, 0))
    return img


def draw_pose_skeleton(img: Image.Image, pose_result) -> Image.Image:
    """Draw MediaPipe's pose landmarks + skeleton connections for every
    detected person, in cyan -- a distinct visual layer from the red player
    boxes, so you can see exactly what the pose-contact features (used by
    the classifier) are derived from."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for landmarks in pose_result.pose_landmarks:
        pts = [(lm.x * w, lm.y * h) for lm in landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                draw.line([pts[a], pts[b]], fill=(0, 255, 255), width=2)
        for x, y in pts:
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 255))
    return img


_banner_font = None


def _get_banner_font():
    global _banner_font
    if _banner_font is None:
        try:
            _banner_font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            _banner_font = ImageFont.load_default()
    return _banner_font


def draw_banner(img: Image.Image, banner_lines) -> Image.Image:
    """Persistent bottom banner burned into every frame. banner_lines is a
    list of (text, (r,g,b)) tuples, one per line -- callers decide content
    (this function is display-only, mode-agnostic)."""
    draw = ImageDraw.Draw(img)
    font = _get_banner_font()
    w, h = img.size
    line_height = 26
    padding = 6
    bar_height = line_height * len(banner_lines) + 2 * padding
    draw.rectangle([0, h - bar_height, w, h], fill=(0, 0, 0))
    for i, (text, color) in enumerate(banner_lines):
        y = h - bar_height + padding + i * line_height
        draw.text((padding, y), text, fill=color, font=font)
    return img


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
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with build_pose_landmarker(running_mode=vision.RunningMode.VIDEO) as pose_landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, contact = detect_players_and_contact(frame_rgb, box_detector)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            img = Image.fromarray(frame_rgb)
            img = draw_player_overlay(img, boxes, contact)
            img = draw_pose_skeleton(img, pose_result)
            img = draw_banner(img, banner_lines)

            writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
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


def predict_single_clip(video_path: Path, min_score: float):
    if not MODEL_PATH.exists():
        print(f"[!] {MODEL_PATH} not found. Run train_foul_rf.py first.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    clf, threshold = bundle["clf"], bundle["threshold"]
    cnn_model, preprocess, device = build_extractor()
    box_detector = build_player_detector(min_score)

    with build_pose_landmarker() as pose_landmarker:   # IMAGE mode, for feature extraction
        X, _ = build_feature_vector(video_path, cnn_model, preprocess, device, pose_landmarker)
    _check_feature_shape(clf, X)

    prob = clf.predict_proba(X)[0][1]
    pred = int(prob >= threshold)
    label = "FOUL" if pred == 1 else "NO FOUL"
    print(f"Prediction: {label}  (foul probability: {prob:.2%}, threshold: {threshold:.2%})")

    banner = [(f"Predicted: {label} ({prob:.0%})", (255, 255, 0))]
    out_path = render_annotated_video(video_path, box_detector, banner)
    print(f"[OK] Rendered annotated video -> {out_path}")
    os.startfile(str(out_path))


def evaluate_random_holdout_slice(fraction: float, show: int, min_score: float):
    if not MODEL_PATH.exists():
        print(f"[!] {MODEL_PATH} not found. Run train_foul_rf.py first.")
        sys.exit(1)
    if not HOLDOUT_PATH.exists():
        print(f"[!] {HOLDOUT_PATH} not found. Re-run train_foul_rf.py to regenerate it.")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
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

    print(f"[->] Evaluating on {sample_size}/{n} randomly sampled unseen clips "
          f"({fraction*100:.0f}% of the held-out set, threshold={threshold:.2f})")
    print(f"\n=== Accuracy: {acc:.4f} ({acc*100:.1f}%) "
          f"| Balanced accuracy: {bal_acc:.4f} ({bal_acc*100:.1f}%) "
          f"| Precision (Foul): {prec:.4f} ({prec*100:.1f}%) ===")
    print(classification_report(y_sample, y_pred, target_names=["No Foul", "Foul"], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_sample, y_pred))

    if show > 0:
        if not paths_test:
            print("\n[!] No frame paths stored in holdout_test.pkl -- re-run train_foul_rf.py "
                  "to regenerate it with video previews enabled.")
        else:
            box_detector = build_player_detector(min_score)
            show_n = min(show, sample_size)
            print(f"\n[->] Rendering {show_n} annotated video(s) of the sampled clips...")
            for i in idx[:show_n]:
                video_path = Path(paths_test[i])
                prob_i = clf.predict_proba(X_test[i:i + 1])[0][1]
                pred_i = int(prob_i >= threshold)
                true_i = int(y_test[i])
                correct = pred_i == true_i
                pred_label = "FOUL" if pred_i == 1 else "NO FOUL"
                true_label = "FOUL" if true_i == 1 else "NO FOUL"
                banner = [
                    (f"Predicted: {pred_label} ({prob_i:.0%})", (255, 255, 0)),
                    (f"Ground truth: {true_label}  [{'CORRECT' if correct else 'WRONG'}]",
                     (0, 200, 0) if correct else (220, 50, 50)),
                ]
                out_path = render_annotated_video(video_path, box_detector, banner)
                print(f"=== true={true_label}  predicted={pred_label} (prob={prob_i:.2%})  "
                      f"[{'CORRECT' if correct else 'WRONG'}] -> {out_path} ===")
                os.startfile(str(out_path))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("video", nargs="?", default=None,
                         help="Path to a single clip_N.mp4 video file for one-clip prediction. "
                              "Omit to instead evaluate on a random unseen slice.")
    parser.add_argument("--fraction", type=float, default=0.5,
                         help="Fraction of the held-out set to sample each run (default 0.5).")
    parser.add_argument("--show", type=int, default=5,
                         help="How many of the sampled clips to render+play as annotated "
                              "videos, with prediction vs. true label (default 5; 0 to disable).")
    parser.add_argument("--min-score", type=float, default=0.3,
                         help="MediaPipe player-detection confidence threshold (default 0.3 -- "
                              "frames are native-resolution decoded video now, not 224x224 crops).")
    args = parser.parse_args()

    if args.video is None:
        evaluate_random_holdout_slice(args.fraction, args.show, args.min_score)
    else:
        predict_single_clip(Path(args.video), args.min_score)


if __name__ == "__main__":
    main()
