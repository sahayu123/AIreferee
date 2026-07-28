"""
MediaPipe PoseLandmarker feature extraction for the foul classifier.

train_foul_rf.py runs in the main env (Python 3.14, no MediaPipe wheels
available for it). This script runs separately in .venv-mediapipe (Python
3.12) and precomputes MediaPipe pose-landmark-derived contact/proximity
features for every clip, caching them to pose_features_cache.npz. Unlike the
person-detection boxes predict_foul.py draws for the viewer (purely
cosmetic), these features are real classifier input: train_foul_rf.py
appends them to each clip's CNN embedding before the RandomForest sees it.

Reuses collect_clip_items()/load_clip_tensor() from train_foul_rf.py
directly (importing it only defines functions/constants -- its pipeline is
gated behind `if __name__ == "__main__"`), which guarantees this script pools
clips in the exact same order and samples the exact same 8 frames per clip
that training uses -- no risk of index drift between the two feature
sources.

Usage (from .venv-mediapipe):
    .\\.venv-mediapipe\\Scripts\\python extract_pose_features.py

This is a long, CPU-bound, mostly-unattended run (~8,200 clips x 8 frames =
~65,000 pose-detector calls). Progress is checkpointed periodically to
pose_features_cache.partial.npz so an interruption doesn't lose the whole
pass -- re-running this script picks back up where it left off.
"""

import sys
import time
import urllib.request

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from pathlib import Path

from train_foul_rf import (
    ALL_SPLITS, OUT_DIR, collect_clip_items, load_clip_tensor,
)
from pose_feature_config import POSE_FEATURE_NAMES, NUM_POSE_FEATURES

MEDIAPIPE_MODELS_DIR = OUT_DIR / "mediapipe_models"
POSE_MODEL_PATH = MEDIAPIPE_MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

POSE_FEATURES_CACHE = OUT_DIR / "pose_features_cache.npz"
POSE_FEATURES_PARTIAL = OUT_DIR / "pose_features_cache.partial.npz"

NUM_POSES = 4                    # cap tracked people per frame (main duel + nearby players/ref)
CONTACT_DIST_THRESHOLD = 0.15    # fraction of frame diagonal counted as "close" (2D)
CONTACT_DIST_THRESHOLD_3D = 0.15 # fraction of frame diagonal counted as "close" (3D, depth-aware)
MISSING_DIST_SENTINEL = 1.0      # "far apart / unknown" when <2 people detected in a frame
CHECKPOINT_EVERY = 500
PROGRESS_EVERY = 200

# MediaPipe's 33-landmark BlazePose skeleton starts with 11 face landmarks
# (nose, eyes, ears, mouth -- indices 0-10) before the torso/limb landmarks
# begin at 11 (shoulders) through 32 (feet). Face landmarks are pure noise
# for a player-contact signal -- two players' faces can appear near each
# other in the 2D projection purely from camera angle, unrelated to any
# foul -- so all contact-distance features below are computed only over the
# torso/limb subset.
CONTACT_LANDMARK_START = 11
CONTACT_LANDMARK_IDXS = list(range(CONTACT_LANDMARK_START, 33))

# The well-defined "no evidence" vector for a clip where no pose was ever
# detected (or the clip couldn't be decoded at all) -- distances default to
# the "far apart / unknown" sentinel, not 0, so the RF never confuses "no
# evidence" with "touching."
_DEFAULT_FEATURES = np.array(
    [0.0, 0.0, 0.0,
     MISSING_DIST_SENTINEL, MISSING_DIST_SENTINEL, MISSING_DIST_SENTINEL, 0.0,
     0.0, 0.0,
     MISSING_DIST_SENTINEL, MISSING_DIST_SENTINEL, MISSING_DIST_SENTINEL, 0.0,
     MISSING_DIST_SENTINEL],
    dtype=np.float32,
)
assert len(_DEFAULT_FEATURES) == NUM_POSE_FEATURES


def ensure_pose_model() -> Path:
    """Download MediaPipe's PoseLandmarker-lite bundle on first use (mirrors
    predict_foul.py's ensure_detector_model(), same models/ folder)."""
    MEDIAPIPE_MODELS_DIR.mkdir(exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        print(f"[*] Downloading pose-landmark model -> {POSE_MODEL_PATH}")
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        except Exception as e:
            print(f"[!] Download failed ({e}). Manually download this file and save it to "
                  f"{POSE_MODEL_PATH}:\n    {POSE_MODEL_URL}")
            sys.exit(1)
    return POSE_MODEL_PATH


def build_pose_landmarker(running_mode=vision.RunningMode.IMAGE, num_poses=NUM_POSES):
    """IMAGE mode by default -- used here (and by predict_foul.py's
    build_feature_vector) since the 8 sampled frames per clip are temporally
    sparse with no continuity assumption between them. predict_foul.py's
    separate full-video RENDERING pass uses VIDEO mode instead, since it
    walks every native, temporally continuous frame."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=running_mode,
        num_poses=num_poses,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _landmarks_to_pixel_xyz(landmarks, h: int, w: int) -> np.ndarray:
    """(len(CONTACT_LANDMARK_IDXS), 3) pixel-comparable array from MediaPipe's
    normalized landmarks, restricted to torso/limb landmarks (face excluded).
    x_px = lm.x * w, y_px = lm.y * h, as before -- pixel space (not raw
    normalized coords) is required before computing distances, since
    normalizing x/y independently by width/height isn't distance-preserving
    for non-square frames. z_px = lm.z * w: MediaPipe's docs state z "uses
    roughly the same scale as x" (both normalized relative to image WIDTH,
    hip-midpoint origin, smaller/more-negative = closer to camera), so
    scaling by w (not h) is what makes z_px commensurate with x_px/y_px --
    mirrors MediaPipe's own definition of z's units, not an arbitrary choice."""
    return np.array(
        [[landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z * w] for i in CONTACT_LANDMARK_IDXS],
        dtype=np.float64,
    )


def _frame_diagonal(h: int, w: int) -> float:
    return float((h ** 2 + w ** 2) ** 0.5)


def _closest_landmark_pair_2d(people_xyz):
    """None if <2 people. Else (min_2d_dist_px, person_i, person_j, lm_i,
    lm_j) for the closest landmark pair in the 2D (x,y) pixel projection --
    same metric the original code used, but now also returns which exact
    landmarks achieved it, so the z-gap at that specific pair can be read
    off (the direct "is this apparent 2D closeness actually depth-separated"
    signal)."""
    if len(people_xyz) < 2:
        return None
    best = None
    for i in range(len(people_xyz)):
        for j in range(i + 1, len(people_xyz)):
            diff_xy = people_xyz[i][:, None, :2] - people_xyz[j][None, :, :2]
            d2 = (diff_xy ** 2).sum(axis=-1)
            li, lj = np.unravel_index(int(np.argmin(d2)), d2.shape)
            d = float(np.sqrt(d2[li, lj]))
            if best is None or d < best[0]:
                best = (d, i, j, int(li), int(lj))
    return best


def _closest_landmark_pair_3d(people_xyz):
    """Same as _closest_landmark_pair_2d but using full 3D (x,y,z) Euclidean
    distance -- the truly-nearest landmark pair once depth is accounted for
    (may differ from the 2D-closest pair, e.g. when the 2D-closest pair is
    actually depth-separated). Returns (min_3d_dist_px, person_i, person_j,
    lm_i, lm_j) or None if <2 people."""
    if len(people_xyz) < 2:
        return None
    best = None
    for i in range(len(people_xyz)):
        for j in range(i + 1, len(people_xyz)):
            diff = people_xyz[i][:, None, :] - people_xyz[j][None, :, :]
            d2 = (diff ** 2).sum(axis=-1)
            li, lj = np.unravel_index(int(np.argmin(d2)), d2.shape)
            d = float(np.sqrt(d2[li, lj]))
            if best is None or d < best[0]:
                best = (d, i, j, int(li), int(lj))
    return best


def _aggregate_centroid(people_xy):
    """None if 0 people. Else the mean (x,y) over all landmarks of all
    detected people in the frame -- summarizes "where the pose activity is,"
    used only for the frame-to-frame displacement proxy (deliberately not
    per-person-identity-tracked, and deliberately 2D-only -- this is a
    motion proxy, not a contact signal, so depth doesn't apply here)."""
    if not people_xy:
        return None
    return np.concatenate(people_xy, axis=0).mean(axis=0)


def compute_pose_features_from_frames(frames, landmarker) -> np.ndarray:
    """frames: list of HWC uint8 RGB numpy arrays, in temporal order (8 of
    them for the offline/classification path). Single source of truth for
    the pose-feature math -- both this script and predict_foul.py call this
    exact function, so training and inference can never disagree on the
    feature formula.

    Computes both 2D (pure pixel-projection) and 3D (depth-aware) contact
    distances per frame, plus the z-gap at whichever landmark pair is
    closest in 2D -- the direct signal for "is this apparent 2D closeness
    actually depth-separated" (e.g. one player's arm passing behind
    another's, which should NOT score as contact just because it overlaps
    in the camera's projection)."""
    people_counts = []
    per_frame_min_dist_2d = []
    per_frame_min_dist_3d = []
    per_frame_zgap_at_2d = []   # None for frames with <2 people
    centroids = []   # list of (centroid_xy, frame_diagonal) or None

    for frame in frames:
        h, w = frame.shape[:2]
        diag = _frame_diagonal(h, w)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect(mp_image)
        people_xyz = [_landmarks_to_pixel_xyz(lm_list, h, w) for lm_list in result.pose_landmarks]

        people_counts.append(len(people_xyz))

        pair2d = _closest_landmark_pair_2d(people_xyz)
        if pair2d is None or diag <= 0:
            per_frame_min_dist_2d.append(MISSING_DIST_SENTINEL)
            per_frame_zgap_at_2d.append(None)
        else:
            d2d, pi, pj, li, lj = pair2d
            per_frame_min_dist_2d.append(min(d2d / diag, MISSING_DIST_SENTINEL))
            z_gap = abs(people_xyz[pi][li, 2] - people_xyz[pj][lj, 2])
            per_frame_zgap_at_2d.append(min(z_gap / diag, MISSING_DIST_SENTINEL))

        pair3d = _closest_landmark_pair_3d(people_xyz)
        if pair3d is None or diag <= 0:
            per_frame_min_dist_3d.append(MISSING_DIST_SENTINEL)
        else:
            per_frame_min_dist_3d.append(min(pair3d[0] / diag, MISSING_DIST_SENTINEL))

        centroid = _aggregate_centroid([p[:, :2] for p in people_xyz])
        centroids.append((centroid, diag) if centroid is not None else None)

    people_counts = np.array(people_counts, dtype=np.float64)
    per_frame_min_dist_2d = np.array(per_frame_min_dist_2d, dtype=np.float64)
    per_frame_min_dist_3d = np.array(per_frame_min_dist_3d, dtype=np.float64)
    zgap_frames = [z for z in per_frame_zgap_at_2d if z is not None]

    displacements = []
    for i in range(len(centroids) - 1):
        a, b = centroids[i], centroids[i + 1]
        if a is not None and b is not None:
            (ca, diag_a), (cb, diag_b) = a, b
            diag = (diag_a + diag_b) / 2.0
            if diag > 0:
                displacements.append(float(np.linalg.norm(ca - cb)) / diag)

    return np.array([
        float((people_counts >= 1).sum()),
        float(people_counts.mean()),
        float(people_counts.max()),
        float(per_frame_min_dist_2d.min()),
        float(per_frame_min_dist_2d.mean()),
        float(per_frame_min_dist_2d.max()),
        float((per_frame_min_dist_2d < CONTACT_DIST_THRESHOLD).sum() / len(frames)),
        float(max(displacements)) if displacements else 0.0,
        float(np.mean(displacements)) if displacements else 0.0,
        float(per_frame_min_dist_3d.min()),
        float(per_frame_min_dist_3d.mean()),
        float(per_frame_min_dist_3d.max()),
        float((per_frame_min_dist_3d < CONTACT_DIST_THRESHOLD_3D).sum() / len(frames)),
        float(np.mean(zgap_frames)) if zgap_frames else MISSING_DIST_SENTINEL,
    ], dtype=np.float32)


def compute_pose_features_for_clip(video_path: Path, landmarker) -> np.ndarray:
    """Thin wrapper for the offline/batch path: decodes the same 8
    evenly-spaced frames train_foul_rf.py's CNN embeds, via the identical
    load_clip_tensor() -- guarantees pose features and CNN embeddings are
    always computed from the same frames."""
    vid = load_clip_tensor(video_path)
    if vid is None:
        print(f"[!] Could not decode {video_path} -- using default 'no evidence' pose features")
        return _DEFAULT_FEATURES.copy()
    frames = [f.permute(1, 2, 0).contiguous().numpy() for f in vid]
    return compute_pose_features_from_frames(frames, landmarker)


def main():
    items = []
    for split in ALL_SPLITS:
        items.extend(collect_clip_items(split))
    y_all = np.array([label for label, _ in items])
    paths_all = np.array([str(p) for _, p in items])
    n = len(items)
    print(f"[OK] Pooled {n} clips total for pose-feature extraction")

    P = np.zeros((n, NUM_POSE_FEATURES), dtype=np.float32)
    start_idx = 0
    if POSE_FEATURES_PARTIAL.exists():
        partial = np.load(POSE_FEATURES_PARTIAL)
        n_done = partial["P"].shape[0]
        if n_done <= n and np.array_equal(partial["paths"], paths_all[:n_done]):
            P[:n_done] = partial["P"]
            start_idx = n_done
            print(f"[OK] Resuming from checkpoint: {start_idx}/{n} clips already done")
        else:
            print("[!] Partial checkpoint doesn't match today's pooled clip list -- starting fresh")

    landmarker = build_pose_landmarker()
    t0 = time.time()
    for i in range(start_idx, n):
        _, video_path = items[i]
        P[i] = compute_pose_features_for_clip(video_path, landmarker)

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            done_this_run = (i + 1) - start_idx
            rate = done_this_run / elapsed if elapsed > 0 else 0.0
            remaining_s = (n - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"[->] {i + 1}/{n} clips  ({elapsed:.0f}s elapsed, ~{remaining_s:.0f}s remaining)")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            np.savez(POSE_FEATURES_PARTIAL, P=P[:i + 1], y=y_all[:i + 1], paths=paths_all[:i + 1])

    np.savez(POSE_FEATURES_CACHE, P=P, y=y_all, paths=paths_all)
    if POSE_FEATURES_PARTIAL.exists():
        POSE_FEATURES_PARTIAL.unlink()
    print(f"[OK] Pose features cached -> {POSE_FEATURES_CACHE} {P.shape}")


if __name__ == "__main__":
    main()
