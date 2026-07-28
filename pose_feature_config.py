"""Shared pose-feature schema -- pure Python, zero heavy dependencies (no
mediapipe/numpy/torch), so it's safely importable from BOTH the main env
(Python 3.14, train_foul_rf.py) and .venv-mediapipe (Python 3.12,
extract_pose_features.py) without creating an import cycle or requiring
mediapipe in the main env.

Bump this list whenever the feature *formula* in
extract_pose_features.compute_pose_features_from_frames changes, even if the
count doesn't -- train_foul_rf.py checks the cached pose_features_cache.npz's
column count against NUM_POSE_FEATURES and raises loudly on mismatch, so a
stale cache can never be silently trained on.
"""

POSE_FEATURE_NAMES = [
    "n_frames_with_pose",          # count of 8 sampled frames with >=1 person detected
    "mean_people_per_frame",       # mean people-count across the 8 frames
    "max_people_in_any_frame",     # max people-count across the 8 frames
    "min_pair_dist_min",           # closest inter-person 2D approach anywhere in the clip
    "min_pair_dist_mean",          # mean of per-frame closest-pair 2D distance
    "min_pair_dist_max",           # least-close frame's closest-pair 2D distance
    "frac_frames_close_contact",   # fraction of frames with 2D min_pair_dist < CONTACT_DIST_THRESHOLD
    "centroid_displacement_max",   # max frame-to-frame aggregate pose-centroid jump (7 gaps)
    "centroid_displacement_mean",  # mean of the same
    "min_pair_dist_3d_min",        # closest inter-person 3D (depth-aware) approach anywhere in the clip
    "min_pair_dist_3d_mean",       # mean of per-frame closest-pair 3D distance
    "min_pair_dist_3d_max",        # least-close frame's closest-pair 3D distance
    "frac_frames_close_contact_3d",# fraction of frames with 3D min_pair_dist < CONTACT_DIST_THRESHOLD_3D
    "z_gap_at_min2d_mean",         # mean depth separation at the landmark pair closest in 2D --
                                    # large values mean apparent 2D closeness is a depth illusion
                                    # (e.g. one arm passing behind another), not real contact
]
NUM_POSE_FEATURES = len(POSE_FEATURE_NAMES)
