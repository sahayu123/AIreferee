from pathlib import Path

import cv2
import numpy as np

from handball_annotator.config import AppConfig
from handball_annotator.geometry import closest_arm, point_to_segment
from handball_annotator.gallery import frame_gallery_html
from handball_annotator.storage import AnnotationStore


def test_point_to_segment_distance():
    distance = point_to_segment(np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([2.0, 0.0]))
    assert distance == 2.0


def test_closest_arm():
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[5], keypoints[7], keypoints[9] = (10, 10, 1), (10, 30, 1), (10, 50, 1)
    match = closest_arm(np.array([12.0, 40.0]), keypoints, player_height=100)
    assert match is not None
    assert match.side == "left" and match.segment == "forearm"
    assert match.normalized_distance == 0.02

    edge_match = closest_arm(np.array([14.0, 40.0]), keypoints, player_height=100, ball_radius=3.0)
    assert edge_match is not None
    assert edge_match.distance == 1.0


def test_frame_gallery_has_fullscreen_navigation(tmp_path: Path):
    frame = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame), np.zeros((4, 4, 3), dtype=np.uint8))
    html = frame_gallery_html([frame], 0)
    assert "requestFullscreen" in html
    assert "ArrowLeft" in html and "ArrowRight" in html


def test_label_store_copies_only_clean_artifacts(tmp_path: Path):
    config = AppConfig("detector", "pose", "tracker", "cpu", .25, 0, 32, .12, .3, .1, 20, 20, 40, 3,
                       tmp_path / "uploads", tmp_path / "candidates", tmp_path / "dataset", tmp_path / "state")
    store = AnnotationStore(config)
    candidate = config.candidates_dir / "event_1"
    (candidate / "clean_frames").mkdir(parents=True)
    (candidate / "evidence_frames").mkdir()
    (candidate / "clean.mp4").write_bytes(b"clean")
    cv2.imwrite(str(candidate / "clean_frames" / "frame.jpg"), np.zeros((4, 4, 3), dtype=np.uint8))
    cv2.imwrite(str(candidate / "evidence_frames" / "frame.jpg"), np.ones((4, 4, 3), dtype=np.uint8))
    (candidate / "evidence.mp4").write_bytes(b"overlay")
    (candidate / "metadata.json").write_text('{"candidate_id":"event_1","source_name":"match.mp4"}', encoding="utf-8")

    store.label(candidate, "handball")
    output = config.dataset_dir / "handball" / "event_1"
    assert (output / "clip.mp4").read_bytes() == b"clean"
    assert not (output / "evidence.mp4").exists()
    assert not (output / "evidence_frames").exists()

    # Repeating a label is idempotent, while changing it atomically moves the example.
    store.label(candidate, "handball")
    assert (output / "frames" / "frame.jpg").is_file()
    store.label(candidate, "uncertain")
    assert not output.exists()
    assert (config.dataset_dir / "uncertain" / "event_1" / "frames" / "frame.jpg").is_file()
