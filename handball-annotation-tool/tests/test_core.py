from pathlib import Path

import cv2
import numpy as np

from handball_annotator.config import AppConfig
from handball_annotator.geometry import closest_arm, point_to_segment
from handball_annotator.gallery import frame_gallery_html
from handball_annotator.miner import create_manual_candidate, create_manual_candidate_from_video
from handball_annotator.negative_sampler import NegativeReviewStore, create_sample
from handball_annotator.similarity import find_handball_duplicates
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
    assert "let current = 0" in html


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


def test_manual_candidate_has_exact_frame_window(tmp_path: Path):
    config = AppConfig("detector", "pose", "tracker", "cpu", .25, 0, 32, .12, .3, .1, 20, 20, 40, 3,
                       tmp_path / "uploads", tmp_path / "candidates", tmp_path / "dataset", tmp_path / "state")
    config.uploads_dir.mkdir(parents=True)
    source = config.uploads_dir / "match.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24))
    for index in range(100):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()
    parent = config.candidates_dir / "detected"
    parent.mkdir(parents=True)
    (parent / "metadata.json").write_text(
        '{"candidate_id":"detected","source_name":"match.mp4","source_path":"' + str(source) + '"}',
        encoding="utf-8",
    )
    created = create_manual_candidate(parent, 50, config)
    assert len(list((created / "clean_frames").glob("*.jpg"))) == 41
    metadata = __import__("json").loads((created / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["center_frame"] == 50 and metadata["manual_selection"] is True


def test_manual_candidate_can_be_created_without_detection(tmp_path: Path):
    config = AppConfig("detector", "pose", "tracker", "cpu", .25, 0, 32, .12, .3, .1, 20, 20, 40, 3,
                       tmp_path / "uploads", tmp_path / "candidates", tmp_path / "dataset", tmp_path / "state")
    config.uploads_dir.mkdir(parents=True)
    source = config.uploads_dir / "no_detection.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24))
    for index in range(80):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()
    created = create_manual_candidate_from_video(source, 40, config)
    metadata = __import__("json").loads((created / "metadata.json").read_text(encoding="utf-8"))
    assert len(list((created / "clean_frames").glob("*.jpg"))) == 41
    assert metadata["source_name"] == source.name
    assert metadata["parent_candidate_id"] is None


def test_duplicate_finder_reports_existing_handball(tmp_path: Path):
    candidate = tmp_path / "candidate"
    candidate_frames = candidate / "clean_frames"
    existing = tmp_path / "dataset" / "handball" / "existing"
    existing_frames = existing / "frames"
    candidate_frames.mkdir(parents=True)
    existing_frames.mkdir(parents=True)
    for index in range(10):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(image, (10 + index * 3, 30), 8, (255, 255, 255), -1)
        cv2.imwrite(str(candidate_frames / f"frame_{index:04d}.jpg"), image)
        cv2.imwrite(str(existing_frames / f"frame_{index:04d}.jpg"), image)
    (candidate / "metadata.json").write_text(
        '{"candidate_id":"new","source_name":"new.mp4","center_frame":100}', encoding="utf-8"
    )
    (existing / "metadata.json").write_text(
        '{"candidate_id":"existing","source_name":"old.mp4","center_frame":200,"center_time_seconds":8.0}',
        encoding="utf-8",
    )
    matches = find_handball_duplicates(candidate, tmp_path / "dataset")
    assert matches and matches[0].candidate_id == "existing" and matches[0].similarity == 1.0


def test_negative_sampler_creates_same_dataset_format(tmp_path: Path):
    config = AppConfig("detector", "pose", "tracker", "cpu", .25, 0, 32, .12, .3, .1, 20, 20, 40, 3,
                       tmp_path / "uploads", tmp_path / "candidates", tmp_path / "dataset", tmp_path / "state")
    config.uploads_dir.mkdir(parents=True)
    source = config.uploads_dir / "full_match.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (32, 24))
    for index in range(80):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()
    candidate = create_sample(source, 40, config, tmp_path / "negative_candidates")
    store = AnnotationStore(config)
    store.label(candidate, "not_handball")
    output = config.dataset_dir / "not_handball" / candidate.name
    assert (output / "clip.mp4").is_file()
    assert (output / "metadata.json").is_file()
    assert len(list((output / "frames").glob("*.jpg"))) == 41
    store.unlabel(candidate.name)
    assert not output.exists() and candidate.is_dir()


def test_negative_review_decisions_can_change(tmp_path: Path):
    source = tmp_path / "match.mp4"
    source.write_bytes(b"video")
    reviews = NegativeReviewStore(tmp_path / "state")
    reviews.set_decision(source, 20, "candidate", "rejected")
    reviews.set_decision(source, 20, "candidate", "accepted")
    assert reviews.decisions(source)[0]["decision"] == "accepted"
