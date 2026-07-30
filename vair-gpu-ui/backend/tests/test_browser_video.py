from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import server as server_module
from backend.server import (
    BrowserVideoError,
    mp4_top_level_atoms,
    publish_browser_video,
    validate_browser_video,
)


def write_mpeg4_fixture(
    path: Path,
    *,
    frames: int = 12,
    fps: int | Fraction = 12,
) -> None:
    frame_rate = Fraction(fps)
    frame_time_base = Fraction(frame_rate.denominator, frame_rate.numerator)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=frame_rate)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = (index * 17) % 255
            image[8:40, 12:52, 1] = 180
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            frame.time_base = frame_time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_publish_browser_video_preserves_frames_and_uses_fast_start(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "annotated-review.mp4"
    write_mpeg4_fixture(source)

    details = publish_browser_video(source, destination)

    assert details["codec"] == "h264"
    assert details["codec_tag"] == "avc1"
    assert details["pixel_format"] == "yuv420p"
    assert details["frames"] == 12
    assert details["frame_rate"] == "12"
    assert details["fast_start"] is True
    assert validate_browser_video(destination, expected_frames=12) == details
    atoms = mp4_top_level_atoms(destination)
    assert atoms[b"moov"] < atoms[b"mdat"]
    assert source.exists()

    with av.open(str(destination), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 12
        assert stream.codec_context.width == 64
        assert stream.codec_context.height == 48
        assert sum(1 for _ in container.decode(stream)) == 12


def test_publish_browser_video_preserves_rational_frame_rate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    destination = tmp_path / "annotated-review.mp4"
    frame_rate = Fraction(30000, 1001)
    write_mpeg4_fixture(source, frames=17, fps=frame_rate)

    publish_browser_video(source, destination)

    with av.open(str(destination), mode="r") as container:
        stream = container.streams.video[0]
        assert stream.average_rate == frame_rate
        assert stream.duration * stream.time_base == Fraction(17, 1) / frame_rate
        assert sum(1 for _ in container.decode(stream)) == 17


def test_publish_browser_video_fast_path_preserves_valid_h264_bytes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    first_destination = tmp_path / "first.mp4"
    second_destination = tmp_path / "second.mp4"
    write_mpeg4_fixture(source)
    publish_browser_video(source, first_destination)

    details = publish_browser_video(first_destination, second_destination)

    assert details["codec"] == "h264"
    assert second_destination.read_bytes() == first_destination.read_bytes()


def test_validate_browser_video_rejects_mpeg4_part_two(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    write_mpeg4_fixture(source)

    with pytest.raises(BrowserVideoError, match="not browser-safe H.264"):
        validate_browser_video(source)


def test_publish_browser_video_does_not_publish_invalid_input(
    tmp_path: Path,
) -> None:
    source = tmp_path / "invalid.mp4"
    destination = tmp_path / "annotated-review.mp4"
    source.write_bytes(b"not a video")
    destination.write_bytes(b"previous valid publication")

    with pytest.raises(BrowserVideoError):
        publish_browser_video(source, destination)

    assert destination.read_bytes() == b"previous valid publication"
    assert list(tmp_path.glob(".annotated-review-*.mp4")) == []


def test_artifact_route_streams_browser_video_with_byte_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    artifact_root = tmp_path / "artifacts"
    destination = artifact_root / "test-job" / "annotated-review.mp4"
    write_mpeg4_fixture(source)
    publish_browser_video(source, destination)
    monkeypatch.setattr(server_module, "ARTIFACT_ROOT", artifact_root)

    response = TestClient(server_module.app).get(
        "/api/artifacts/test-job/annotated-review.mp4",
        headers={"Range": "bytes=0-31"},
    )

    assert response.status_code == 206
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-range"].startswith("bytes 0-31/")
    assert len(response.content) == 32
