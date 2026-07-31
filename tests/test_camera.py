"""Camera tests. A fake capture stands in for the webcam, so these run headless."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import FRAME_SHAPE, frame_marker, make_frame

from cvflair import CameraError, Theme


def test_start_reads_a_frame(camera_factory):
    camera, _ = camera_factory(frame_count=3)
    camera.start()

    frame = camera.read(timeout=2.0)

    assert frame is not None
    assert frame.shape == FRAME_SHAPE
    assert frame_marker(frame) >= 1


def test_start_is_idempotent(camera_factory):
    camera, _ = camera_factory(frame_count=10, delay=0.01)
    camera.start()
    thread = camera._thread

    camera.start()

    assert camera._thread is thread


def test_unopened_source_raises_and_releases(camera_factory):
    camera, capture = camera_factory(opened=False)

    with pytest.raises(CameraError, match="Could not open video source"):
        camera.start()

    assert capture.released
    assert not camera.is_running


def test_requested_properties_reach_the_device(camera_factory):
    camera, capture = camera_factory(frame_count=1, width=640, height=480, fps=30)
    camera.start()

    assert capture.properties[cv2.CAP_PROP_FRAME_WIDTH] == 640
    assert capture.properties[cv2.CAP_PROP_FRAME_HEIGHT] == 480
    assert capture.properties[cv2.CAP_PROP_FPS] == 30


def test_stream_yields_frames_in_order_then_releases(camera_factory):
    camera, capture = camera_factory(frame_count=8, delay=0.01)

    markers = [frame_marker(frame) for frame in camera.stream(timeout=2.0)]

    assert markers, "stream produced nothing"
    assert markers == sorted(set(markers)), "frames arrived out of order or repeated"
    assert len(markers) <= 8
    assert capture.released
    assert not camera.is_running


def test_queue_keeps_only_the_latest_frame(camera_factory):
    # Driven directly rather than through the thread: the drop policy is what
    # is under test here, and racing the reader would make it non-deterministic.
    camera, _ = camera_factory(frame_count=0)

    camera._publish(make_frame(1))
    camera._publish(make_frame(2))
    camera._publish(make_frame(3))

    assert camera.frames_dropped == 2
    assert frame_marker(camera.read(timeout=0)) == 3
    assert camera.read(timeout=0) is None


def test_stream_with_a_model_yields_pairs(camera_factory, detections):
    camera, _ = camera_factory(frame_count=4, delay=0.01)
    seen = []

    for frame, found in camera.stream(timeout=2.0, model=lambda f: detections):
        seen.append((frame, found))

    assert seen, "stream produced nothing"
    assert all(found is detections for _, found in seen)
    assert all(frame.shape == FRAME_SHAPE for frame, _ in seen)


def test_stream_without_a_model_yields_bare_frames(camera_factory):
    camera, _ = camera_factory(frame_count=3, delay=0.01)

    for item in camera.stream(timeout=2.0):
        assert isinstance(item, np.ndarray)


def test_stream_reports_a_broken_model(camera_factory):
    camera, _ = camera_factory(frame_count=3, delay=0.01)

    with pytest.raises(TypeError, match="expected detections"):
        for _ in camera.stream(timeout=2.0, model=lambda frame: None):
            pass

    assert not camera.is_running, "the device must be released when the loop raises"


def test_exhausted_source_ends_the_stream(camera_factory):
    camera, _ = camera_factory(frame_count=0)

    assert list(camera.stream(timeout=0.5)) == []
    assert camera.read(timeout=0) is None


def test_context_manager_starts_and_releases(camera_factory):
    camera, capture = camera_factory(frame_count=5, delay=0.01)

    with camera as opened:
        assert opened is camera
        assert camera.is_running

    assert capture.released
    assert not camera.is_running


def test_theme_can_be_set_by_name(camera_factory):
    camera, _ = camera_factory(frame_count=1)

    assert camera.theme.name == "minimal"
    camera.theme = "neon"
    assert camera.theme.name == "neon"
    assert isinstance(camera.theme, Theme)

    with pytest.raises(ValueError, match="Unknown theme"):
        camera.theme = "nope"


def test_annotate_applies_the_theme_in_place(camera_factory, detections):
    camera, _ = camera_factory(frame_count=1, theme="neon")
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    annotated = camera.annotate(frame, detections)

    assert annotated is frame
    assert frame.any(), "nothing was drawn"


def test_annotate_without_detections_is_a_no_op(camera_factory):
    camera, _ = camera_factory(frame_count=1)
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    assert camera.annotate(frame, None) is frame
    assert not frame.any()


def test_repr_reports_source_and_theme(camera_factory):
    camera, _ = camera_factory(frame_count=1, theme="neon")

    assert repr(camera) == "Camera(source='fake', theme='neon', stopped)"


def test_dropping_is_on_by_default(camera_factory):
    camera, _ = camera_factory(frame_count=1)

    assert camera.drop_frames is True


def test_backpressure_keeps_every_frame(camera_factory):
    """drop_frames=False: dosya kaynağında hiçbir kare atlanmamalı."""
    camera, _ = camera_factory(frame_count=25, delay=0.001, drop_frames=False)

    markers = [frame_marker(frame) for frame in camera.stream(timeout=2.0)]

    assert markers == list(range(1, 26))
    assert camera.frames_dropped == 0
