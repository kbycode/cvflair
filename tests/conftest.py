"""Shared fixtures. Nothing here touches a real camera or a GUI window."""

from __future__ import annotations

import time

import numpy as np
import pytest

from cvflair import Camera, Detections

FRAME_SHAPE = (120, 160, 3)


def make_frame(marker: int) -> np.ndarray:
    """A solid frame whose pixel value identifies which frame it is."""
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    frame[:] = marker % 256
    return frame


def frame_marker(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0])


class FakeCapture:
    """Stand-in for ``cv2.VideoCapture`` with a finite supply of frames."""

    def __init__(self, frame_count: int = 5, *, opened: bool = True, delay: float = 0.0) -> None:
        self.frame_count = frame_count
        self.delay = delay
        self.properties: dict[int, float] = {}
        self.released = False
        self.served = 0
        self._opened = opened

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's spelling
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.served >= self.frame_count:
            return False, None
        if self.delay:
            time.sleep(self.delay)
        self.served += 1
        return True, make_frame(self.served)

    def set(self, prop: int, value: float) -> bool:
        self.properties[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.properties.get(prop, 0.0)

    def release(self) -> None:
        self.released = True


@pytest.fixture
def camera_factory():
    """Build cameras backed by a :class:`FakeCapture`, closed after the test."""
    cameras: list[Camera] = []

    def build(
        frame_count: int = 5,
        *,
        opened: bool = True,
        delay: float = 0.0,
        **kwargs,
    ) -> tuple[Camera, FakeCapture]:
        capture = FakeCapture(frame_count, opened=opened, delay=delay)
        camera = Camera(source="fake", capture_factory=lambda source: capture, **kwargs)
        cameras.append(camera)
        return camera, capture

    yield build

    for camera in cameras:
        camera.close()


@pytest.fixture
def detections() -> Detections:
    return Detections(
        xyxy=np.array([[20, 20, 90, 80], [100, 30, 150, 100]], dtype=np.float32),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.7], dtype=np.float32),
        names=np.array(["kisi", "kopek"], dtype=object),
    )
