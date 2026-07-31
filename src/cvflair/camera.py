"""
Video capture on a background thread, with a one-slot latest-frame queue.

For a live camera the reader never blocks on the consumer: before publishing a
new frame it drops the pending one, so ``read()`` always returns the most recent
frame and latency does not build up when annotation is slower than the camera.
Video files want the opposite -- see ``drop_frames``.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, overload

import cv2
import numpy as np

from .detections import Detections
from .models import ModelLike, resolve_detector
from .themes import Theme, get_theme

__all__ = ["Camera", "CameraError"]

QUIT_KEYS = (ord("q"), ord("Q"), 27)  # 27 == ESC


class CameraError(RuntimeError):
    """The capture device could not be opened, or died while streaming."""


class Camera:
    """
    A threaded video source with a theme attached.

    ::

        cam = Camera(source=0, theme="neon")
        for frame in cam.stream():
            cam.show(frame, detections)

    ``stream()`` starts the reader thread on first use and releases the device
    when the generator ends, so the loop above needs no explicit teardown.
    Pressing ``q`` or ``ESC`` in the preview window ends the stream.

    Any object with OpenCV's ``VideoCapture`` interface can be injected through
    ``capture_factory``; that is how the test suite runs without a camera.
    """

    def __init__(
        self,
        source: int | str = 0,
        theme: str | Theme = "minimal",
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        window_name: str = "cvflair",
        drop_frames: bool = True,
        capture_factory: Callable[[Any], Any] = cv2.VideoCapture,
    ) -> None:
        self.source = source
        #: Dropping the stale frame is right for a live camera. A video file has
        #: nothing to be late for and every frame counts, so set this to False
        #: and the reader waits for the consumer instead.
        self.drop_frames = drop_frames
        self.window_name = window_name
        self.width = width
        self.height = height
        self.fps = fps

        self._theme = get_theme(theme)
        self._capture_factory = capture_factory
        self._capture: Any | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._window_open = False
        self._last_key = -1
        self._frames_read = 0
        self._frames_dropped = 0
        # Tüketilen karelerin zaman damgaları; ölçülen hız buradan çıkıyor.
        self._read_times: deque[float] = deque(maxlen=30)

    # -- configuration ---------------------------------------------------

    @property
    def theme(self) -> Theme:
        return self._theme

    @theme.setter
    def theme(self, theme: str | Theme) -> None:
        self._theme = get_theme(theme)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def frames_read(self) -> int:
        """Frames pulled off the device since :meth:`start`."""
        return self._frames_read

    @property
    def frames_dropped(self) -> int:
        """Frames discarded because the consumer was still busy."""
        return self._frames_dropped

    @property
    def measured_fps(self) -> float:
        """
        Frames actually consumed per second, averaged over the last 30.

        This is the rate the loop achieves, not the rate requested from the
        device through ``fps``; annotation and inference slow it down.
        """
        if len(self._read_times) < 2:
            return 0.0
        span = self._read_times[-1] - self._read_times[0]
        return (len(self._read_times) - 1) / span if span > 0 else 0.0

    # -- lifecycle -------------------------------------------------------

    def start(self) -> Camera:
        """Open the device and start the reader thread. Safe to call twice."""
        if self.is_running:
            return self

        capture = self._capture_factory(self.source)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open video source {self.source!r}. "
                "Check that the device index is right and no other application is using it."
            )
        self._apply_properties(capture)

        self._capture = capture
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name=f"cvflair-reader-{self.source}", daemon=True
        )
        self._thread.start()
        return self

    def _apply_properties(self, capture: Any) -> None:
        if self.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps is not None:
            capture.set(cv2.CAP_PROP_FPS, self.fps)

    def _read_loop(self) -> None:
        capture = self._capture
        if capture is None:  # pragma: no cover - start() always sets it first
            return
        while not self._stop_event.is_set():
            ok, frame = capture.read()
            if not ok or frame is None:
                break  # end of file, or the device went away
            self._frames_read += 1
            self._publish(frame)
        self._stop_event.set()

    def _publish(self, frame: np.ndarray) -> None:
        """
        Hand ``frame`` to the consumer.

        With ``drop_frames`` the pending frame is thrown away so the queue always
        holds the newest one; without it the reader waits, which is what a video
        file needs -- there every frame matters and there is nothing to be late for.
        """
        if not self.drop_frames:
            while not self._stop_event.is_set():
                try:
                    self._queue.put(frame, timeout=0.1)
                    return
                except queue.Full:
                    continue
            return

        try:
            self._queue.get_nowait()
            self._frames_dropped += 1
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(frame)
        except queue.Full:  # pragma: no cover - single producer, cannot happen
            self._frames_dropped += 1

    def close(self) -> None:
        """Stop the thread, release the device, close the preview window."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._window_open:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:  # pragma: no cover - headless build, or already gone
                pass
            self._window_open = False

    def __enter__(self) -> Camera:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "running" if self.is_running else "stopped"
        return f"Camera(source={self.source!r}, theme={self._theme.name!r}, {state})"

    # -- reading ---------------------------------------------------------

    def read(self, timeout: float = 5.0) -> np.ndarray | None:
        """
        Return the latest frame, or ``None`` if the source ended or stalled.

        A frame is never returned twice: the queue slot is consumed here.
        """
        try:
            frame = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        self._read_times.append(time.perf_counter())
        return frame

    @overload
    def stream(self, timeout: float = ..., *, model: None = ...) -> Iterator[np.ndarray]: ...

    @overload
    def stream(
        self, timeout: float = ..., *, model: ModelLike
    ) -> Iterator[tuple[np.ndarray, Detections]]: ...

    def stream(self, timeout: float = 5.0, *, model: ModelLike | None = None):
        """
        Yield frames until the source ends, ``q``/``ESC`` is pressed, or no
        frame arrives within ``timeout`` seconds. Releases the device on exit.

        Without ``model`` the frames come through bare. With one -- a weights
        path, an Ultralytics model, or any callable returning detections
        -- each item is a ``(frame, detections)`` pair::

            for frame, detections in cam.stream(model="yolov8n.pt"):
                cam.show(frame, detections)

        Inference runs in this loop, not in the reader thread: while a frame is
        being processed the reader keeps replacing the queued frame, so the next
        iteration gets the newest one rather than a backlog.
        """
        detector = resolve_detector(model)
        self.start()
        try:
            while not self._stop_event.is_set():
                frame = self.read(timeout=timeout)
                if frame is None:
                    break
                yield frame if detector is None else (frame, detector(frame))
        finally:
            self.close()

    # -- output ----------------------------------------------------------

    def annotate(
        self,
        frame: np.ndarray,
        detections: Detections | None = None,
        labels: Sequence[str] | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Apply the active theme in place. Useful when not using :meth:`show`.

        Themes with a HUD get frame rate and detection count for free; ``stats``
        adds to that panel and wins on a repeated key.
        """
        return self._theme.annotate(
            frame,
            detections if detections is not None else Detections.empty(),
            labels=labels,
            stats=self._hud_stats(detections, stats),
        )

    def _hud_stats(
        self, detections: Detections | None, extra: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        if self._theme._hud_annotator is None:
            return None
        stats: dict[str, Any] = {
            "FPS": f"{self.measured_fps:.0f}",
            "Objects": 0 if detections is None else len(detections),
        }
        if extra:
            stats.update(extra)
        return stats

    def show(
        self,
        frame: np.ndarray,
        detections: Detections | None = None,
        labels: Sequence[str] | None = None,
        *,
        stats: Mapping[str, Any] | None = None,
        wait: int = 1,
    ) -> bool:
        """
        Annotate and display ``frame``.

        Returns ``False`` once the user asks to quit (``q``, ``ESC``, or the
        window's close button), which also ends an active :meth:`stream`.

        Whatever key was pressed stays available through :attr:`key` and
        :meth:`pressed`, so a demo can react to it::

            cam.show(frame, detections)
            if cam.pressed("1"):
                cam.theme = "neon"
        """
        self.annotate(frame, detections, labels=labels, stats=stats)
        cv2.imshow(self.window_name, frame)
        self._window_open = True

        pressed = cv2.waitKey(wait) & 0xFF
        # waitKey returns -1 when nothing was pressed, which masks to 255.
        self._last_key = -1 if pressed == 255 else pressed

        if self._last_key in QUIT_KEYS or self._window_closed():
            self._stop_event.set()
            return False
        return True

    @property
    def key(self) -> int:
        """Key code from the last :meth:`show`, or ``-1`` when nothing was pressed."""
        return self._last_key

    def pressed(self, key: str | int) -> bool:
        """True when the last :meth:`show` saw ``key`` -- ``cam.pressed("1")``."""
        code = ord(key) if isinstance(key, str) else int(key)
        return self._last_key == code

    def _window_closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:  # pragma: no cover - headless build without HighGUI
            return False
