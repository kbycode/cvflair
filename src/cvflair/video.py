"""
Writing annotated frames to a video file.

``cv2.VideoWriter`` wants the frame size before the first frame and silently
drops anything that does not match it -- the file ends up short or empty with no
error anywhere. This wrapper takes the size from the first frame it is given and
refuses mismatched ones out loud.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

__all__ = ["VideoWriter", "VideoWriteError"]


class VideoWriteError(RuntimeError):
    """The output file could not be opened, or a frame did not fit it."""


class VideoWriter:
    """
    Write frames to a video file.

    The file opens on the first :meth:`write`, so the size does not have to be
    known in advance::

        with VideoWriter("out.mp4", fps=30) as writer:
            for frame, detections in cam.stream(model="yolov8n.pt"):
                writer.write(cam.annotate(frame, detections))

    Codecs come from the OpenCV build, not from here. ``mp4v`` is the one
    present nearly everywhere; if a file will not open, that is the first thing
    to change.
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 25.0,
        codec: str = "mp4v",
        writer_factory: Any = cv2.VideoWriter,
    ) -> None:
        self.path = Path(path)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        self.fps = float(fps)
        self.codec = codec
        self._writer_factory = writer_factory
        self._writer: Any | None = None
        self._size: tuple[int, int] | None = None
        self._frames_written = 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def size(self) -> tuple[int, int] | None:
        """``(width, height)`` once the first frame has set it."""
        return self._size

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def write(self, frame: np.ndarray) -> None:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise VideoWriteError("Boş kare yazılamaz.")

        height, width = frame.shape[:2]
        writer = self._writer
        if writer is None:
            writer = self._open(width, height)
        elif (width, height) != self._size:
            # Karışık boyutlu kareler OpenCV tarafında sessizce düşüyor; dosya
            # eksik çıkıyor ve nedeni hiçbir yerde görünmüyor.
            raise VideoWriteError(
                f"Kare boyutu {width}x{height}, dosya {self._size[0]}x{self._size[1]} "  # type: ignore[index]
                "olarak açıldı. Yazmadan önce yeniden boyutlandır."
            )

        writer.write(frame)
        self._frames_written += 1

    def _open(self, width: int, height: int) -> Any:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # OpenCV 4 bunu modül düzeyinde, 5 sınıfın üstünde veriyor; ikisi de duruyor
        # ama tip bilgisi yalnızca sınıftakini tanıyor.
        writer = self._writer_factory(
            str(self.path), cv2.VideoWriter.fourcc(*self.codec), self.fps, (width, height)
        )
        if not writer.isOpened():
            raise VideoWriteError(
                f"Çıktı dosyası açılamadı: {self.path} (codec {self.codec!r}). "
                "Kurulu OpenCV bu codec'i desteklemiyor olabilir; 'mp4v' ya da "
                "'.avi' uzantısıyla 'MJPG' deneyebilirsin."
            )
        self._writer = writer
        self._size = (width, height)
        return writer

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = f"{self._size[0]}x{self._size[1]}" if self._size else "açılmadı"
        return f"VideoWriter({self.path.name!r}, {state}, {self._frames_written} kare)"
