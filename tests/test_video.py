"""Video dosyasına yazma."""

from __future__ import annotations

import numpy as np
import pytest

from cvflair import VideoWriteError, VideoWriter


class FakeWriter:
    """``cv2.VideoWriter`` yerine geçer; kodek ve disk gerektirmez."""

    def __init__(self, path, fourcc, fps, size, *, opened: bool = True) -> None:
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.size = size
        self.frames: list[np.ndarray] = []
        self.released = False
        self._opened = opened

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV yazımı
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


def frame(width: int = 64, height: int = 48, value: int = 7) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def writer_for(tmp_path, **kwargs) -> tuple[VideoWriter, list[FakeWriter]]:
    made: list[FakeWriter] = []

    def factory(*args):
        made.append(FakeWriter(*args, **kwargs))
        return made[-1]

    return VideoWriter(tmp_path / "cikti.mp4", writer_factory=factory), made


def test_file_opens_on_the_first_frame(tmp_path):
    video, made = writer_for(tmp_path)

    assert not video.is_open and video.size is None
    video.write(frame())

    assert video.is_open
    assert video.size == (64, 48)
    assert made[0].size == (64, 48), "boyut ilk kareden alınmalı"


def test_frames_are_counted(tmp_path):
    video, made = writer_for(tmp_path)

    for _ in range(5):
        video.write(frame())

    assert video.frames_written == 5
    assert len(made[0].frames) == 5


def test_size_change_is_refused(tmp_path):
    video, _ = writer_for(tmp_path)
    video.write(frame(64, 48))

    with pytest.raises(VideoWriteError, match="80x60"):
        video.write(frame(80, 60))


def test_empty_frame_is_refused(tmp_path):
    video, _ = writer_for(tmp_path)

    with pytest.raises(VideoWriteError, match="Boş kare"):
        video.write(np.empty((0, 0, 3), dtype=np.uint8))


def test_unopenable_file_says_so(tmp_path):
    video, _ = writer_for(tmp_path, opened=False)

    with pytest.raises(VideoWriteError, match="açılamadı"):
        video.write(frame())


@pytest.mark.parametrize("fps", [0, -1])
def test_bad_fps_is_refused(tmp_path, fps):
    with pytest.raises(ValueError, match="fps must be positive"):
        VideoWriter(tmp_path / "x.mp4", fps=fps)


def test_context_manager_releases(tmp_path):
    video, made = writer_for(tmp_path)

    with video:
        video.write(frame())

    assert made[0].released
    assert not video.is_open


def test_closing_twice_is_harmless(tmp_path):
    video, _ = writer_for(tmp_path)
    video.write(frame())

    video.close()
    video.close()


def test_missing_folder_is_created(tmp_path):
    target = tmp_path / "yeni" / "alt" / "cikti.mp4"
    video = VideoWriter(target, writer_factory=lambda *args: FakeWriter(*args))

    video.write(frame())

    assert target.parent.is_dir()


def test_repr_shows_progress(tmp_path):
    video, _ = writer_for(tmp_path)

    assert "açılmadı" in repr(video)
    video.write(frame())
    assert "64x48" in repr(video) and "1 kare" in repr(video)


def test_real_file_is_written_and_readable(tmp_path):
    """Sahte yazıcı OpenCV'nin kendi davranışını doğrulamıyor; bu onu doğruluyor."""
    cv2 = pytest.importorskip("cv2")
    target = tmp_path / "gercek.mp4"

    with VideoWriter(target, fps=10) as video:
        for value in range(6):
            video.write(frame(value=value * 20))

    assert target.exists() and target.stat().st_size > 0
    capture = cv2.VideoCapture(str(target))
    read = 0
    while capture.read()[0]:
        read += 1
    capture.release()
    assert read == 6, "yazılan kare sayısı geri okunanla eşleşmeli"
