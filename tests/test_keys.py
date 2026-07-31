"""
Klavye: `show()` basılan tuşu tutuyor mu, çıkış tuşları hâlâ çalışıyor mu.

Pencere açılmıyor; `cv2.imshow`, `waitKey` ve pencere sorgusu yerine sahteleri
konuyor, böylece testler başsız ortamda da geçiyor.
"""

from __future__ import annotations

import cv2
import pytest
from conftest import make_frame


@pytest.fixture
def headless(monkeypatch):
    """GUI çağrılarını sahteler; dönen tuş kodunu testin belirlemesine izin verir."""
    state = {"key": -1, "visible": 1.0, "shown": 0}

    monkeypatch.setattr(cv2, "imshow", lambda *args: state.__setitem__("shown", state["shown"] + 1))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: state["key"])
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: state["visible"])
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    return state


def test_no_key_reads_as_minus_one(camera_factory, headless):
    camera, _ = camera_factory(frame_count=1)
    headless["key"] = -1

    assert camera.show(make_frame(1)) is True
    assert camera.key == -1
    assert not camera.pressed("1")


def test_pressed_key_is_kept(camera_factory, headless):
    camera, _ = camera_factory(frame_count=1)
    headless["key"] = ord("1")

    camera.show(make_frame(1))

    assert camera.key == ord("1")
    assert camera.pressed("1")
    assert camera.pressed(ord("1"))
    assert not camera.pressed("2")


def test_key_is_refreshed_every_frame(camera_factory, headless):
    camera, _ = camera_factory(frame_count=1)

    headless["key"] = ord("3")
    camera.show(make_frame(1))
    assert camera.pressed("3")

    headless["key"] = -1
    camera.show(make_frame(2))
    assert camera.key == -1, "önceki tuş takılı kalmamalı"


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_quit_keys_stop_the_stream(camera_factory, headless, key):
    camera, _ = camera_factory(frame_count=1)
    headless["key"] = key

    assert camera.show(make_frame(1)) is False
    assert camera._stop_event.is_set()


def test_closed_window_stops_the_stream(camera_factory, headless):
    camera, _ = camera_factory(frame_count=1)
    headless["visible"] = 0.0

    assert camera.show(make_frame(1)) is False
    assert camera._stop_event.is_set()


def test_theme_can_be_switched_from_a_key(camera_factory, headless):
    """Örneklerdeki kullanım: tuşa göre tema değiştirmek."""
    camera, _ = camera_factory(frame_count=1, theme="minimal")

    headless["key"] = ord("2")
    camera.show(make_frame(1))
    if camera.pressed("2"):
        camera.theme = "neon"

    assert camera.theme.name == "neon"
