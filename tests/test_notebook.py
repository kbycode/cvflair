"""Not defteri gösterimi. IPython kurulu olmadan da çalışması gereken kısımlar."""

from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from cvflair.notebook import in_notebook, show, to_png


def frame(value: int = 90) -> np.ndarray:
    return np.full((20, 30, 3), value, dtype=np.uint8)


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


# -- kodlama ----------------------------------------------------------------


def test_png_is_produced():
    data = to_png(frame())

    assert data[:4] == b"\x89PNG"
    assert decode(data).shape == (20, 30, 3)


def test_pixels_survive_the_round_trip():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    original[:, :, 0] = 200  # yalnızca mavi kanal

    assert np.array_equal(decode(to_png(original)), original)


def test_rgb_input_is_swapped():
    """bgr=False verildiğinde kanallar çevrilmeli, yoksa kırmızı maviye dönüyor."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255  # RGB düzeninde kırmızı

    decoded = decode(to_png(rgb, bgr=False))

    assert decoded[0, 0, 2] == 255, "kırmızı, BGR'de üçüncü kanala gitmeli"
    assert decoded[0, 0, 0] == 0


def test_float_frame_in_zero_to_one_is_scaled():
    decoded = decode(to_png(np.ones((4, 4, 3), dtype=np.float32)))

    assert decoded.max() == 255, "0-1 aralığı 0-255'e ölçeklenmeli"


def test_float_frame_in_zero_to_255_is_kept():
    decoded = decode(to_png(np.full((4, 4, 3), 128.0, dtype=np.float64)))

    assert decoded.max() == 128


def test_grayscale_is_accepted():
    data = to_png(np.full((8, 8), 120, dtype=np.uint8))

    assert decode(data) is not None


def test_empty_frame_is_refused():
    with pytest.raises(ValueError, match="Boş kare"):
        to_png(np.empty((0, 0, 3), dtype=np.uint8))


# -- gösterim ---------------------------------------------------------------


class FakeDisplayModule(types.ModuleType):
    """`IPython.display` yerine geçer; çağrıları kaydeder."""

    def __init__(self) -> None:
        super().__init__("IPython.display")
        self.shown: list[object] = []
        self.Image = lambda data, format, width: {"data": data, "width": width}  # noqa: N803
        self.display = self.shown.append


@pytest.fixture
def fake_ipython(monkeypatch):
    display_module = FakeDisplayModule()
    root = types.ModuleType("IPython")
    root.display = display_module
    root.get_ipython = lambda: None  # varsayılan: kabuk yok
    monkeypatch.setitem(sys.modules, "IPython", root)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    return root, display_module


def test_show_needs_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)

    with pytest.raises(ImportError, match="IPython gerektiriyor"):
        show(frame())


def test_show_returns_the_image_outside_a_notebook(fake_ipython):
    _, display_module = fake_ipython

    result = show(frame())

    assert result["data"][:4] == b"\x89PNG"
    assert display_module.shown == [], "kabuk yokken çizilecek bir yer yok"


def test_show_draws_inside_a_notebook(fake_ipython, monkeypatch):
    root, display_module = fake_ipython
    shell = type("ZMQInteractiveShell", (), {})()
    monkeypatch.setattr(root, "get_ipython", lambda: shell)

    show(frame(), width=320)

    assert len(display_module.shown) == 1
    assert display_module.shown[0]["width"] == 320


def test_terminal_ipython_is_not_a_notebook(fake_ipython, monkeypatch):
    root, display_module = fake_ipython
    shell = type("TerminalInteractiveShell", (), {})()
    monkeypatch.setattr(root, "get_ipython", lambda: shell)

    assert in_notebook() is False
    show(frame())
    assert display_module.shown == [], "terminalde resim çizilemez"


def test_in_notebook_is_false_without_ipython(monkeypatch):
    monkeypatch.setitem(sys.modules, "IPython", None)

    assert in_notebook() is False
