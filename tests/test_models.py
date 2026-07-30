"""
Model binding tests.

The Ultralytics path is exercised with a stand-in Results object, so the real
conversion is covered without pulling in torch or downloading weights.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from conftest import FRAME_SHAPE, make_frame

from cvflair import Detections
from cvflair.models import UltralyticsDetector, load_ultralytics, resolve_detector

ULTRALYTICS_INSTALLED = importlib.util.find_spec("ultralytics") is not None


class _Tensor:
    """Mimics the torch tensors Ultralytics hands back."""

    def __init__(self, array) -> None:
        self._array = np.asarray(array)

    def cpu(self) -> _Tensor:
        return self

    def int(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _Boxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self.cls = _Tensor(cls)
        self.id = None


class FakeResults:
    """The subset of ultralytics.engine.results.Results that supervision reads."""

    def __init__(self) -> None:
        self.boxes = _Boxes(
            xyxy=[[10.0, 20.0, 60.0, 90.0], [70.0, 30.0, 120.0, 100.0]],
            conf=[0.9, 0.5],
            cls=[0, 1],
        )
        self.masks = None
        self.obb = None
        self.names = {0: "kisi", 1: "top"}
        self.orig_shape = FRAME_SHAPE[:2]


class FakeYOLO:
    """Callable model that records how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, frame, **kwargs):
        self.calls.append(kwargs)
        return [FakeResults()]


def test_none_means_no_detector():
    assert resolve_detector(None) is None


def test_callable_returning_detections_is_used(detections):
    detector = resolve_detector(lambda frame: detections)

    assert detector(make_frame(1)) is detections


def test_callable_returning_something_else_fails_clearly():
    detector = resolve_detector(lambda frame: "kutular")

    with pytest.raises(TypeError, match="expected detections"):
        detector(make_frame(1))


def test_non_callable_model_is_rejected():
    with pytest.raises(TypeError, match="model must be a weights path"):
        resolve_detector(42)


def test_unknown_weights_suffix_is_rejected():
    with pytest.raises(ValueError, match="Unrecognised weights file"):
        resolve_detector("model.txt")


@pytest.mark.skipif(ULTRALYTICS_INSTALLED, reason="ultralytics is installed here")
def test_missing_ultralytics_points_at_the_extra():
    with pytest.raises(ImportError, match=r'cvflair\[yolo\]'):
        load_ultralytics("yolov8n.pt")


def test_ultralytics_results_are_converted():
    detector = UltralyticsDetector(FakeYOLO())

    detections = detector(make_frame(1))

    assert isinstance(detections, Detections)
    assert len(detections) == 2
    assert detections.xyxy[0].tolist() == [10.0, 20.0, 60.0, 90.0]
    assert detections.class_id.tolist() == [0, 1]
    assert list(detections.names) == ["kisi", "top"]


def test_predict_kwargs_reach_the_model():
    model = FakeYOLO()
    detector = UltralyticsDetector(model, conf=0.4, device="cpu")

    detector(make_frame(1))

    assert model.calls == [{"verbose": False, "conf": 0.4, "device": "cpu"}]


def test_ultralytics_instances_are_detected_by_module(monkeypatch):
    monkeypatch.setattr(FakeYOLO, "__module__", "ultralytics.engine.model")

    assert isinstance(resolve_detector(FakeYOLO()), UltralyticsDetector)
