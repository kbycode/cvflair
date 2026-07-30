"""
Optional model binding.

cvflair ships no model and depends on none. This module only turns whatever was
passed as ``model`` into a callable that maps a frame to ``sv.Detections``:

* a weights path (``"yolov8n.pt"``) -- loaded through Ultralytics, which is an
  extra (``pip install "cvflair[yolo]"``), never a hard dependency;
* an already constructed Ultralytics model -- its results are converted;
* any callable returning ``sv.Detections`` -- used as it is.

Keeping Ultralytics out of the dependency list is deliberate: it is AGPL-3.0,
and pulling it in would push that licence onto every cvflair user.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import supervision as sv

__all__ = [
    "Detector",
    "ModelLike",
    "UltralyticsDetector",
    "load_ultralytics",
    "resolve_detector",
]

#: Weight file suffixes Ultralytics can load.
WEIGHT_SUFFIXES = frozenset({".pt", ".onnx", ".engine", ".torchscript", ".mlpackage"})


@runtime_checkable
class Detector(Protocol):
    """Anything that turns one frame into detections."""

    def __call__(self, frame: np.ndarray) -> sv.Detections: ...


ModelLike = str | Path | Detector | Any


class UltralyticsDetector:
    """
    Runs an Ultralytics model and converts its output to ``sv.Detections``.

    ``predict_kwargs`` is forwarded to every call (``conf``, ``iou``, ``device``,
    ``classes`` ...). ``verbose`` defaults to ``False`` so the capture loop is
    not drowned in per-frame logging.
    """

    def __init__(
        self,
        model: Any,
        *,
        convert: Callable[[Any], sv.Detections] | None = None,
        **predict_kwargs: Any,
    ) -> None:
        self.model = model
        self.predict_kwargs = {"verbose": False, **predict_kwargs}
        self._convert = convert or sv.Detections.from_ultralytics

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, **self.predict_kwargs)
        return self._convert(results[0])

    def __repr__(self) -> str:
        return f"UltralyticsDetector({type(self.model).__name__})"


class _CallableDetector:
    """Wraps a user callable so a wrong return type fails with a clear message."""

    def __init__(self, function: Callable[[np.ndarray], sv.Detections]) -> None:
        self.function = function

    def __call__(self, frame: np.ndarray) -> sv.Detections:
        detections = self.function(frame)
        if not isinstance(detections, sv.Detections):
            raise TypeError(
                f"model returned {type(detections).__name__}, expected supervision.Detections. "
                "Convert the model output first, e.g. sv.Detections.from_ultralytics(result)."
            )
        return detections

    def __repr__(self) -> str:
        name = getattr(self.function, "__name__", type(self.function).__name__)
        return f"_CallableDetector({name})"


def load_ultralytics(weights: str | Path, **predict_kwargs: Any) -> UltralyticsDetector:
    """Load Ultralytics weights. Raises ``ImportError`` when the extra is missing."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            f"Loading {str(weights)!r} needs the ultralytics package: "
            'pip install "cvflair[yolo]". It is an optional extra because Ultralytics '
            "is AGPL-3.0 licensed; cvflair itself stays MIT."
        ) from exc
    return UltralyticsDetector(YOLO(str(weights)), **predict_kwargs)


def _is_ultralytics_model(model: Any) -> bool:
    return type(model).__module__.split(".")[0] == "ultralytics"


def resolve_detector(model: ModelLike | None) -> Detector | None:
    """
    Turn ``model`` into a detector callable, or ``None`` when nothing was given.

    Weights are loaded here, so calling this is what triggers the (slow) model
    load -- ``Camera.stream()`` does it on the first iteration, not before.
    """
    if model is None:
        return None

    if isinstance(model, (str, Path)):
        suffix = Path(model).suffix.lower()
        if suffix not in WEIGHT_SUFFIXES:
            raise ValueError(
                f"Unrecognised weights file {str(model)!r}. "
                f"Expected one of: {', '.join(sorted(WEIGHT_SUFFIXES))}."
            )
        return load_ultralytics(model)

    if _is_ultralytics_model(model):
        return UltralyticsDetector(model)

    if callable(model):
        return _CallableDetector(model)

    raise TypeError(
        f"model must be a weights path, an Ultralytics model, or a callable returning "
        f"supervision.Detections; got {type(model).__name__}."
    )
