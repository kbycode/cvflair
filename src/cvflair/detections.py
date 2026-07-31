"""
The detection container.

cvflair draws whatever carries boxes: its own :class:`Detections`, a
``supervision.Detections``, or any object with the same attribute names. The
drawing code only ever reads ``len()``, ``xyxy``, ``class_id``, ``confidence``,
``tracker_id`` and the class names, so nothing has to be converted first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Detections", "is_detections", "detection_names"]


@dataclass
class Detections:
    """
    Boxes and their metadata.

    ``xyxy`` is the only required field: an ``(N, 4)`` array of
    ``[x1, y1, x2, y2]`` in pixels. The rest is optional and, when present,
    must have one entry per box.
    """

    xyxy: np.ndarray
    class_id: np.ndarray | None = None
    confidence: np.ndarray | None = None
    tracker_id: np.ndarray | None = None
    #: Class names, one per box. ``supervision`` keeps these in ``data``; both
    #: are read by :func:`detection_names`.
    names: np.ndarray | None = None
    #: Segmentation masks, ``(N, H, W)`` boolean -- one full-frame mask per box.
    #: Only carried and drawn; nothing here computes them.
    mask: np.ndarray | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        count = len(self.xyxy)

        for name in ("class_id", "tracker_id"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, np.asarray(value, dtype=int).reshape(-1))
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        if self.names is not None:
            self.names = np.asarray(self.names, dtype=object).reshape(-1)
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.ndim == 2:  # a single mask was handed over
                self.mask = self.mask[None, ...]
            if self.mask.ndim != 3:
                raise ValueError(f"mask must be shaped (N, H, W), got {self.mask.shape}.")

        for name in ("class_id", "confidence", "tracker_id", "names", "mask"):
            value = getattr(self, name)
            if value is not None and len(value) != count:
                raise ValueError(
                    f"{name} has {len(value)} entries but xyxy has {count} boxes."
                )

    def __len__(self) -> int:
        return len(self.xyxy)

    @classmethod
    def empty(cls) -> Detections:
        return cls(xyxy=np.empty((0, 4), dtype=np.float32))

    @classmethod
    def from_arrays(
        cls,
        xyxy: Any,
        class_id: Any = None,
        confidence: Any = None,
        names: Any = None,
        tracker_id: Any = None,
    ) -> Detections:
        """Explicit constructor for models that hand back plain arrays."""
        return cls(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            names=names,
            tracker_id=tracker_id,
        )

    @classmethod
    def from_ultralytics(cls, result: Any) -> Detections:
        """
        Read one Ultralytics ``Results`` object.

        Detection boxes only -- segmentation masks and oriented boxes are not
        carried over; ``supervision.Detections.from_ultralytics`` handles those
        and its output can be passed to cvflair unchanged.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return cls.empty()

        class_id = _to_numpy(boxes.cls).astype(int)
        lookup = getattr(result, "names", None)
        names = (
            np.array([lookup[int(index)] for index in class_id], dtype=object)
            if lookup is not None
            else None
        )
        tracker = getattr(boxes, "id", None)

        return cls(
            xyxy=_to_numpy(boxes.xyxy),
            class_id=class_id,
            confidence=_to_numpy(boxes.conf),
            tracker_id=_to_numpy(tracker).astype(int) if tracker is not None else None,
            names=names,
        )

    def __repr__(self) -> str:
        return f"Detections({len(self)} boxes)"


def _to_numpy(value: Any) -> np.ndarray:
    """Torch tensors, numpy arrays and plain sequences all end up as numpy."""
    for step in ("cpu", "numpy"):
        method = getattr(value, step, None)
        if callable(method):
            value = method()
    return np.asarray(value)


def is_detections(value: Any) -> bool:
    """True when ``value`` carries enough to be drawn."""
    return hasattr(value, "xyxy") and hasattr(value, "__len__")


def detection_names(detections: Any) -> np.ndarray | None:
    """
    Class names for the detections, or ``None``.

    Reads cvflair's ``names`` first, then ``data["class_name"]``, which is where
    ``supervision`` keeps them.
    """
    names = getattr(detections, "names", None)
    if names is not None:
        return np.asarray(names, dtype=object)

    data = getattr(detections, "data", None)
    if isinstance(data, dict) and "class_name" in data:
        return np.asarray(data["class_name"], dtype=object)
    return None
