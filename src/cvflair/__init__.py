"""
cvflair -- theme-based visualisation for computer vision detections.

Numpy and OpenCV are the only dependencies. The camera loop, the themes and the
drawing all live here, so the everyday case fits in three lines::

    from cvflair import Camera

    cam = Camera(source=0, theme="neon")
    for frame, detections in cam.stream(model="yolov8n.pt"):
        cam.show(frame, detections)

Detections may be cvflair's own :class:`~cvflair.detections.Detections`, a
``supervision.Detections``, or any object carrying the same fields.
"""

from __future__ import annotations

from .annotators import (
    HUD_POSITIONS,
    BoxAnnotator,
    BoxCornerAnnotator,
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    HudAnnotator,
    LabelAnnotator,
    RoundBoxAnnotator,
    TargetBoxAnnotator,
)
from .camera import Camera, CameraError
from .colors import Color, ColorLookup, ColorPalette
from .detections import Detections
from .models import Detector, UltralyticsDetector, load_ultralytics, resolve_detector
from .themes import BOX_STYLES, Theme, available_themes, get_theme

__version__ = "0.5.0"

__all__ = [
    "BOX_STYLES",
    "HUD_POSITIONS",
    "BoxAnnotator",
    "BoxCornerAnnotator",
    "BracketBoxAnnotator",
    "Camera",
    "CameraError",
    "Color",
    "ColorLookup",
    "ColorPalette",
    "CrosshairAnnotator",
    "DashedBoxAnnotator",
    "DashedCornerAnnotator",
    "Detections",
    "Detector",
    "HudAnnotator",
    "LabelAnnotator",
    "RoundBoxAnnotator",
    "TargetBoxAnnotator",
    "Theme",
    "UltralyticsDetector",
    "available_themes",
    "get_theme",
    "load_ultralytics",
    "resolve_detector",
    "__version__",
]
