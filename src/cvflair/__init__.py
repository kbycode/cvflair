"""
cvflair -- theme-based visualisation for computer vision detections.

The package draws nothing itself: it configures ``supervision`` annotators and
wraps the camera loop, so the everyday case fits in three lines::

    from cvflair import Camera

    cam = Camera(source=0, theme="neon")
    for frame in cam.stream():
        cam.show(frame)
"""

from __future__ import annotations

from .annotators import (
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    TargetBoxAnnotator,
)
from .camera import Camera, CameraError
from .models import Detector, UltralyticsDetector, load_ultralytics, resolve_detector
from .themes import BOX_STYLES, Theme, available_themes, get_theme

__version__ = "0.3.0"

__all__ = [
    "BOX_STYLES",
    "BracketBoxAnnotator",
    "Camera",
    "CameraError",
    "CrosshairAnnotator",
    "DashedBoxAnnotator",
    "DashedCornerAnnotator",
    "Detector",
    "TargetBoxAnnotator",
    "Theme",
    "UltralyticsDetector",
    "available_themes",
    "get_theme",
    "load_ultralytics",
    "resolve_detector",
    "__version__",
]
