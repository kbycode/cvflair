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

from . import notebook
from .annotators import (
    HUD_POSITIONS,
    BlurAnnotator,
    BoxAnnotator,
    BoxCornerAnnotator,
    BracketBoxAnnotator,
    ConfidenceBarAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    EdgeAnnotator,
    HudAnnotator,
    LabelAnnotator,
    MaskAnnotator,
    PulseAnnotator,
    RoundBoxAnnotator,
    SketchBoxAnnotator,
    TargetBoxAnnotator,
    TraceAnnotator,
    VertexAnnotator,
    ZoneAnnotator,
)
from .camera import Camera, CameraError
from .colors import Color, ColorLookup, ColorPalette
from .detections import Detections
from .keypoints import HAND_21, POSE_17, SKELETONS, KeyPoints, Skeleton
from .models import Detector, UltralyticsDetector, load_ultralytics, resolve_detector
from .themes import BOX_STYLES, Theme, available_themes, get_theme
from .video import VideoWriteError, VideoWriter

__version__ = "0.10.0"

__all__ = [
    "notebook",
    "BOX_STYLES",
    "BlurAnnotator",
    "ZoneAnnotator",
    "HUD_POSITIONS",
    "BoxAnnotator",
    "BoxCornerAnnotator",
    "BracketBoxAnnotator",
    "Camera",
    "CameraError",
    "Color",
    "ConfidenceBarAnnotator",
    "ColorLookup",
    "ColorPalette",
    "CrosshairAnnotator",
    "DashedBoxAnnotator",
    "DashedCornerAnnotator",
    "Detections",
    "VertexAnnotator",
    "Skeleton",
    "SKELETONS",
    "POSE_17",
    "KeyPoints",
    "HAND_21",
    "EdgeAnnotator",
    "Detector",
    "HudAnnotator",
    "LabelAnnotator",
    "MaskAnnotator",
    "PulseAnnotator",
    "RoundBoxAnnotator",
    "SketchBoxAnnotator",
    "TargetBoxAnnotator",
    "TraceAnnotator",
    "Theme",
    "UltralyticsDetector",
    "available_themes",
    "get_theme",
    "load_ultralytics",
    "resolve_detector",
    "VideoWriter",
    "VideoWriteError",
    "__version__",
]
