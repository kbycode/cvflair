"""
Box outline styles that ``supervision`` does not ship.

Everything here subclasses ``supervision``'s ``BaseAnnotator`` and resolves
colours through its own ``resolve_color``, so colour palettes, ``ColorLookup``
and the ``Detections`` contract behave exactly as in the built-in annotators.
Only the outline geometry is new: dashes, rounded brackets, a reticle and a
target lock.

Each annotator draws with a handful of OpenCV calls per detection, which is the
same shape as ``supervision``'s own annotators; the accent colour is a second
palette used for the parts that should stand out from the edges.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import supervision as sv
from supervision.annotators.base import BaseAnnotator
from supervision.annotators.utils import resolve_color

__all__ = [
    "DashedBoxAnnotator",
    "DashedCornerAnnotator",
    "BracketBoxAnnotator",
    "CrosshairAnnotator",
    "TargetBoxAnnotator",
]

Palette = sv.Color | sv.ColorPalette


class _OutlineAnnotator(BaseAnnotator):
    """Shared plumbing: iterate detections, resolve colours, hand over the box."""

    def __init__(
        self,
        color: Palette = sv.ColorPalette.DEFAULT,
        thickness: int = 2,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
        accent_color: Palette | None = None,
    ) -> None:
        self.color = color
        self.thickness = max(1, int(thickness))
        self.color_lookup = color_lookup
        self.accent_color = accent_color

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        custom_color_lookup: np.ndarray | None = None,
    ) -> np.ndarray:
        lookup = self.color_lookup if custom_color_lookup is None else custom_color_lookup
        for index in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[index].astype(int)
            colour = resolve_color(self.color, detections, index, lookup).as_bgr()
            accent = (
                resolve_color(self.accent_color, detections, index, lookup).as_bgr()
                if self.accent_color is not None
                else colour
            )
            self.draw(scene, (int(x1), int(y1), int(x2), int(y2)), colour, accent)
        return scene

    def draw(
        self,
        scene: np.ndarray,
        box: tuple[int, int, int, int],
        colour: tuple[int, int, int],
        accent: tuple[int, int, int],
    ) -> None:  # pragma: no cover - overridden by every subclass
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(thickness={self.thickness})"


def _dashed_line(
    scene: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int,
    dash: int,
    gap: int,
) -> None:
    length = math.dist(start, end)
    if length == 0:
        return
    step = dash + gap
    ux = (end[0] - start[0]) / length
    uy = (end[1] - start[1]) / length
    travelled = 0.0
    while travelled < length:
        run = min(dash, length - travelled)
        end_at = travelled + run
        first = (round(start[0] + ux * travelled), round(start[1] + uy * travelled))
        second = (round(start[0] + ux * end_at), round(start[1] + uy * end_at))
        cv2.line(scene, first, second, colour, thickness)
        travelled += step


class DashedBoxAnnotator(_OutlineAnnotator):
    """A rectangle drawn as dashes. Reads as "tracked but not confirmed"."""

    def __init__(self, *args: Any, dash_length: int = 12, gap_length: int = 8, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dash_length = max(1, int(dash_length))
        self.gap_length = max(1, int(gap_length))

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
            _dashed_line(
                scene, start, end, colour, self.thickness, self.dash_length, self.gap_length
            )


class DashedCornerAnnotator(_OutlineAnnotator):
    """
    The dashed frame with solid corner brackets laid over it.

    Both styles are drawn: the full rectangle as dashes in the detection colour,
    then the corner arms as unbroken lines. With ``accent_color`` set the corners
    take the second colour, which is what makes the hybrid readable -- a dashed
    outline that still has hard, solid corners marking the box.
    """

    def __init__(
        self,
        *args: Any,
        corner_length: int = 26,
        dash_length: int = 7,
        gap_length: int = 5,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))
        self.dash_length = max(1, int(dash_length))
        self.gap_length = max(1, int(gap_length))

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
            _dashed_line(
                scene, start, end, colour, self.thickness, self.dash_length, self.gap_length
            )

        arm = int(min(self.corner_length, min(width, height) / 2))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(scene, (cx, cy), (cx + dx * arm, cy), accent, self.thickness)
            cv2.line(scene, (cx, cy), (cx, cy + dy * arm), accent, self.thickness)


class BracketBoxAnnotator(_OutlineAnnotator):
    """
    Corner brackets with rounded elbows -- the corner and round styles combined.

    ``roundness`` sets the elbow radius as a share of the shorter box side, the
    same meaning it has in ``supervision``'s ``RoundBoxAnnotator``.
    """

    def __init__(
        self, *args: Any, corner_length: int = 22, roundness: float = 0.4, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))
        self.roundness = min(max(float(roundness), 0.01), 1.0)

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return

        # Kol, yayın bittiği yerden devam eder: yarıçap kolun tamamını yerse
        # düz parça sıfır uzunlukta kalır ve köşe sadece yay gibi görünür.
        arm = int(min(self.corner_length, min(width, height) / 2))
        radius = int(min(self.roundness * min(width, height) / 2, arm * 0.6))

        corners = (
            ((x1, y1), (1, 1), 180),
            ((x2, y1), (-1, 1), 270),
            ((x2, y2), (-1, -1), 0),
            ((x1, y2), (1, -1), 90),
        )
        for (cx, cy), (dx, dy), start_angle in corners:
            centre = (cx + dx * radius, cy + dy * radius)
            cv2.ellipse(
                scene, centre, (radius, radius), 0, start_angle, start_angle + 90,
                accent, self.thickness, cv2.LINE_AA,
            )
            cv2.line(scene, (cx + dx * radius, cy), (cx + dx * arm, cy), colour, self.thickness)
            cv2.line(scene, (cx, cy + dy * radius), (cx, cy + dy * arm), colour, self.thickness)


class CrosshairAnnotator(_OutlineAnnotator):
    """Mid-edge ticks plus a centre cross: a reticle instead of a frame."""

    def __init__(self, *args: Any, arm_length: int = 18, center_size: int = 10, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.arm_length = max(1, int(arm_length))
        self.center_size = max(0, int(center_size))

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        arm_x = int(min(self.arm_length, (x2 - x1) / 2))
        arm_y = int(min(self.arm_length, (y2 - y1) / 2))

        cv2.line(scene, (cx - arm_x, y1), (cx + arm_x, y1), colour, self.thickness)
        cv2.line(scene, (cx - arm_x, y2), (cx + arm_x, y2), colour, self.thickness)
        cv2.line(scene, (x1, cy - arm_y), (x1, cy + arm_y), colour, self.thickness)
        cv2.line(scene, (x2, cy - arm_y), (x2, cy + arm_y), colour, self.thickness)

        if self.center_size:
            half = self.center_size // 2
            cv2.line(scene, (cx - half, cy), (cx + half, cy), accent, self.thickness)
            cv2.line(scene, (cx, cy - half), (cx, cy + half), accent, self.thickness)


class TargetBoxAnnotator(_OutlineAnnotator):
    """
    Thin full rectangle with heavy corner brackets -- a target-lock frame.

    With ``accent_color`` set, the brackets take the accent and the rectangle
    keeps the detection colour, which is where the two-tone look comes from.
    """

    def __init__(
        self, *args: Any, corner_length: int = 26, edge_thickness: int = 1, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))
        self.edge_thickness = max(1, int(edge_thickness))

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return

        cv2.rectangle(scene, (x1, y1), (x2, y2), colour, self.edge_thickness)

        arm = int(min(self.corner_length, min(width, height) / 2))
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(scene, (cx, cy), (cx + dx * arm, cy), accent, self.thickness)
            cv2.line(scene, (cx, cy), (cx, cy + dy * arm), accent, self.thickness)
