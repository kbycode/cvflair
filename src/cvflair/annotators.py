"""
Every outline style cvflair can draw.

The three familiar ones (box, rounded box, corners) plus the five it adds
(dashes, dashed corners, brackets, reticle, target lock) and the label plate.
All of them share one loop: walk the detections, resolve a colour per box, draw
with a handful of OpenCV calls. Nothing here allocates per frame beyond what
OpenCV needs, and the annotator instances are meant to be built once and reused
-- :class:`cvflair.Theme` does exactly that.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .colors import Color, ColorLookup, ColorPalette, resolve_color, resolve_palette
from .detections import detection_names

__all__ = [
    "BoxAnnotator",
    "RoundBoxAnnotator",
    "BoxCornerAnnotator",
    "DashedBoxAnnotator",
    "DashedCornerAnnotator",
    "BracketBoxAnnotator",
    "CrosshairAnnotator",
    "TargetBoxAnnotator",
    "LabelAnnotator",
    "HudAnnotator",
    "HUD_POSITIONS",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: Where a HUD panel can sit.
HUD_POSITIONS: tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")


# -- shared drawing helpers -------------------------------------------------


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
        end_at = travelled + min(dash, length - travelled)
        first = (round(start[0] + ux * travelled), round(start[1] + uy * travelled))
        second = (round(start[0] + ux * end_at), round(start[1] + uy * end_at))
        cv2.line(scene, first, second, colour, thickness)
        travelled += step


def _rounded_outline(
    scene: np.ndarray,
    box: tuple[int, int, int, int],
    radius: int,
    colour: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = box
    radius = int(min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    if radius <= 0:
        cv2.rectangle(scene, (x1, y1), (x2, y2), colour, thickness)
        return

    cv2.line(scene, (x1 + radius, y1), (x2 - radius, y1), colour, thickness)
    cv2.line(scene, (x1 + radius, y2), (x2 - radius, y2), colour, thickness)
    cv2.line(scene, (x1, y1 + radius), (x1, y2 - radius), colour, thickness)
    cv2.line(scene, (x2, y1 + radius), (x2, y2 - radius), colour, thickness)

    for centre, start_angle in (
        ((x1 + radius, y1 + radius), 180),
        ((x2 - radius, y1 + radius), 270),
        ((x2 - radius, y2 - radius), 0),
        ((x1 + radius, y2 - radius), 90),
    ):
        cv2.ellipse(
            scene, centre, (radius, radius), 0, start_angle, start_angle + 90,
            colour, thickness, cv2.LINE_AA,
        )


def _rounded_fill(
    scene: np.ndarray,
    box: tuple[int, int, int, int],
    radius: int,
    colour: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    radius = int(min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
    if radius <= 0:
        cv2.rectangle(scene, (x1, y1), (x2, y2), colour, -1)
        return

    cv2.rectangle(scene, (x1 + radius, y1), (x2 - radius, y2), colour, -1)
    cv2.rectangle(scene, (x1, y1 + radius), (x2, y2 - radius), colour, -1)
    for centre in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(scene, centre, radius, colour, -1, cv2.LINE_AA)


def _corner_arm(width: int, height: int, corner_length: int) -> int:
    return int(min(corner_length, min(width, height) / 2))


CORNERS = ((0, 1, 1, 1), (2, 1, -1, 1), (0, 3, 1, -1), (2, 3, -1, -1))


# -- outline annotators -----------------------------------------------------


class _OutlineAnnotator:
    """Shared plumbing: iterate detections, resolve colours, hand over the box."""

    def __init__(
        self,
        color: Any = None,
        thickness: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        accent_color: Any = None,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.thickness = max(1, int(thickness))
        self.color_lookup = color_lookup
        self.accent_color = None if accent_color is None else resolve_palette(accent_color)

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        for index in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[index].astype(int)
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            accent = (
                resolve_color(self.accent_color, detections, index, self.color_lookup).as_bgr()
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


class BoxAnnotator(_OutlineAnnotator):
    """A plain rectangle."""

    def draw(self, scene, box, colour, accent) -> None:
        cv2.rectangle(scene, (box[0], box[1]), (box[2], box[3]), colour, self.thickness)


class RoundBoxAnnotator(_OutlineAnnotator):
    """A rectangle with rounded corners. ``roundness`` is a share of the shorter side."""

    def __init__(self, *args: Any, roundness: float = 0.6, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.roundness = min(max(float(roundness), 0.01), 1.0)

    def draw(self, scene, box, colour, accent) -> None:
        short_side = min(box[2] - box[0], box[3] - box[1])
        _rounded_outline(scene, box, int(self.roundness * short_side / 2), colour, self.thickness)


class BoxCornerAnnotator(_OutlineAnnotator):
    """Only the four corners, as solid arms."""

    def __init__(self, *args: Any, corner_length: int = 15, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))

    def draw(self, scene, box, colour, accent) -> None:
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            return
        arm = _corner_arm(width, height, self.corner_length)
        for ix, iy, dx, dy in CORNERS:
            cx, cy = box[ix], box[iy]
            cv2.line(scene, (cx, cy), (cx + dx * arm, cy), colour, self.thickness)
            cv2.line(scene, (cx, cy), (cx, cy + dy * arm), colour, self.thickness)


class DashedBoxAnnotator(_OutlineAnnotator):
    """A rectangle drawn as dashes."""

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


class DashedCornerAnnotator(DashedBoxAnnotator):
    """
    The dashed frame with solid corner brackets laid over it.

    With ``accent_color`` set the corners take the second colour, which is what
    makes the hybrid readable.
    """

    def __init__(self, *args: Any, corner_length: int = 26, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))

    def draw(self, scene, box, colour, accent) -> None:
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            return
        super().draw(scene, box, colour, accent)

        arm = _corner_arm(width, height, self.corner_length)
        for ix, iy, dx, dy in CORNERS:
            cx, cy = box[ix], box[iy]
            cv2.line(scene, (cx, cy), (cx + dx * arm, cy), accent, self.thickness)
            cv2.line(scene, (cx, cy), (cx, cy + dy * arm), accent, self.thickness)


class BracketBoxAnnotator(_OutlineAnnotator):
    """Corner brackets with rounded elbows -- the corner and round styles combined."""

    def __init__(
        self, *args: Any, corner_length: int = 22, roundness: float = 0.4, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.corner_length = max(1, int(corner_length))
        self.roundness = min(max(float(roundness), 0.01), 1.0)

    def draw(self, scene, box, colour, accent) -> None:
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            return

        # The arm continues where the elbow ends; a radius as long as the arm
        # would leave no straight part at all.
        arm = _corner_arm(width, height, self.corner_length)
        radius = int(min(self.roundness * min(width, height) / 2, arm * 0.6))

        for (ix, iy, dx, dy), start_angle in zip(CORNERS, (180, 270, 90, 0), strict=True):
            cx, cy = box[ix], box[iy]
            cv2.ellipse(
                scene, (cx + dx * radius, cy + dy * radius), (radius, radius), 0,
                start_angle, start_angle + 90, accent, self.thickness, cv2.LINE_AA,
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
    """Thin full rectangle with heavy corner brackets -- a target-lock frame."""

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
        arm = _corner_arm(width, height, self.corner_length)
        for ix, iy, dx, dy in CORNERS:
            cx, cy = box[ix], box[iy]
            cv2.line(scene, (cx, cy), (cx + dx * arm, cy), accent, self.thickness)
            cv2.line(scene, (cx, cy), (cx, cy + dy * arm), accent, self.thickness)


# -- labels -----------------------------------------------------------------


class LabelAnnotator:
    """
    The filled plate above each box, with the class name inside.

    Text comes from ``labels`` when given, otherwise from the detections' class
    names, then the class id, then the box index.
    """

    def __init__(
        self,
        color: Any = None,
        text_color: Any = Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 6,
        border_radius: int = 0,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.text_color = resolve_palette(text_color).colors[0]
        self.text_scale = float(text_scale)
        self.text_thickness = max(1, int(text_thickness))
        self.text_padding = max(0, int(text_padding))
        self.border_radius = max(0, int(border_radius))
        self.color_lookup = color_lookup

    def annotate(
        self, scene: np.ndarray, detections: Any, labels: Any = None
    ) -> np.ndarray:
        texts = self._texts(detections, labels)
        height_limit = scene.shape[0]

        for index in range(len(detections)):
            x1, y1 = detections.xyxy[index][:2].astype(int)
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            text = texts[index]

            (text_width, text_height), _ = cv2.getTextSize(
                text, FONT, self.text_scale, self.text_thickness
            )
            plate_width = text_width + self.text_padding * 2
            plate_height = text_height + self.text_padding * 2

            top = int(y1) - plate_height
            if top < 0:  # no room above the box, so the plate sits inside it
                top = int(y1)
            bottom = min(top + plate_height, height_limit)
            left = int(x1)

            plate = (left, top, left + plate_width, bottom)
            _rounded_fill(scene, plate, self.border_radius, colour)
            cv2.putText(
                scene,
                text,
                (left + self.text_padding, bottom - self.text_padding),
                FONT,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
                cv2.LINE_AA,
            )
        return scene

    def _texts(self, detections: Any, labels: Any) -> list[str]:
        count = len(detections)
        if labels is not None:
            labels = list(labels)
            if len(labels) != count:
                raise ValueError(
                    f"Got {len(labels)} labels for {count} detections; they must match."
                )
            return [str(label) for label in labels]

        names = detection_names(detections)
        if names is not None:
            return [str(name) for name in names]

        class_id = getattr(detections, "class_id", None)
        if class_id is not None:
            return [str(int(value)) for value in class_id]
        return [str(index) for index in range(count)]

    def __repr__(self) -> str:
        return f"LabelAnnotator(text_scale={self.text_scale})"


# -- heads-up display -------------------------------------------------------


class HudAnnotator:
    """
    A small stats panel in one corner: FPS, counters, whatever is handed over.

    ``annotate`` takes a mapping, not detections -- the numbers come from the
    loop around the frame, not from the boxes in it. :class:`cvflair.Camera`
    fills in frame rate and detection count; anything else is the caller's.

    The plate is blended into the frame so the scene stays visible behind it,
    while the text is drawn at full strength to stay readable.
    """

    def __init__(
        self,
        color: Any = None,
        text_color: Any = Color.WHITE,
        background: Any = "#0B0D11",
        opacity: float = 0.6,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        padding: int = 10,
        line_gap: int = 8,
        position: str = "top_left",
        margin: int = 14,
        border_radius: int = 8,
    ) -> None:
        if position not in HUD_POSITIONS:
            raise ValueError(
                f"Unknown hud position {position!r}. Use one of: {', '.join(HUD_POSITIONS)}."
            )
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.text_color = resolve_palette(text_color).colors[0]
        self.background = resolve_palette(background).colors[0]
        self.opacity = min(max(float(opacity), 0.0), 1.0)
        self.text_scale = float(text_scale)
        self.text_thickness = max(1, int(text_thickness))
        self.padding = max(0, int(padding))
        self.line_gap = max(0, int(line_gap))
        self.position = position
        self.margin = max(0, int(margin))
        self.border_radius = max(0, int(border_radius))

    def annotate(self, scene: np.ndarray, stats: Any) -> np.ndarray:
        if not stats:
            return scene

        lines = [f"{key}  {value}" for key, value in dict(stats).items()]
        sizes = [
            cv2.getTextSize(line, FONT, self.text_scale, self.text_thickness)[0] for line in lines
        ]
        line_height = max(height for _, height in sizes)
        bar_width = 3

        width = max(w for w, _ in sizes) + self.padding * 2 + bar_width + 6
        height = line_height * len(lines) + self.line_gap * (len(lines) - 1) + self.padding * 2

        frame_height, frame_width = scene.shape[:2]
        width = min(width, frame_width)
        height = min(height, frame_height)

        at_left = self.position.endswith("left")
        at_top = self.position.startswith("top")
        left = self.margin if at_left else frame_width - width - self.margin
        top = self.margin if at_top else frame_height - height - self.margin
        left = max(0, min(left, frame_width - width))
        top = max(0, min(top, frame_height - height))

        region = scene[top : top + height, left : left + width]
        plate = region.copy()
        _rounded_fill(
            plate, (0, 0, width - 1, height - 1), self.border_radius, self.background.as_bgr()
        )
        cv2.addWeighted(plate, self.opacity, region, 1 - self.opacity, 0, region)

        accent = self.color.by_index(0).as_bgr()
        cv2.rectangle(
            region,
            (self.padding // 2, self.padding),
            (self.padding // 2 + bar_width, height - self.padding),
            accent,
            -1,
        )

        text_left = self.padding + bar_width + 6
        for index, line in enumerate(lines):
            baseline = self.padding + line_height * (index + 1) + self.line_gap * index
            cv2.putText(
                region,
                line,
                (text_left, baseline),
                FONT,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
                cv2.LINE_AA,
            )
        return scene

    def __repr__(self) -> str:
        return f"HudAnnotator(position={self.position!r})"
