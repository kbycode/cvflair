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
    "ConfidenceBarAnnotator",
    "BlurAnnotator",
    "MaskAnnotator",
    "ZoneAnnotator",
    "HudAnnotator",
    "EdgeAnnotator",
    "VertexAnnotator",
    "HUD_POSITIONS",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX

#: OpenCV drawing raises on coordinates past the int32 range. Boxes are clipped
#: to this instead of to the frame -- a box hanging off the edge should still be
#: drawn as far as it reaches.
COORDINATE_LIMIT = 100_000

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
    line_type: int = cv2.LINE_AA,
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
            colour, thickness, line_type,
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


def _int_box(xyxy: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    Turn one box into drawable integers, or ``None`` when it cannot be drawn.

    Model output does contain NaN and infinity now and then -- a division by zero,
    a tracker that diverged. OpenCV raises on those, so a single bad detection used
    to take the whole stream down; now that one is skipped and the rest are drawn.
    """
    if not np.all(np.isfinite(xyxy)):
        return None
    clipped = np.clip(xyxy, -COORDINATE_LIMIT, COORDINATE_LIMIT)
    return (int(clipped[0]), int(clipped[1]), int(clipped[2]), int(clipped[3]))


def _as_uint8(mask: np.ndarray) -> np.ndarray:
    """Boolean maskeyi OpenCV'nin beklediği uint8'e, mümkünse kopyalamadan çevirir."""
    if mask.dtype == np.bool_ and mask.flags["C_CONTIGUOUS"]:
        return mask.view(np.uint8)
    return np.ascontiguousarray(mask, dtype=np.uint8)


def _overlaps(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    """İki plaka dikdörtgeni kesişiyor mu."""
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


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
        line_type: int = cv2.LINE_AA,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.thickness = max(1, int(thickness))
        self.color_lookup = color_lookup
        self.accent_color = None if accent_color is None else resolve_palette(accent_color)
        #: Line type for curves. Anti-aliasing costs about four times as much
        #: per arc and is invisible on the dimmed glow pass, so :class:`cvflair.Theme`
        #: hands ``cv2.LINE_8`` to that one.
        self.line_type = line_type

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        for index in range(len(detections)):
            box = _int_box(detections.xyxy[index])
            if box is None:
                continue
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            accent = (
                resolve_color(self.accent_color, detections, index, self.color_lookup).as_bgr()
                if self.accent_color is not None
                else colour
            )
            self.draw(scene, box, colour, accent)
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
        _rounded_outline(
            scene, box, int(self.roundness * short_side / 2), colour, self.thickness, self.line_type
        )


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
                start_angle, start_angle + 90, accent, self.thickness, self.line_type,
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
        avoid_overlap: bool = True,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.text_color = resolve_palette(text_color).colors[0]
        #: Plakalar birbirini ezmesin diye boş bir yere kaydırılır.
        self.avoid_overlap = bool(avoid_overlap)
        self.text_scale = float(text_scale)
        self.text_thickness = max(1, int(text_thickness))
        self.text_padding = max(0, int(text_padding))
        self.border_radius = max(0, int(border_radius))
        self.color_lookup = color_lookup

    def annotate(
        self, scene: np.ndarray, detections: Any, labels: Any = None
    ) -> np.ndarray:
        texts = self._texts(detections, labels)
        frame_height, frame_width = scene.shape[:2]
        placed: list[tuple[int, int, int, int]] = []

        for index in range(len(detections)):
            box = _int_box(detections.xyxy[index])
            if box is None:
                continue
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            text = texts[index]

            (text_width, text_height), _ = cv2.getTextSize(
                text, FONT, self.text_scale, self.text_thickness
            )
            size = (text_width + self.text_padding * 2, text_height + self.text_padding * 2)
            plate, moved = self._place(box, size, placed, frame_width, frame_height)
            placed.append(plate)

            if moved:
                # Plaka kutusundan uzaklaştıysa hangi kutuya ait olduğu
                # anlaşılmıyor; ince bir işaretçi çizgisi bağlıyor.
                cv2.line(scene, (plate[0], plate[3]), (box[0], box[1]), colour, 1, cv2.LINE_AA)

            _rounded_fill(scene, plate, self.border_radius, colour)
            cv2.putText(
                scene,
                text,
                (plate[0] + self.text_padding, plate[3] - self.text_padding),
                FONT,
                self.text_scale,
                self.text_color.as_bgr(),
                self.text_thickness,
                cv2.LINE_AA,
            )
        return scene

    def _place(
        self,
        box: tuple[int, int, int, int],
        size: tuple[int, int],
        placed: list[tuple[int, int, int, int]],
        frame_width: int,
        frame_height: int,
    ) -> tuple[tuple[int, int, int, int], bool]:
        """
        Plaka için bir yer seçer.

        Varsayılan yer kutunun üstü. ``avoid_overlap`` açıkken daha önce
        yerleştirilmiş plakalarla çakışıyorsa sırayla başka yerler denenir;
        hiçbiri boş değilse varsayılana dönülür ve üst üste binmeye izin verilir.
        """
        x1, y1, x2, y2 = box
        width, height = size
        candidates = [(x1, y1 - height)]
        if self.avoid_overlap:
            candidates += [
                (x1, y1),                       # kutunun içinde, üstte
                (x1, y2),                       # kutunun altında
                (x2 - width, y1 - height),      # üstte, sağa hizalı
                (x1, y1 - 2 * height - 3),      # bir kat yukarı
                (x1, y2 + height + 3),          # bir kat aşağı
                (x1 - width - 3, y1),           # solda
                (x2 + 3, y1),                   # sağda
            ]

        def fit(left: float, top: float) -> tuple[int, int, int, int]:
            x = max(0, min(int(left), frame_width - width))
            y = max(0, min(int(top), frame_height - height))
            return (x, y, x + width, y + height)

        options = [fit(left, top) for left, top in candidates]
        for order, plate in enumerate(options):
            if not any(_overlaps(plate, other) for other in placed):
                return plate, order > 0
        return options[0], False  # her yer dolu: varsayılana dön, üst üste binsin

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


class ConfidenceBarAnnotator:
    """
    A thin bar under each box, filled in proportion to the detection's score.

    Detections without ``confidence`` are skipped -- there is nothing to show.
    """

    def __init__(
        self,
        color: Any = None,
        height: int = 4,
        gap: int = 3,
        background: Any = "#101418",
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.height = max(1, int(height))
        self.gap = max(0, int(gap))
        self.background = resolve_palette(background).colors[0]
        self.color_lookup = color_lookup

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        confidence = getattr(detections, "confidence", None)
        if confidence is None:
            return scene

        for index in range(len(detections)):
            box = _int_box(detections.xyxy[index])
            if box is None:
                continue
            score = float(confidence[index])
            if not np.isfinite(score):
                continue

            x1, _, x2, y2 = box
            width = x2 - x1
            if width <= 0:
                continue
            top = y2 + self.gap
            bottom = top + self.height

            cv2.rectangle(scene, (x1, top), (x2, bottom), self.background.as_bgr(), -1)
            filled = x1 + int(width * min(max(score, 0.0), 1.0))
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            cv2.rectangle(scene, (x1, top), (filled, bottom), colour, -1)
        return scene

    def __repr__(self) -> str:
        return f"ConfidenceBarAnnotator(height={self.height})"


class BlurAnnotator:
    """
    Hide what is inside each box: blur it, or drop it to blocks.

    Nothing is drawn on top -- the region itself is replaced, which is what
    face hiding needs. Boxes are clipped to the frame, so a detection hanging
    off the edge blurs the part that is visible.
    """

    def __init__(self, mode: str = "blur", strength: int = 15) -> None:
        if mode not in ("blur", "pixelate"):
            raise ValueError(f"Unknown mode {mode!r}. Use 'blur' or 'pixelate'.")
        self.mode = mode
        #: Blur radius, or the block size for pixelation. Bigger hides more.
        self.strength = max(1, int(strength))

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        height, width = scene.shape[:2]
        for index in range(len(detections)):
            box = _int_box(detections.xyxy[index])
            if box is None:
                continue
            x1 = max(0, min(box[0], box[2]))
            y1 = max(0, min(box[1], box[3]))
            x2 = min(width, max(box[0], box[2]))
            y2 = min(height, max(box[1], box[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            region = scene[y1:y2, x1:x2]
            scene[y1:y2, x1:x2] = (
                self._pixelate(region) if self.mode == "pixelate" else self._blur(region)
            )
        return scene

    def _blur(self, region: np.ndarray) -> np.ndarray:
        # Çekirdek tek sayı olmalı ve bölgeden büyük olmamalı.
        limit = min(region.shape[0], region.shape[1])
        size = min(self.strength * 2 + 1, limit if limit % 2 else limit - 1)
        return cv2.GaussianBlur(region, (max(3, size), max(3, size)), 0)

    def _pixelate(self, region: np.ndarray) -> np.ndarray:
        height, width = region.shape[:2]
        blocks = (max(1, width // self.strength), max(1, height // self.strength))
        small = cv2.resize(region, blocks, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)

    def __repr__(self) -> str:
        return f"BlurAnnotator(mode={self.mode!r}, strength={self.strength})"


class MaskAnnotator:
    """
    Segmentation masks: a tinted fill, an outline along the mask edge, or both.

    Nothing here computes a mask -- the model produces it and ``Detections.mask``
    carries it. Masks are drawn before the boxes so the outline stays crisp.
    """

    def __init__(
        self,
        color: Any = None,
        opacity: float = 0.4,
        outline: int = 2,
        color_lookup: ColorLookup = ColorLookup.CLASS,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.opacity = min(max(float(opacity), 0.0), 1.0)
        #: Outline weight along the mask edge; 0 draws only the tint.
        self.outline = max(0, int(outline))
        self.color_lookup = color_lookup
        #: Slack around the detection box when cropping the mask to a window.
        self.margin = 4

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        masks = getattr(detections, "mask", None)
        if masks is None or len(masks) == 0:
            return scene

        height, width = scene.shape[:2]
        # Her maske için tüm kareyi taramak pahalı: nesne kadrajın küçük bir
        # parçasını kaplıyor. Maskenin kendi sınırlayıcı kutusuna inip bütün işi
        # o pencerede yapmak aynı sonucu onlarca kat ucuza veriyor.
        windows = []
        for index in range(len(detections)):
            mask = np.asarray(masks[index], dtype=bool)
            if mask.shape != (height, width):
                continue  # başka boyuttaki maske çizilemez, sessizce atlanır
            bounds = self._window(mask, detections.xyxy[index], width, height)
            if bounds is None:
                continue
            left, top, right, bottom = bounds
            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            windows.append((mask[top:bottom, left:right], colour, left, top))

        if self.opacity > 0:
            for patch, colour, left, top in windows:
                region = scene[top : top + patch.shape[0], left : left + patch.shape[1]]
                # `region[patch] = ...` biçimindeki dizin atama bu işin en pahalı
                # yolu; harmanlamayı pencerenin tamamına yapıp maskeli kopyayı
                # OpenCV'ye bırakmak üç kat hızlı.
                block = np.full_like(region, colour)
                blended = cv2.addWeighted(region, 1 - self.opacity, block, self.opacity, 0)
                cv2.copyTo(blended, _as_uint8(patch), region)

        # Kontur harmanlamadan sonra: çizgi tam opaklıkta kalsın.
        if self.outline:
            for patch, colour, left, top in windows:
                edges, _ = cv2.findContours(
                    _as_uint8(patch), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    offset=(left, top),
                )
                cv2.drawContours(scene, edges, -1, colour, self.outline, cv2.LINE_AA)
        return scene

    def _window(
        self, mask: np.ndarray, xyxy: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """
        Pick the rectangle the mask is drawn in.

        The detection box is used first, widened by a small margin: it normally
        bounds the mask and costs nothing to read. If the mask still touches the
        edge of that window it may run further, and only then is the whole mask
        scanned -- the expensive path stays the exception.
        """
        box = _int_box(xyxy)
        if box is not None:
            left = max(0, min(box[0], box[2]) - self.margin)
            top = max(0, min(box[1], box[3]) - self.margin)
            right = min(width, max(box[0], box[2]) + self.margin)
            bottom = min(height, max(box[1], box[3]) + self.margin)
            patch = mask[top:bottom, left:right]
            if patch.size and not self._touches_edge(patch):
                return (left, top, right, bottom) if patch.any() else None

        rows, columns = np.any(mask, axis=1), np.any(mask, axis=0)
        if not rows.any():
            return None
        return (
            int(np.argmax(columns)),
            int(np.argmax(rows)),
            width - int(np.argmax(columns[::-1])),
            height - int(np.argmax(rows[::-1])),
        )

    @staticmethod
    def _touches_edge(patch: np.ndarray) -> bool:
        return bool(
            patch[0].any() or patch[-1].any() or patch[:, 0].any() or patch[:, -1].any()
        )

    def __repr__(self) -> str:
        return f"MaskAnnotator(opacity={self.opacity}, outline={self.outline})"


class ZoneAnnotator:
    """
    A polygon or a line drawn in the theme's colours.

    cvflair draws the region; deciding what falls inside it is the caller's
    business. ``fill_opacity`` blends a tint inside a closed polygon.
    """

    def __init__(
        self,
        color: Any = None,
        thickness: int = 2,
        fill_opacity: float = 0.0,
        closed: bool = True,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.thickness = max(1, int(thickness))
        self.fill_opacity = min(max(float(fill_opacity), 0.0), 1.0)
        self.closed = bool(closed)

    def annotate(self, scene: np.ndarray, points: Any, color_index: int = 0) -> np.ndarray:
        polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(polygon) < 2 or not np.all(np.isfinite(polygon)):
            return scene

        clipped = np.clip(polygon, -COORDINATE_LIMIT, COORDINATE_LIMIT).astype(np.int32)
        colour = self.color.by_index(color_index).as_bgr()

        if self.closed and self.fill_opacity > 0 and len(clipped) >= 3:
            overlay = scene.copy()
            cv2.fillPoly(overlay, [clipped], colour)
            cv2.addWeighted(overlay, self.fill_opacity, scene, 1 - self.fill_opacity, 0, scene)

        cv2.polylines(
            scene, [clipped], self.closed and len(clipped) >= 3, colour,
            self.thickness, cv2.LINE_AA,
        )
        return scene

    def __repr__(self) -> str:
        return f"ZoneAnnotator(thickness={self.thickness}, closed={self.closed})"


# -- key points -------------------------------------------------------------


class _SkeletonAnnotator:
    """
    Shared plumbing for the joint family: one colour per skeleton, points that
    are not finite or fall below ``min_confidence`` are skipped.

    Colour lookup follows the skeleton index, or ``class_id`` when the caller
    supplies one -- two hands in different colours, or every person in the same.
    """

    def __init__(
        self,
        color: Any = None,
        color_lookup: ColorLookup = ColorLookup.CLASS,
        min_confidence: float = 0.3,
    ) -> None:
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.color_lookup = color_lookup
        self.min_confidence = float(min_confidence)

    def _colour(self, keypoints: Any, index: int) -> tuple[int, int, int]:
        class_id = getattr(keypoints, "class_id", None)
        position = index if class_id is None or self.color_lookup is ColorLookup.INDEX else (
            int(class_id[index])
        )
        return self.color.by_index(position).as_bgr()

    def _points(self, keypoints: Any, index: int) -> list[tuple[int, int] | None]:
        """Drawable points for one skeleton; ``None`` where it cannot be drawn."""
        confidence = getattr(keypoints, "confidence", None)
        drawable: list[tuple[int, int] | None] = []
        for point_index, point in enumerate(keypoints.xy[index]):
            weak = (
                confidence is not None
                and float(confidence[index][point_index]) < self.min_confidence
            )
            if weak or not np.all(np.isfinite(point)):
                drawable.append(None)
                continue
            clipped = np.clip(point, -COORDINATE_LIMIT, COORDINATE_LIMIT)
            drawable.append((int(clipped[0]), int(clipped[1])))
        return drawable


class EdgeAnnotator(_SkeletonAnnotator):
    """The bones: a line for every connected pair the skeleton lists."""

    def __init__(self, *args: Any, thickness: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.thickness = max(1, int(thickness))

    def annotate(self, scene: np.ndarray, keypoints: Any, skeleton: Any) -> np.ndarray:
        for index in range(len(keypoints)):
            points = self._points(keypoints, index)
            colour = self._colour(keypoints, index)
            for first, second in skeleton:
                if first >= len(points) or second >= len(points):
                    continue  # skeleton describes more points than the model gave
                start, end = points[first], points[second]
                if start is None or end is None:
                    continue
                cv2.line(scene, start, end, colour, self.thickness, cv2.LINE_AA)
        return scene

    def __repr__(self) -> str:
        return f"EdgeAnnotator(thickness={self.thickness})"


class VertexAnnotator(_SkeletonAnnotator):
    """The joints: a filled dot on every drawable point."""

    def __init__(self, *args: Any, radius: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.radius = max(1, int(radius))

    def annotate(self, scene: np.ndarray, keypoints: Any, skeleton: Any = None) -> np.ndarray:
        for index in range(len(keypoints)):
            colour = self._colour(keypoints, index)
            for point in self._points(keypoints, index):
                if point is None:
                    continue
                cv2.circle(scene, point, self.radius, colour, -1, cv2.LINE_AA)
        return scene

    def __repr__(self) -> str:
        return f"VertexAnnotator(radius={self.radius})"


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
