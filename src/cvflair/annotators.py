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
import time
from collections import deque
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
    "SketchBoxAnnotator",
    "PulseAnnotator",
    "TraceAnnotator",
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


class SketchBoxAnnotator(_OutlineAnnotator):
    """
    A rectangle that looks drawn by hand: wobbling lines, gone over twice.

    The wobble is seeded from the box itself, so the same box wobbles the same
    way on every frame -- a random jitter per frame would boil and look broken.
    """

    def __init__(self, *args: Any, wobble: float = 2.5, passes: int = 2, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.wobble = max(0.0, float(wobble))
        self.passes = max(1, int(passes))

    def draw(self, scene, box, colour, accent) -> None:
        x1, y1, x2, y2 = box
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return

        # Konum sekizer piksellik ızgaraya yuvarlanıyor: tespit kutusu kare kare
        # bir iki piksel oynadığında desen sabit kalıyor, ancak nesne gerçekten
        # yer değiştirdiğinde yenileniyor.
        grid = (abs(x1) // 8, abs(y1) // 8, abs(x2) // 8)
        seed = (grid[0] * 73856093) ^ (grid[1] * 19349663) ^ (grid[2] * 83492791)
        rng = np.random.default_rng(seed % (2**32))
        corners = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float64)

        # Kutunun bütün kenarları -- her ikinci geçiş dahil -- tek bir dizide
        # üretilip tek bir OpenCV çağrısıyla çiziliyor. Kenar başına ayrı hesap
        # ve ayrı çağrı aynı görüntüyü verir ama ölçümde maliyetin çoğu oraya
        # gidiyordu: diziler küçük, çağrı sayısı çok.
        starts = np.tile(corners, (self.passes, 1))
        ends = np.tile(np.roll(corners, -1, axis=0), (self.passes, 1))

        steps = int(min(max(max(x2 - x1, y2 - y1) // 18, 3), 16))
        ratios = np.arange(steps + 1, dtype=np.float64) / steps
        # Tek bir sapma sütunu hem x'e hem y'ye ekleniyor: kalem çizgiden
        # çapraz kayıyor, dik değil.
        drift = rng.uniform(-self.wobble, self.wobble, size=(len(starts), steps + 1, 1))
        drift[:, 0] = drift[:, -1] = 0.0  # uçlar köşelere çakılı

        spans = (ends - starts)[:, None, :]
        points = starts[:, None, :] + ratios[None, :, None] * spans + drift
        cv2.polylines(
            scene, list(points.round().astype(np.int32)), False,
            colour, self.thickness, cv2.LINE_AA,
        )


class PulseAnnotator(_OutlineAnnotator):
    """
    A ring that swells and fades around each box: the lock-on look.

    The phase comes from the clock, so the effect keeps moving even on a frozen
    frame. Pass ``moment`` to drive it yourself -- a GIF or a test wants the
    same frame to look the same every run.
    """

    def __init__(self, *args: Any, speed: float = 1.4, reach: int = 14, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.speed = float(speed)
        #: How far the ring travels away from the box at its widest.
        self.reach = max(1, int(reach))
        self._phase = 0.0

    def annotate(
        self, scene: np.ndarray, detections: Any, moment: float | None = None
    ) -> np.ndarray:
        clock = time.monotonic() if moment is None else float(moment)
        self._phase = (clock * self.speed) % 1.0
        return super().annotate(scene, detections)

    def draw(self, scene, box, colour, accent) -> None:
        grow = int(self.reach * self._phase)
        # Halka açıldıkça sönüyor: rengi arka plana doğru karartmak, kare kopyası
        # gerektiren gerçek saydamlıktan çok daha ucuz.
        fade = max(0.05, 1.0 - self._phase)
        faded = tuple(int(channel * fade) for channel in colour)
        cv2.rectangle(
            scene,
            (box[0] - grow, box[1] - grow),
            (box[2] + grow, box[3] + grow),
            faded,
            self.thickness,
        )


class TraceAnnotator:
    """
    The path each tracked object has taken.

    This is the one annotator that remembers anything: a short history of points
    per ``tracker_id``. Detections without tracker ids draw nothing -- cvflair
    does not track, it only draws what a tracker already decided.
    """

    def __init__(
        self,
        color: Any = None,
        thickness: int = 2,
        length: int = 32,
        anchor: str = "bottom",
        color_lookup: ColorLookup = ColorLookup.TRACK,
        forget_after: int = 30,
    ) -> None:
        if anchor not in ("bottom", "center"):
            raise ValueError(f"Unknown anchor {anchor!r}. Use 'bottom' or 'center'.")
        self.color = resolve_palette(color if color is not None else ColorPalette.DEFAULT)
        self.thickness = max(1, int(thickness))
        self.length = max(2, int(length))
        self.anchor = anchor
        self.color_lookup = color_lookup
        #: Kaç kare görünmeyen kimliğin izi unutulur.
        self.forget_after = max(1, int(forget_after))
        self._paths: dict[int, deque[tuple[int, int]]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame = 0

    def annotate(self, scene: np.ndarray, detections: Any) -> np.ndarray:
        tracker_id = getattr(detections, "tracker_id", None)
        if tracker_id is None:
            return scene

        self._frame += 1
        for index in range(len(detections)):
            box = _int_box(detections.xyxy[index])
            if box is None:
                continue
            identity = int(tracker_id[index])
            point = (
                (box[0] + box[2]) // 2,
                box[3] if self.anchor == "bottom" else (box[1] + box[3]) // 2,
            )
            path = self._paths.setdefault(identity, deque(maxlen=self.length))
            path.append(point)
            self._last_seen[identity] = self._frame

            colour = resolve_color(self.color, detections, index, self.color_lookup).as_bgr()
            self._draw_path(scene, path, colour)

        self._forget_stale()
        return scene

    #: İz kaç kademede inceltilip soldurulacak. Parça başına ayrı çizgi çekmek
    #: yumuşak bir geçiş verirdi ama her parça ayrı bir OpenCV çağrısı demek;
    #: birkaç kademe gözle aynı görünüp maliyeti kata böler.
    BANDS = 4

    def _draw_path(self, scene: np.ndarray, path: deque, colour: tuple[int, int, int]) -> None:
        points = np.array(path, dtype=np.int32)
        if len(points) < 2:
            return

        edges = np.linspace(0, len(points) - 1, self.BANDS + 1).astype(int)
        for band in range(self.BANDS):
            # Kademeler uçlarda bir nokta örtüşüyor, yoksa iz aralarından kopar.
            chunk = points[edges[band] : edges[band + 1] + 1]
            if len(chunk) < 2:  # kısa izde bazı kademelere nokta düşmez
                continue
            weight = (band + 1) / self.BANDS  # en yeni kademe en kalın
            faded = tuple(int(channel * (0.35 + 0.65 * weight)) for channel in colour)
            cv2.polylines(
                scene, [chunk], False, faded,
                max(1, int(round(self.thickness * weight))), cv2.LINE_AA,
            )

    def _forget_stale(self) -> None:
        stale = [
            identity
            for identity, seen in self._last_seen.items()
            if self._frame - seen > self.forget_after
        ]
        for identity in stale:
            self._paths.pop(identity, None)
            self._last_seen.pop(identity, None)

    def reset(self) -> None:
        """Bütün izleri unut -- kaynak değiştiğinde ya da akış baştan başladığında."""
        self._paths.clear()
        self._last_seen.clear()
        self._frame = 0

    def __repr__(self) -> str:
        return f"TraceAnnotator(length={self.length}, tracked={len(self._paths)})"


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


#: Aynı boyuttaki oval maskeler yeniden kullanılıyor: her karede yeniden
#: çizmek, gizlenen yüz sayısıyla çarpılan gereksiz bir maliyet.
_OVAL_CACHE: dict[tuple[int, int], np.ndarray] = {}
_OVAL_CACHE_LIMIT = 64


def _oval_mask(shape: tuple[int, int]) -> np.ndarray:
    """Dikdörtgene içten teğet, kenarı yumuşatılmış oval maske."""
    cached = _OVAL_CACHE.get(shape)
    if cached is not None:
        return cached

    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask, (width // 2, height // 2), (max(width // 2 - 1, 1), max(height // 2 - 1, 1)),
        0, 0, 360, 255, -1, cv2.LINE_AA,
    )
    if len(_OVAL_CACHE) >= _OVAL_CACHE_LIMIT:
        _OVAL_CACHE.clear()
    _OVAL_CACHE[shape] = mask
    return mask


class BlurAnnotator:
    """
    Hide what is inside each box: blur it, or drop it to blocks.

    Nothing is drawn on top -- the region itself is replaced, which is what
    face hiding needs. Boxes are clipped to the frame, so a detection hanging
    off the edge blurs the part that is visible.

    ``shape`` decides what gets replaced. A rectangle is the default; ``"ellipse"``
    hides an oval inscribed in the box, which sits better on faces and matches
    the rounded box styles.
    """

    def __init__(self, mode: str = "blur", strength: int = 15, shape: str = "box") -> None:
        if mode not in ("blur", "pixelate"):
            raise ValueError(f"Unknown mode {mode!r}. Use 'blur' or 'pixelate'.")
        if shape not in ("box", "ellipse"):
            raise ValueError(f"Unknown shape {shape!r}. Use 'box' or 'ellipse'.")
        self.mode = mode
        self.shape = shape
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
            hidden = self._pixelate(region) if self.mode == "pixelate" else self._blur(region)
            if self.shape == "ellipse":
                cv2.copyTo(hidden, _oval_mask(region.shape[:2]), region)
            else:
                scene[y1:y2, x1:x2] = hidden
        return scene

    def __repr__(self) -> str:
        return f"BlurAnnotator(mode={self.mode!r}, shape={self.shape!r}, strength={self.strength})"

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
