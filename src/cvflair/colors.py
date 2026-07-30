"""
Colours and palettes.

Deliberately small: a colour is three integers, a palette is a list of them,
and a lookup rule says which detection gets which colour. Everything that takes
a palette also accepts plain hex strings, so the everyday case never needs an
import::

    Theme(palette=["#39FF14", "#FF00E5"], text_color="#101010")
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

__all__ = ["Color", "ColorLookup", "ColorPalette", "resolve_color", "resolve_palette"]

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class Color:
    """An RGB colour. OpenCV wants BGR, which :meth:`as_bgr` hands over."""

    r: int
    g: int
    b: int

    # ClassVar keeps these out of the generated __init__.
    BLACK: ClassVar[Color]
    WHITE: ClassVar[Color]

    def __post_init__(self) -> None:
        for name in ("r", "g", "b"):
            value = getattr(self, name)
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be in 0-255, got {value}.")

    @classmethod
    def from_hex(cls, value: str) -> Color:
        match = _HEX.match(value.strip())
        if match is None:
            raise ValueError(f"Expected a colour like '#39FF14', got {value!r}.")
        digits = match.group(1)
        return cls(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))

    def as_bgr(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

    def as_hex(self) -> str:
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}"

    def dim(self, factor: float) -> Color:
        """A darker copy, used for glow halos."""
        return Color(int(self.r * factor), int(self.g * factor), int(self.b * factor))


Color.BLACK = Color(0, 0, 0)
Color.WHITE = Color(255, 255, 255)


class ColorLookup(str, Enum):
    """Which detection field decides the colour."""

    INDEX = "index"
    CLASS = "class"
    TRACK = "track"


class ColorPalette:
    """A cycle of colours; detections index into it."""

    DEFAULT: ColorPalette

    def __init__(self, colors: Sequence[Color]) -> None:
        if not colors:
            raise ValueError("A palette needs at least one colour.")
        self.colors = list(colors)

    @classmethod
    def from_hex(cls, values: Sequence[str]) -> ColorPalette:
        return cls([Color.from_hex(value) for value in values])

    def by_index(self, index: int) -> Color:
        return self.colors[int(index) % len(self.colors)]

    def dim(self, factor: float) -> ColorPalette:
        return ColorPalette([color.dim(factor) for color in self.colors])

    def __len__(self) -> int:
        return len(self.colors)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ColorPalette) and self.colors == other.colors

    def __repr__(self) -> str:
        return f"ColorPalette({[color.as_hex() for color in self.colors]})"


ColorPalette.DEFAULT = ColorPalette.from_hex(
    [
        "#A351FB", "#FF4040", "#FF7B33", "#FFB633", "#D1D435", "#4CFB12",
        "#94CF1A", "#40DE8A", "#1B9640", "#00D6C1", "#2E9CAA", "#00C4FF",
        "#364797", "#6675FF", "#0019EF", "#863AFF", "#530087", "#CD3AFF",
    ]
)


def resolve_palette(value: Any) -> ColorPalette:
    """
    Turn whatever was passed into a :class:`ColorPalette`.

    Accepts a palette, a single colour, a hex string, a sequence of either, and
    anything with a ``colors`` list of r/g/b objects -- which is how a
    ``supervision.ColorPalette`` comes through without an import.
    """
    if isinstance(value, ColorPalette):
        return value
    if isinstance(value, Color):
        return ColorPalette([value])
    if isinstance(value, str):
        return ColorPalette([Color.from_hex(value)])

    foreign = getattr(value, "colors", None)
    if foreign is not None:
        return ColorPalette([_as_color(color) for color in foreign])

    if isinstance(value, Sequence):
        if not value:
            raise ValueError("A palette needs at least one colour.")
        return ColorPalette([_as_color(item) for item in value])

    return ColorPalette([_as_color(value)])


def _as_color(value: Any) -> Color:
    if isinstance(value, Color):
        return value
    if isinstance(value, str):
        return Color.from_hex(value)
    # Duck type: supervision's Color, or any object carrying r/g/b.
    if all(hasattr(value, name) for name in ("r", "g", "b")):
        return Color(int(value.r), int(value.g), int(value.b))
    raise TypeError(f"Cannot read a colour from {value!r}.")


def resolve_color(
    palette: ColorPalette,
    detections: Any,
    detection_index: int,
    color_lookup: ColorLookup = ColorLookup.CLASS,
) -> Color:
    """Pick the colour for one detection, falling back to its position."""
    if color_lookup is ColorLookup.CLASS:
        class_id = getattr(detections, "class_id", None)
        index = detection_index if class_id is None else class_id[detection_index]
    elif color_lookup is ColorLookup.TRACK:
        tracker_id = getattr(detections, "tracker_id", None)
        if tracker_id is None:
            raise ValueError(
                "color_lookup=ColorLookup.TRACK needs detections carrying tracker_id."
            )
        index = tracker_id[detection_index]
    else:
        index = detection_index
    return palette.by_index(int(index))
