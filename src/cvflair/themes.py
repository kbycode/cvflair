"""
Visual themes -- a thin configuration layer over ``supervision`` annotators.

A :class:`Theme` holds nothing but presentation settings plus the annotator
instances built from them. Annotators are created once, when the theme is
constructed, and reused for every frame; rebuilding them inside the capture
loop is the most common avoidable cost in this kind of pipeline.

No drawing maths lives here. Every pixel is still produced by ``supervision``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import supervision as sv

from .annotators import (
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    TargetBoxAnnotator,
)

__all__ = ["Theme", "get_theme", "available_themes", "BoxStyle", "BOX_STYLES"]

BoxStyle = Literal["box", "round", "corner", "dashed", "bracket", "crosshair", "target"]

#: Every accepted ``box_style``. The first three come from ``supervision``, the
#: rest from :mod:`cvflair.annotators`.
BOX_STYLES: tuple[str, ...] = ("box", "round", "corner", "dashed", "bracket", "crosshair", "target")


def _dim(color: sv.Color, factor: float) -> sv.Color:
    """Return a darker copy of ``color`` (used for the neon halo pass)."""
    return sv.Color(
        r=int(color.r * factor),
        g=int(color.g * factor),
        b=int(color.b * factor),
    )


def _dim_palette(palette: sv.ColorPalette, factor: float) -> sv.ColorPalette:
    return sv.ColorPalette([_dim(color, factor) for color in palette.colors])


@dataclass
class Theme:
    """
    A named bundle of annotator settings.

    The defaults describe a plain white box with a plain label; the shipped
    presets (see :func:`get_theme`) are just different field values, which is
    also how a user-defined theme is written::

        Theme(name="my-theme", palette=sv.ColorPalette.from_hex(["#39FF14"]),
              box_style="round", thickness=3)

    Attributes are read at construction time only. Changing a field afterwards
    does not rebuild the annotators -- construct a new theme instead.
    """

    name: str = "custom"
    palette: sv.ColorPalette = field(default_factory=lambda: sv.ColorPalette.DEFAULT)
    #: Second colour for the parts meant to stand out: bracket elbows, the
    #: reticle centre, target corners. ``None`` keeps everything one colour.
    accent_palette: sv.ColorPalette | None = None
    box_style: BoxStyle = "box"
    thickness: int = 2
    #: Corner rounding for ``box_style="round"`` and ``"bracket"``, in (0, 1].
    roundness: float = 0.5
    #: Corner arm length in pixels for ``"corner"``, ``"bracket"`` and ``"target"``.
    corner_length: int = 20
    #: Dash geometry for ``box_style="dashed"``.
    dash_length: int = 12
    gap_length: int = 8
    #: Reticle geometry for ``box_style="crosshair"``.
    arm_length: int = 18
    center_size: int = 10
    #: Rectangle weight behind the corners of ``box_style="target"``.
    edge_thickness: int = 1
    #: Draw a dimmed, thicker box behind the main one to fake a glow.
    glow: bool = False
    glow_thickness: int = 5
    glow_dim: float = 0.45
    labels: bool = True
    text_color: sv.Color = field(default_factory=lambda: sv.Color.BLACK)
    text_scale: float = 0.5
    text_thickness: int = 1
    text_padding: int = 6
    label_radius: int = 0
    color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS

    def __post_init__(self) -> None:
        if self.box_style not in BOX_STYLES:
            raise ValueError(
                f"Unknown box_style {self.box_style!r}. Use one of: {', '.join(BOX_STYLES)}."
            )
        if self.thickness < 1:
            raise ValueError(f"thickness must be >= 1, got {self.thickness}.")
        if not 0.0 < self.roundness <= 1.0:
            raise ValueError(f"roundness must be in (0, 1], got {self.roundness}.")

        self._glow_annotator = (
            self._build_box_annotator(
                palette=_dim_palette(self.palette, self.glow_dim),
                thickness=self.thickness + self.glow_thickness,
                accent=(
                    _dim_palette(self.accent_palette, self.glow_dim)
                    if self.accent_palette is not None
                    else None
                ),
            )
            if self.glow
            else None
        )
        self._box_annotator = self._build_box_annotator(
            palette=self.palette,
            thickness=self.thickness,
            accent=self.accent_palette,
        )
        self._label_annotator = (
            sv.LabelAnnotator(
                color=self.palette,
                text_color=self.text_color,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                text_padding=self.text_padding,
                border_radius=self.label_radius,
                color_lookup=self.color_lookup,
            )
            if self.labels
            else None
        )

    def _build_box_annotator(
        self,
        palette: sv.ColorPalette,
        thickness: int,
        accent: sv.ColorPalette | None = None,
    ):
        common = {"color": palette, "thickness": thickness, "color_lookup": self.color_lookup}

        if self.box_style == "round":
            return sv.RoundBoxAnnotator(**common, roundness=self.roundness)
        if self.box_style == "corner":
            return sv.BoxCornerAnnotator(**common, corner_length=self.corner_length)
        if self.box_style == "dashed":
            return DashedBoxAnnotator(
                **common,
                accent_color=accent,
                dash_length=self.dash_length,
                gap_length=self.gap_length,
            )
        if self.box_style == "bracket":
            return BracketBoxAnnotator(
                **common,
                accent_color=accent,
                corner_length=self.corner_length,
                roundness=self.roundness,
            )
        if self.box_style == "crosshair":
            return CrosshairAnnotator(
                **common,
                accent_color=accent,
                arm_length=self.arm_length,
                center_size=self.center_size,
            )
        if self.box_style == "target":
            return TargetBoxAnnotator(
                **common,
                accent_color=accent,
                corner_length=self.corner_length,
                edge_thickness=self.edge_thickness,
            )
        return sv.BoxAnnotator(**common)

    def annotate(
        self,
        scene: np.ndarray,
        detections: sv.Detections,
        labels: Sequence[str] | None = None,
    ) -> np.ndarray:
        """
        Draw ``detections`` onto ``scene`` in place and return the same array.

        ``labels`` is passed straight through to ``supervision``: when it is
        ``None`` the label text falls back to the detections' ``class_name``
        data field, then to the class id.
        """
        if len(detections) == 0:
            return scene
        if self._glow_annotator is not None:
            self._glow_annotator.annotate(scene=scene, detections=detections)
        self._box_annotator.annotate(scene=scene, detections=detections)
        if self._label_annotator is not None:
            self._label_annotator.annotate(scene=scene, detections=detections, labels=labels)
        return scene


def _minimal() -> Theme:
    """Thin single-colour box, plain label. Reads well in a screen recording."""
    return Theme(
        name="minimal",
        palette=sv.ColorPalette([sv.Color.WHITE]),
        box_style="box",
        thickness=1,
        text_color=sv.Color.BLACK,
        text_scale=0.45,
        text_padding=4,
    )


def _neon() -> Theme:
    """Saturated per-class colours, rounded box, dimmed halo behind it."""
    return Theme(
        name="neon",
        palette=sv.ColorPalette.from_hex(
            ["#39FF14", "#FF00E5", "#00E5FF", "#FFE600", "#FF2D55", "#7B5CFF"]
        ),
        box_style="round",
        thickness=3,
        roundness=0.6,
        glow=True,
        glow_thickness=6,
        glow_dim=0.4,
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        text_padding=8,
        label_radius=6,
    )


def _pastel() -> Theme:
    """Soft tones, generous rounding, dark text. Easy on a projector."""
    return Theme(
        name="pastel",
        palette=sv.ColorPalette.from_hex(
            ["#FFADAD", "#A0C4FF", "#B9FBC0", "#FFD6A5", "#CDB4DB", "#9BF6FF"]
        ),
        box_style="round",
        thickness=2,
        roundness=0.8,
        text_color=sv.Color.from_hex("#2E2E2E"),
        text_scale=0.45,
        text_padding=8,
        label_radius=10,
    )


def _cyberpunk() -> Theme:
    """High contrast target lock: thin rectangle, heavy white corners."""
    return Theme(
        name="cyberpunk",
        palette=sv.ColorPalette.from_hex(["#00F0FF", "#FF206E", "#FFD400", "#8AFF00"]),
        accent_palette=sv.ColorPalette.from_hex(["#FFFFFF"]),
        box_style="target",
        thickness=3,
        corner_length=26,
        edge_thickness=1,
        text_color=sv.Color.BLACK,
        text_scale=0.45,
        text_padding=6,
    )


_THEMES: dict[str, Callable[[], Theme]] = {
    "minimal": _minimal,
    "neon": _neon,
    "pastel": _pastel,
    "cyberpunk": _cyberpunk,
}


def available_themes() -> list[str]:
    """Names accepted by :func:`get_theme` and by ``Camera(theme=...)``."""
    return sorted(_THEMES)


def get_theme(theme: str | Theme) -> Theme:
    """
    Resolve a theme name to a fresh :class:`Theme`; pass instances through.

    Each call builds a new instance, so two cameras never share annotators.
    """
    if isinstance(theme, Theme):
        return theme
    if not isinstance(theme, str):
        raise TypeError(f"theme must be a name or a Theme instance, got {type(theme).__name__}.")
    try:
        return _THEMES[theme.strip().lower()]()
    except KeyError:
        raise ValueError(
            f"Unknown theme {theme!r}. Available: {', '.join(available_themes())}."
        ) from None
