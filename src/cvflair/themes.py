"""
Visual themes -- the settings bundle behind every drawn frame.

A :class:`Theme` holds presentation settings plus the annotator instances built
from them. Annotators are created once, when the theme is constructed, and
reused for every frame; rebuilding them inside the capture loop is the most
common avoidable cost in this kind of pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from .annotators import (
    HUD_POSITIONS,
    BoxAnnotator,
    BoxCornerAnnotator,
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    EdgeAnnotator,
    HudAnnotator,
    LabelAnnotator,
    RoundBoxAnnotator,
    TargetBoxAnnotator,
    VertexAnnotator,
)
from .colors import Color, ColorLookup, ColorPalette, resolve_palette
from .keypoints import Skeleton, resolve_skeleton

__all__ = ["Theme", "get_theme", "available_themes", "BoxStyle", "BOX_STYLES"]

BoxStyle = Literal[
    "box", "round", "corner", "dashed", "dashed_corner", "bracket", "crosshair", "target"
]

#: Every accepted ``box_style``.
BOX_STYLES: tuple[str, ...] = (
    "box", "round", "corner", "dashed", "dashed_corner", "bracket", "crosshair", "target",
)


@dataclass
class Theme:
    """
    A named bundle of drawing settings.

    Colours are written the short way -- ``palette=["#39FF14", "#FF00E5"]``,
    ``text_color="#101010"`` -- but a :class:`~cvflair.colors.ColorPalette`, a
    single :class:`~cvflair.colors.Color`, or a ``supervision`` palette are all
    accepted too.

    Fields are read at construction time only. Changing one afterwards does not
    rebuild the annotators -- construct a new theme instead.
    """

    name: str = "custom"
    palette: Any = field(default_factory=lambda: ColorPalette.DEFAULT)
    #: Second colour for the parts meant to stand out: corner brackets, bracket
    #: elbows, the reticle centre. ``None`` keeps everything one colour.
    accent_palette: Any = None
    box_style: BoxStyle = "box"
    thickness: int = 2
    #: Corner rounding for ``"round"`` and ``"bracket"``, in (0, 1].
    roundness: float = 0.5
    #: Corner arm length in pixels for ``"corner"``, ``"dashed_corner"``,
    #: ``"bracket"`` and ``"target"``.
    corner_length: int = 20
    #: Dash geometry for ``"dashed"`` and ``"dashed_corner"``.
    dash_length: int = 12
    gap_length: int = 8
    #: Reticle geometry for ``"crosshair"``.
    arm_length: int = 18
    center_size: int = 10
    #: Rectangle weight behind the corners of ``"target"``.
    edge_thickness: int = 1
    #: Draw a dimmed, thicker outline behind the main one to fake a glow.
    glow: bool = False
    glow_thickness: int = 5
    glow_dim: float = 0.45
    labels: bool = True
    text_color: Any = field(default_factory=lambda: Color.BLACK)
    text_scale: float = 0.5
    text_thickness: int = 1
    text_padding: int = 6
    label_radius: int = 0
    color_lookup: ColorLookup = ColorLookup.CLASS
    #: Corner stats panel. When on, :meth:`annotate` draws whatever ``stats``
    #: it is handed; with no data there is no panel.
    hud: bool = False
    hud_position: str = "top_left"
    hud_opacity: float = 0.6
    #: Skeleton drawing: bone weight, joint size, and the confidence a point
    #: needs before it is drawn at all.
    pose_thickness: int = 2
    pose_radius: int = 3
    pose_confidence: float = 0.3

    def __post_init__(self) -> None:
        if self.box_style not in BOX_STYLES:
            raise ValueError(
                f"Unknown box_style {self.box_style!r}. Use one of: {', '.join(BOX_STYLES)}."
            )
        if self.thickness < 1:
            raise ValueError(f"thickness must be >= 1, got {self.thickness}.")
        if not 0.0 < self.roundness <= 1.0:
            raise ValueError(f"roundness must be in (0, 1], got {self.roundness}.")
        if self.hud_position not in HUD_POSITIONS:
            raise ValueError(
                f"Unknown hud_position {self.hud_position!r}. "
                f"Use one of: {', '.join(HUD_POSITIONS)}."
            )

        self.palette = resolve_palette(self.palette)
        if self.accent_palette is not None:
            self.accent_palette = resolve_palette(self.accent_palette)
        self.text_color = resolve_palette(self.text_color).colors[0]

        self._glow_annotator = (
            self._build_box_annotator(
                palette=self.palette.dim(self.glow_dim),
                thickness=self.thickness + self.glow_thickness,
                accent=(
                    self.accent_palette.dim(self.glow_dim)
                    if self.accent_palette is not None
                    else None
                ),
                # The halo is dim and sits behind the main outline, so its
                # anti-aliasing never shows -- and it costs four times as much
                # per arc.
                line_type=cv2.LINE_8,
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
            LabelAnnotator(
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
        self._edge_annotator = EdgeAnnotator(
            color=self.palette,
            color_lookup=self.color_lookup,
            thickness=self.pose_thickness,
            min_confidence=self.pose_confidence,
        )
        self._vertex_annotator = VertexAnnotator(
            color=self.accent_palette if self.accent_palette is not None else self.palette,
            color_lookup=self.color_lookup,
            radius=self.pose_radius,
            min_confidence=self.pose_confidence,
        )
        self._glow_edge_annotator = (
            EdgeAnnotator(
                color=self.palette.dim(self.glow_dim),
                color_lookup=self.color_lookup,
                thickness=self.pose_thickness + self.glow_thickness,
                min_confidence=self.pose_confidence,
            )
            if self.glow
            else None
        )
        self._hud_annotator = (
            HudAnnotator(
                color=self.palette,
                text_scale=self.text_scale,
                text_thickness=self.text_thickness,
                position=self.hud_position,
                opacity=self.hud_opacity,
            )
            if self.hud
            else None
        )

    def _build_box_annotator(
        self,
        palette: ColorPalette,
        thickness: int,
        accent: ColorPalette | None = None,
        line_type: int = cv2.LINE_AA,
    ):
        common: dict[str, Any] = {
            "color": palette,
            "thickness": thickness,
            "color_lookup": self.color_lookup,
            "line_type": line_type,
        }

        if self.box_style == "round":
            return RoundBoxAnnotator(**common, roundness=self.roundness)
        if self.box_style == "corner":
            return BoxCornerAnnotator(**common, corner_length=self.corner_length)
        if self.box_style == "dashed":
            return DashedBoxAnnotator(
                **common,
                accent_color=accent,
                dash_length=self.dash_length,
                gap_length=self.gap_length,
            )
        if self.box_style == "dashed_corner":
            return DashedCornerAnnotator(
                **common,
                accent_color=accent,
                corner_length=self.corner_length,
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
        return BoxAnnotator(**common)

    def annotate(
        self,
        scene: np.ndarray,
        detections: Any,
        labels: Sequence[str] | None = None,
        stats: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Draw ``detections`` onto ``scene`` in place and return the same array.

        ``detections`` is cvflair's :class:`~cvflair.detections.Detections` or
        anything carrying the same fields. ``stats`` feeds the HUD panel when
        the theme has one -- ``{"FPS": 30, "Objects": 3}`` and so on.
        """
        if len(detections):
            if self._glow_annotator is not None:
                self._glow_annotator.annotate(scene, detections)
            self._box_annotator.annotate(scene, detections)
            if self._label_annotator is not None:
                self._label_annotator.annotate(scene, detections, labels=labels)

        if self._hud_annotator is not None and stats:
            self._hud_annotator.annotate(scene, stats)
        return scene

    def annotate_keypoints(
        self,
        scene: np.ndarray,
        keypoints: Any,
        skeleton: Skeleton | str = "hand",
    ) -> np.ndarray:
        """
        Draw skeletons onto ``scene`` in place and return the same array.

        ``skeleton`` is a name (``"hand"``, ``"pose"``) or a list of index pairs.
        Bones take the palette colour, joints take the accent when the theme has
        one, and ``glow`` puts a dimmed thicker pass behind the bones.
        """
        if len(keypoints) == 0:
            return scene
        wiring = resolve_skeleton(skeleton)

        if self._glow_edge_annotator is not None:
            self._glow_edge_annotator.annotate(scene, keypoints, wiring)
        self._edge_annotator.annotate(scene, keypoints, wiring)
        self._vertex_annotator.annotate(scene, keypoints)
        return scene


def _minimal() -> Theme:
    """Thin single-colour box, plain label. Reads well in a screen recording."""
    return Theme(
        name="minimal",
        palette=["#FFFFFF"],
        box_style="box",
        thickness=1,
        text_color="#000000",
        text_scale=0.45,
        text_padding=4,
    )


def _neon() -> Theme:
    """Saturated per-class colours, rounded box, dimmed halo behind it."""
    return Theme(
        name="neon",
        palette=["#39FF14", "#FF00E5", "#00E5FF", "#FFE600", "#FF2D55", "#7B5CFF"],
        box_style="round",
        thickness=3,
        roundness=0.6,
        glow=True,
        glow_thickness=6,
        glow_dim=0.4,
        text_color="#000000",
        text_scale=0.5,
        text_padding=8,
        label_radius=6,
    )


def _pastel() -> Theme:
    """Soft tones, generous rounding, dark text. Easy on a projector."""
    return Theme(
        name="pastel",
        palette=["#FFADAD", "#A0C4FF", "#B9FBC0", "#FFD6A5", "#CDB4DB", "#9BF6FF"],
        box_style="round",
        thickness=2,
        roundness=0.8,
        text_color="#2E2E2E",
        text_scale=0.45,
        text_padding=8,
        label_radius=10,
    )


def _cyberpunk() -> Theme:
    """High contrast target lock: thin rectangle, heavy white corners."""
    return Theme(
        name="cyberpunk",
        palette=["#00F0FF", "#FF206E", "#FFD400", "#8AFF00"],
        accent_palette="#FFFFFF",
        box_style="target",
        thickness=3,
        corner_length=26,
        edge_thickness=1,
        text_color="#000000",
        text_scale=0.45,
        text_padding=6,
    )


def _hud() -> Theme:
    """Thin corner marks plus a stats panel: made for game and robotics demos."""
    return Theme(
        name="hud",
        palette=["#37E8B0", "#FFC53D", "#FF5C7A", "#5AA9FF"],
        box_style="corner",
        thickness=2,
        corner_length=18,
        text_color="#06231B",
        text_scale=0.45,
        text_padding=5,
        hud=True,
        hud_position="top_left",
        hud_opacity=0.6,
    )


_THEMES: dict[str, Callable[[], Theme]] = {
    "minimal": _minimal,
    "neon": _neon,
    "pastel": _pastel,
    "cyberpunk": _cyberpunk,
    "hud": _hud,
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
