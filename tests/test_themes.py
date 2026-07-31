"""Theme tests: configuration, annotator reuse, and that pixels actually change."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import ColorPalette, Detections, Theme, available_themes, get_theme
from cvflair.annotators import (
    BoxAnnotator,
    BoxCornerAnnotator,
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    RoundBoxAnnotator,
    TargetBoxAnnotator,
)
from cvflair.themes import BOX_STYLES


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted_pixels(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def test_shipped_themes():
    assert available_themes() == ["cyberpunk", "hud", "minimal", "neon", "pastel"]


@pytest.mark.parametrize("name", available_themes())
def test_every_shipped_theme_draws(name, detections):
    frame = blank()

    get_theme(name).annotate(frame, detections)

    assert painted_pixels(frame) > 0


def test_get_theme_by_name():
    assert get_theme("minimal").name == "minimal"
    assert get_theme("  NEON ").name == "neon"


def test_get_theme_passes_instances_through():
    theme = Theme(name="custom")
    assert get_theme(theme) is theme


def test_get_theme_returns_independent_instances():
    assert get_theme("neon") is not get_theme("neon")


def test_unknown_theme_lists_the_options():
    with pytest.raises(ValueError, match="Available: cyberpunk, hud, minimal, neon, pastel"):
        get_theme("vaporwave")


def test_theme_rejects_non_string():
    with pytest.raises(TypeError):
        get_theme(3)


@pytest.mark.parametrize(
    ("box_style", "annotator_type"),
    [
        ("box", BoxAnnotator),
        ("round", RoundBoxAnnotator),
        ("corner", BoxCornerAnnotator),
        ("dashed", DashedBoxAnnotator),
        ("dashed_corner", DashedCornerAnnotator),
        ("bracket", BracketBoxAnnotator),
        ("crosshair", CrosshairAnnotator),
        ("target", TargetBoxAnnotator),
    ],
)
def test_box_style_selects_the_annotator(box_style, annotator_type):
    theme = Theme(box_style=box_style)
    assert isinstance(theme._box_annotator, annotator_type)


@pytest.mark.parametrize("box_style", BOX_STYLES)
def test_every_box_style_draws(box_style, detections):
    frame = blank()

    Theme(box_style=box_style, thickness=2).annotate(frame, detections)

    assert painted_pixels(frame) > 0


def test_accent_palette_reaches_the_annotator():
    accent = ColorPalette.from_hex(["#FFFFFF"])
    theme = Theme(box_style="target", accent_palette=accent)

    assert theme._box_annotator.accent_color is accent


def test_glow_dims_the_accent_too():
    theme = Theme(
        box_style="target",
        accent_palette=ColorPalette.from_hex(["#FFFFFF"]),
        glow=True,
        glow_dim=0.5,
    )

    dimmed = theme._glow_annotator.accent_color.colors[0]
    assert (dimmed.r, dimmed.g, dimmed.b) == (127, 127, 127)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"box_style": "triangle"},
        {"thickness": 0},
        {"roundness": 0.0},
        {"roundness": 1.5},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Theme(**kwargs)


def test_annotators_are_built_once_and_reused(detections):
    theme = get_theme("neon")
    box, label, glow = theme._box_annotator, theme._label_annotator, theme._glow_annotator

    theme.annotate(blank(), detections)
    theme.annotate(blank(), detections)

    assert theme._box_annotator is box
    assert theme._label_annotator is label
    assert theme._glow_annotator is glow


def test_annotate_draws_in_place(detections):
    theme = get_theme("minimal")
    frame = blank()

    result = theme.annotate(frame, detections)

    assert result is frame
    assert painted_pixels(frame) > 0


def test_empty_detections_leave_the_frame_alone():
    frame = blank()

    get_theme("neon").annotate(frame, Detections.empty())

    assert painted_pixels(frame) == 0


def test_neon_glow_covers_more_than_the_plain_box(detections):
    minimal_frame, neon_frame = blank(), blank()

    get_theme("minimal").annotate(minimal_frame, detections)
    get_theme("neon").annotate(neon_frame, detections)

    assert get_theme("neon")._glow_annotator is not None
    assert painted_pixels(neon_frame) > painted_pixels(minimal_frame)


def test_glow_pass_is_thicker_than_the_main_box():
    theme = Theme(glow=True, thickness=2, glow_thickness=6)

    assert theme._glow_annotator.thickness == 8
    assert theme._box_annotator.thickness == 2


def test_labels_can_be_switched_off(detections):
    theme = Theme(labels=False)
    assert theme._label_annotator is None

    frame = blank()
    theme.annotate(frame, detections)
    assert painted_pixels(frame) > 0


def test_custom_labels_are_passed_through(detections):
    theme = get_theme("minimal")
    with_labels, without = blank(), blank()

    theme.annotate(without, detections)
    theme.annotate(with_labels, detections, labels=["kedi", "kopek"])

    assert painted_pixels(with_labels) > 0
    assert not np.array_equal(with_labels, without), "label text was ignored"
