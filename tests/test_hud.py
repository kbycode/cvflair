"""The stats panel: placement, blending and the numbers Camera feeds it."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE, make_frame

from cvflair import Detections, HudAnnotator, Theme, get_theme
from cvflair.annotators import HUD_POSITIONS

PANEL = 44  # kadraj köşesinden bakılacak kare boyutu


def blank(value: int = 0) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def corner(frame: np.ndarray, position: str) -> np.ndarray:
    rows = slice(0, PANEL) if position.startswith("top") else slice(-PANEL, None)
    cols = slice(0, PANEL) if position.endswith("left") else slice(-PANEL, None)
    return frame[rows, cols]


def test_panel_draws_something():
    frame = blank()

    HudAnnotator().annotate(frame, {"FPS": 30})

    assert frame.any(), "panel çizilmedi"


def test_no_stats_means_no_panel():
    frame = blank()

    HudAnnotator().annotate(frame, {})
    HudAnnotator().annotate(frame, None)

    assert not frame.any()


@pytest.mark.parametrize("position", HUD_POSITIONS)
def test_panel_lands_in_the_requested_corner(position):
    frame = blank()

    HudAnnotator(position=position, margin=4).annotate(frame, {"FPS": 30})

    assert corner(frame, position).any(), f"{position} köşesi boş"
    others = [name for name in HUD_POSITIONS if name != position]
    for other in others:
        if other[:3] != position[:3] and other[-4:] != position[-4:]:
            assert not corner(frame, other).any(), f"panel {other} köşesine taştı"


def test_unknown_position_is_rejected():
    with pytest.raises(ValueError, match="Unknown hud position"):
        HudAnnotator(position="middle")


def test_plate_is_blended_not_painted_over():
    """Arka plan yarı saydam kalmalı: panel altındaki sahne tamamen kaybolmamalı."""
    frame = blank(200)

    HudAnnotator(opacity=0.6, background="#000000").annotate(frame, {"FPS": 30})

    plate = frame[:PANEL, :PANEL]
    assert plate.min() > 0, "panel arkası tamamen örtülmüş"
    assert plate.max() < 200 or plate.mean() < 200, "harmanlama uygulanmamış"


def test_opacity_changes_how_much_shows_through():
    light, heavy = blank(200), blank(200)

    HudAnnotator(opacity=0.2, background="#000000").annotate(light, {"FPS": 30})
    HudAnnotator(opacity=0.9, background="#000000").annotate(heavy, {"FPS": 30})

    assert light[:PANEL, :PANEL].mean() > heavy[:PANEL, :PANEL].mean()


def test_more_lines_make_a_taller_panel():
    short, tall = blank(), blank()

    HudAnnotator().annotate(short, {"FPS": 30})
    HudAnnotator().annotate(tall, {"FPS": 30, "Objects": 4, "Skor": 12})

    assert np.count_nonzero(tall.any(axis=2)) > np.count_nonzero(short.any(axis=2))


# -- tema tarafı ------------------------------------------------------------


def test_hud_theme_is_shipped():
    theme = get_theme("hud")

    assert theme.hud is True
    assert theme._hud_annotator is not None


def test_themes_without_hud_have_no_panel():
    assert get_theme("neon")._hud_annotator is None


def test_theme_draws_the_panel_only_with_stats(detections):
    with_stats, without = blank(), blank()

    get_theme("hud").annotate(with_stats, detections, stats={"FPS": 30})
    get_theme("hud").annotate(without, detections)

    assert corner(with_stats, "top_left").any()
    assert not np.array_equal(with_stats, without)


def test_panel_shows_up_without_any_detection():
    frame = blank()

    get_theme("hud").annotate(frame, Detections.empty(), stats={"FPS": 30})

    assert corner(frame, "top_left").any(), "tespit yokken de panel çizilmeli"


def test_invalid_hud_position_is_rejected():
    with pytest.raises(ValueError, match="Unknown hud_position"):
        Theme(hud=True, hud_position="centre")


# -- kamera tarafı ----------------------------------------------------------


def test_measured_fps_starts_at_zero(camera_factory):
    camera, _ = camera_factory(frame_count=0)

    assert camera.measured_fps == 0.0


def test_measured_fps_reflects_the_consumed_rate(camera_factory):
    camera, _ = camera_factory(frame_count=12, delay=0.01)

    frames = list(camera.stream(timeout=2.0))

    assert frames, "akış boş"
    assert camera.measured_fps > 0


def test_camera_feeds_frame_rate_and_count_to_the_panel(camera_factory, detections):
    camera, _ = camera_factory(frame_count=1, theme="hud")
    camera._read_times.extend([0.0, 0.1])  # iki kare arası 0.1 sn -> 10 FPS

    stats = camera._hud_stats(detections, None)

    assert stats == {"FPS": "10", "Objects": 2}


def test_extra_stats_win_over_the_defaults(camera_factory, detections):
    camera, _ = camera_factory(frame_count=1, theme="hud")

    stats = camera._hud_stats(detections, {"Skor": 42, "Objects": 99})

    assert stats["Skor"] == 42
    assert stats["Objects"] == 99


def test_no_stats_are_built_for_themes_without_a_hud(camera_factory, detections):
    camera, _ = camera_factory(frame_count=1, theme="neon")

    assert camera._hud_stats(detections, {"Skor": 1}) is None


def test_annotate_draws_the_panel_through_the_camera(camera_factory, detections):
    camera, _ = camera_factory(frame_count=1, theme="hud")
    frame = make_frame(1)

    camera.annotate(frame, detections)

    assert corner(frame, "top_left").any()
