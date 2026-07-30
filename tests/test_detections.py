"""Detections and the colour helpers behind it."""

from __future__ import annotations

import numpy as np
import pytest

from cvflair import Color, ColorLookup, ColorPalette, Detections
from cvflair.colors import resolve_color, resolve_palette
from cvflair.detections import detection_names, is_detections


def test_xyxy_is_normalised():
    detections = Detections(xyxy=[[10, 20, 30, 40]])

    assert detections.xyxy.shape == (1, 4)
    assert detections.xyxy.dtype == np.float32
    assert len(detections) == 1


def test_empty_detections():
    detections = Detections.empty()

    assert len(detections) == 0
    assert detections.xyxy.shape == (0, 4)


def test_mismatched_field_lengths_are_rejected():
    with pytest.raises(ValueError, match="class_id has 2 entries but xyxy has 1 boxes"):
        Detections(xyxy=[[0, 0, 10, 10]], class_id=[0, 1])


def test_from_arrays_keeps_names():
    detections = Detections.from_arrays(
        xyxy=[[0, 0, 10, 10]], class_id=[3], confidence=[0.5], names=["kedi"]
    )

    assert list(detections.names) == ["kedi"]
    assert detection_names(detections).tolist() == ["kedi"]


def test_is_detections_accepts_anything_with_boxes():
    assert is_detections(Detections.empty())
    assert not is_detections("kutular")
    assert not is_detections(None)


def test_names_fall_back_to_the_data_dict():
    """This is where supervision keeps class names."""
    holder = Detections(xyxy=[[0, 0, 10, 10]])
    holder.data = {"class_name": np.array(["kisi"], dtype=object)}

    assert detection_names(holder).tolist() == ["kisi"]


# -- colours ----------------------------------------------------------------


def test_hex_colours_round_trip():
    assert Color.from_hex("#39FF14").as_hex() == "#39FF14"
    assert Color.from_hex("39ff14") == Color(57, 255, 20)
    assert Color.from_hex("#000000").as_bgr() == (0, 0, 0)


def test_invalid_colours_are_rejected():
    with pytest.raises(ValueError, match="Expected a colour"):
        Color.from_hex("mavi")
    with pytest.raises(ValueError, match="must be in 0-255"):
        Color(300, 0, 0)


def test_palette_accepts_the_short_forms():
    assert resolve_palette("#FF0000").colors == [Color(255, 0, 0)]
    assert len(resolve_palette(["#FF0000", "#00FF00"])) == 2
    assert resolve_palette(Color(1, 2, 3)).colors == [Color(1, 2, 3)]

    palette = ColorPalette.from_hex(["#FF0000"])
    assert resolve_palette(palette) is palette


def test_palette_accepts_a_foreign_palette():
    """Anything exposing r/g/b colours, e.g. supervision's ColorPalette."""

    class ForeignColor:
        r, g, b = 10, 20, 30

    class ForeignPalette:
        colors = [ForeignColor()]

    assert resolve_palette(ForeignPalette()).colors == [Color(10, 20, 30)]


def test_palette_cycles_by_index():
    palette = ColorPalette.from_hex(["#FF0000", "#00FF00"])

    assert palette.by_index(0) == Color(255, 0, 0)
    assert palette.by_index(3) == Color(0, 255, 0)


def test_colour_lookup_modes():
    palette = ColorPalette.from_hex(["#FF0000", "#00FF00", "#0000FF"])
    detections = Detections(
        xyxy=[[0, 0, 10, 10], [0, 0, 10, 10]], class_id=[2, 2], tracker_id=[0, 1]
    )

    assert resolve_color(palette, detections, 1, ColorLookup.CLASS) == Color(0, 0, 255)
    assert resolve_color(palette, detections, 1, ColorLookup.INDEX) == Color(0, 255, 0)
    assert resolve_color(palette, detections, 1, ColorLookup.TRACK) == Color(0, 255, 0)


def test_track_lookup_needs_tracker_ids():
    detections = Detections(xyxy=[[0, 0, 10, 10]], class_id=[0])

    with pytest.raises(ValueError, match="tracker_id"):
        resolve_color(ColorPalette.DEFAULT, detections, 0, ColorLookup.TRACK)


def test_class_lookup_falls_back_to_position():
    detections = Detections(xyxy=[[0, 0, 10, 10]])

    assert resolve_color(ColorPalette.DEFAULT, detections, 0, ColorLookup.CLASS) is not None


def test_dimming_darkens_every_channel():
    dimmed = ColorPalette.from_hex(["#FFFFFF"]).dim(0.5)

    assert dimmed.colors[0] == Color(127, 127, 127)
