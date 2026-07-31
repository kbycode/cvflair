"""Kutu içini gizleme ve bölge çizimi."""

from __future__ import annotations

import numpy as np
import pytest

from cvflair import BlurAnnotator, Detections, Theme, ZoneAnnotator

BOX = [30, 30, 130, 110]


def noisy(seed: int = 0) -> np.ndarray:
    """Bulanıklığın ölçülebilmesi için desenli bir kare."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(160, 200, 3), dtype=np.uint8)


def detail(region: np.ndarray) -> float:
    """Komşu pikseller arası fark: bulanıklaştıkça düşer."""
    return float(np.abs(np.diff(region.astype(np.int16), axis=1)).mean())


def single() -> Detections:
    return Detections(xyxy=[BOX], class_id=[0], confidence=[0.9], names=["kisi"])


# -- gizleme ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_region_loses_detail(mode):
    frame = noisy()
    before = detail(frame[30:110, 30:130])

    BlurAnnotator(mode=mode, strength=12).annotate(frame, single())

    assert detail(frame[30:110, 30:130]) < before / 2


@pytest.mark.parametrize("mode", ["blur", "pixelate"])
def test_outside_the_box_is_untouched(mode):
    frame = noisy()
    original = frame.copy()

    BlurAnnotator(mode=mode, strength=12).annotate(frame, single())

    assert np.array_equal(frame[:25, :], original[:25, :])
    assert np.array_equal(frame[:, 150:], original[:, 150:])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="Use 'blur' or 'pixelate'"):
        BlurAnnotator(mode="karart")


def test_tiny_and_broken_boxes_are_skipped():
    frame = noisy()
    original = frame.copy()
    odd = Detections(
        xyxy=[[40, 40, 41, 41], [np.nan, 0, 50, 50]], class_id=[0, 0], confidence=[0.9, 0.9]
    )

    BlurAnnotator().annotate(frame, odd)

    assert np.array_equal(frame, original)


def test_box_hanging_off_the_edge_is_clipped():
    frame = noisy()
    outside = Detections(xyxy=[[-60, -60, 60, 60]], class_id=[0], confidence=[0.9])

    BlurAnnotator(strength=8).annotate(frame, outside)  # kırpılmazsa OpenCV hata verir

    assert detail(frame[0:60, 0:60]) < detail(noisy()[0:60, 0:60])


def test_theme_hides_under_the_box_not_over_it():
    """Gizleme çerçeveden önce uygulanmalı: kutu çizgisi net kalsın."""
    frame = noisy()

    Theme(hide="blur", palette=["#FF0000"], thickness=3).annotate(frame, single())

    red = np.array([0, 0, 255], dtype=np.uint8)
    drawn = int(np.count_nonzero(np.all(frame == red, axis=2)))
    assert drawn > 0, "çerçeve bulanıklığın altında kalmış"


def test_hiding_is_off_by_default():
    assert Theme()._blur_annotator is None
    assert Theme(hide="pixelate")._blur_annotator is not None


# -- bölge ------------------------------------------------------------------


def blank() -> np.ndarray:
    return np.zeros((160, 200, 3), dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


TRIANGLE = [(20, 20), (150, 40), (90, 140)]


def test_polygon_is_drawn():
    frame = blank()

    ZoneAnnotator(color=["#00FF00"], thickness=2).annotate(frame, TRIANGLE)

    assert painted(frame) > 0


def test_fill_blends_instead_of_covering():
    frame = blank()
    frame[:] = 200

    ZoneAnnotator(color=["#000000"], fill_opacity=0.5).annotate(frame, TRIANGLE)

    inside = frame[70, 90]
    assert 0 < int(inside.max()) < 200, "dolgu harmanlanmalı, örtmemeli"


def test_open_line_does_not_close_the_shape():
    closed, open_line = blank(), blank()

    ZoneAnnotator(color=["#00FF00"]).annotate(closed, TRIANGLE)
    ZoneAnnotator(color=["#00FF00"], closed=False).annotate(open_line, TRIANGLE)

    assert painted(closed) > painted(open_line), "kapalı şekil bir kenar fazla çizmeli"


def test_too_few_or_broken_points_draw_nothing():
    frame = blank()

    ZoneAnnotator().annotate(frame, [(10, 10)])
    ZoneAnnotator().annotate(frame, [(10, 10), (np.nan, 40), (60, 60)])

    assert painted(frame) == 0


def test_theme_draws_zones_with_its_palette():
    frame = blank()

    Theme(palette=["#FF0000"]).annotate_zone(frame, TRIANGLE, fill_opacity=0.3)

    assert painted(frame) > 0
