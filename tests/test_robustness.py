"""
Bozuk kutular: model çıktısı her zaman temiz gelmiyor.

NaN, sonsuz ve int32'yi aşan koordinatlar OpenCV'de hata fırlatıyordu; tek bozuk
tespit bütün akışı düşürüyordu. Çizim artık böyle bir kutuyu atlıyor, diğerlerini
çizmeye devam ediyor.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import Detections, Theme, get_theme
from cvflair.annotators import COORDINATE_LIMIT
from cvflair.themes import BOX_STYLES

BROKEN = {
    "nan": [np.nan, 10, 100, 100],
    "inf": [10, 10, np.inf, 100],
    "-inf": [-np.inf, 10, 100, 100],
    "int32 üstü": [1e12, 10, 1e12 + 50, 100],
}
ODD = {
    "negatif": [-500, -500, 100, 100],
    "ters": [200, 200, 50, 50],
    "kadraj dışı": [5000, 5000, 5200, 5200],
    "sıfır alan": [40, 40, 40, 40],
}


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def single(xyxy) -> Detections:
    return Detections(xyxy=[xyxy], class_id=[0], confidence=[0.9], names=["nesne"])


@pytest.mark.parametrize("name", list(BROKEN))
@pytest.mark.parametrize("box_style", BOX_STYLES)
def test_broken_boxes_do_not_raise(name, box_style):
    Theme(box_style=box_style, hud=True).annotate(
        blank(), single(BROKEN[name]), stats={"FPS": 30}
    )


@pytest.mark.parametrize("name", list(ODD))
@pytest.mark.parametrize("box_style", BOX_STYLES)
def test_odd_but_finite_boxes_do_not_raise(name, box_style):
    Theme(box_style=box_style).annotate(blank(), single(ODD[name]))


@pytest.mark.parametrize("name", list(BROKEN))
def test_non_finite_boxes_are_skipped_not_drawn(name):
    frame = blank()

    get_theme("neon").annotate(frame, single(BROKEN[name]))

    if not np.all(np.isfinite(BROKEN[name])):
        assert painted(frame) == 0, "çizilemeyen kutu için ekrana bir şey düşmemeli"


def test_a_broken_box_does_not_hide_the_good_ones():
    frame = blank()
    detections = Detections(
        xyxy=[[np.nan, 0, 10, 10], [40, 40, 120, 100]],
        class_id=[0, 1],
        names=["bozuk", "saglam"],
    )

    get_theme("neon").annotate(frame, detections)

    assert painted(frame) > 0, "sağlam kutu da çizilmemiş"


def test_huge_coordinates_are_clipped_and_still_drawn():
    frame = blank()

    # Kadrajın içinden başlayıp çok uzağa uzanan kutu: kırpılıp çizilmeli.
    get_theme("minimal").annotate(frame, single([20, 20, 1e12, 1e12]))

    assert painted(frame) > 0


def test_clipping_limit_stays_inside_int32():
    assert COORDINATE_LIMIT < 2**31 - 1


def test_labels_survive_a_broken_box():
    """Etiket plakası da aynı kutuyu okuyor; orada da çökmemeli."""
    frame = blank()
    detections = Detections(
        xyxy=[[np.inf, np.inf, np.inf, np.inf], [30, 30, 100, 90]],
        class_id=[0, 1],
        names=["bozuk", "saglam"],
    )

    get_theme("pastel").annotate(frame, detections, labels=["bozuk 0.10", "saglam 0.99"])

    assert painted(frame) > 0
