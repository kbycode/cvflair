"""Etiket yerleştirme ve güven barı."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import ConfidenceBarAnnotator, Detections, Theme
from cvflair.annotators import LabelAnnotator, _overlaps

CROWD = Detections(
    xyxy=[[20, 40, 90, 100], [30, 44, 100, 104], [40, 48, 110, 108]],
    class_id=[0, 1, 2],
    confidence=[0.95, 0.70, 0.40],
    names=["bir", "iki", "uc"],
)


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def placements(avoid: bool) -> list[tuple[int, int, int, int]]:
    """Üç kutu için seçilen plaka dikdörtgenleri."""
    annotator = LabelAnnotator(text_padding=4, avoid_overlap=avoid)
    chosen: list[tuple[int, int, int, int]] = []
    for index in range(len(CROWD)):
        box = tuple(int(value) for value in CROWD.xyxy[index])
        plate, _ = annotator._place(box, (60, 20), chosen, FRAME_SHAPE[1], FRAME_SHAPE[0])
        chosen.append(plate)
    return chosen


# -- yerleştirme ------------------------------------------------------------


def test_overlap_helper():
    assert _overlaps((0, 0, 10, 10), (5, 5, 15, 15))
    assert not _overlaps((0, 0, 10, 10), (10, 0, 20, 10)), "bitişik olmak çakışma değil"
    assert not _overlaps((0, 0, 10, 10), (0, 10, 10, 20))


def test_plates_do_not_overlap_when_avoiding():
    chosen = placements(avoid=True)

    for first in range(len(chosen)):
        for second in range(first + 1, len(chosen)):
            assert not _overlaps(chosen[first], chosen[second]), (
                f"{chosen[first]} ile {chosen[second]} çakışıyor"
            )


def test_plates_do_overlap_without_avoiding():
    chosen = placements(avoid=False)

    assert any(
        _overlaps(chosen[i], chosen[j])
        for i in range(len(chosen))
        for j in range(i + 1, len(chosen))
    ), "kaydırma kapalıyken varsayılan yerler kullanılmalı"


def test_first_plate_keeps_the_default_slot():
    annotator = LabelAnnotator(avoid_overlap=True)
    box = (30, 60, 120, 140)

    plate, moved = annotator._place(box, (50, 18), [], FRAME_SHAPE[1], FRAME_SHAPE[0])

    assert moved is False
    assert plate == (30, 42, 80, 60), "boş sahnede plaka kutunun üstünde durmalı"


def test_plate_stays_inside_the_frame():
    annotator = LabelAnnotator(avoid_overlap=True)

    plate, _ = annotator._place((0, 0, 40, 30), (60, 20), [], FRAME_SHAPE[1], FRAME_SHAPE[0])

    assert plate[0] >= 0 and plate[1] >= 0
    assert plate[2] <= FRAME_SHAPE[1] and plate[3] <= FRAME_SHAPE[0]


def test_moved_plate_reports_itself():
    annotator = LabelAnnotator(avoid_overlap=True)
    taken = [(30, 42, 80, 60)]

    _, moved = annotator._place((30, 60, 120, 140), (50, 18), taken, FRAME_SHAPE[1], FRAME_SHAPE[0])

    assert moved is True, "dolu yerde kaydırma bildirilmeli"


def test_crowded_labels_cover_more_ground_when_avoiding():
    """Kaydırma açıkken plakalar üst üste binmediği için daha çok piksel boyanır."""
    stacked, spread = blank(), blank()

    Theme(avoid_label_overlap=False, text_padding=4).annotate(stacked, CROWD)
    Theme(avoid_label_overlap=True, text_padding=4).annotate(spread, CROWD)

    assert painted(spread) > painted(stacked)


# -- güven barı -------------------------------------------------------------


def test_bar_is_off_by_default():
    assert Theme()._confidence_bar_annotator is None
    assert Theme(confidence_bar=True)._confidence_bar_annotator is not None


def test_bar_draws_under_the_box():
    frame = blank()
    single = Detections(xyxy=[[20, 20, 100, 60]], class_id=[0], confidence=[1.0])

    ConfidenceBarAnnotator(color=["#FF0000"], height=4, gap=3).annotate(frame, single)

    under = frame[63:67, 20:100]
    assert under.any(), "bar kutunun altına çizilmeli"
    assert not frame[:60, :].any(), "kutunun içine taşmamalı"


def test_bar_length_follows_the_score():
    low, high = blank(), blank()
    annotator = ConfidenceBarAnnotator(color=["#FF0000"], background="#000000")

    annotator.annotate(low, Detections(xyxy=[[10, 10, 110, 40]], class_id=[0], confidence=[0.2]))
    annotator.annotate(high, Detections(xyxy=[[10, 10, 110, 40]], class_id=[0], confidence=[0.9]))

    red = np.array([0, 0, 255], dtype=np.uint8)
    filled = lambda frame: int(np.count_nonzero(np.all(frame == red, axis=2)))  # noqa: E731
    assert filled(high) > filled(low) * 3


def test_detections_without_confidence_get_no_bar():
    frame = blank()

    ConfidenceBarAnnotator().annotate(frame, Detections(xyxy=[[10, 10, 60, 40]], class_id=[0]))

    assert painted(frame) == 0


@pytest.mark.parametrize("score", [np.nan, np.inf])
def test_broken_scores_are_skipped(score):
    frame = blank()

    ConfidenceBarAnnotator().annotate(
        frame, Detections(xyxy=[[10, 10, 60, 40]], class_id=[0], confidence=[score])
    )

    assert painted(frame) == 0
