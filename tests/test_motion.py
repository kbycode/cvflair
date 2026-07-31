"""Eskiz çerçevesi, nabız halkası ve takip izi."""

from __future__ import annotations

import numpy as np
import pytest

from cvflair import Detections, PulseAnnotator, Theme, TraceAnnotator

FRAME = (200, 260, 3)


def blank() -> np.ndarray:
    return np.zeros(FRAME, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def moving(step: int, identity: int = 7) -> Detections:
    """Kare kare sağa kayan tek bir takip edilen kutu."""
    left = 20 + step * 12
    return Detections(
        xyxy=[[left, 60, left + 40, 120]],
        class_id=[0],
        confidence=[0.9],
        tracker_id=[identity],
    )


# -- eskiz ------------------------------------------------------------------


def test_sketch_draws_a_frame():
    frame = blank()

    Theme(box_style="sketch", labels=False, palette=["#00FF00"]).annotate(
        frame, Detections(xyxy=[[40, 40, 180, 150]], class_id=[0])
    )

    assert painted(frame) > 0


def test_sketch_wobble_is_stable_across_frames():
    """Kare kare değişen bir titreşim ekranda kaynar; aynı kutu aynı çizilmeli."""
    first, second = blank(), blank()
    theme = Theme(box_style="sketch", labels=False)
    boxes = Detections(xyxy=[[40, 40, 180, 150]], class_id=[0])

    theme.annotate(first, boxes)
    theme.annotate(second, boxes)

    assert np.array_equal(first, second)


def test_different_boxes_wobble_differently():
    theme = Theme(box_style="sketch", labels=False, palette=["#00FF00"])
    left, right = blank(), blank()

    theme.annotate(left, Detections(xyxy=[[10, 40, 150, 150]], class_id=[0]))
    theme.annotate(right, Detections(xyxy=[[60, 40, 200, 150]], class_id=[0]))

    shifted = np.roll(right, -50, axis=1)
    assert not np.array_equal(left[:, :150], shifted[:, :150])


def spread_of_the_top_edge(wobble: float) -> int:
    """Üst kenarın kaç ayrı satıra yayıldığı: sapmanın doğrudan ölçüsü."""
    frame = blank()
    Theme(box_style="sketch", labels=False, thickness=1, wobble=wobble).annotate(
        frame, Detections(xyxy=[[40, 80, 200, 160]], class_id=[0])
    )
    band = frame[70:92, 60:180].any(axis=2)
    return int(np.count_nonzero(band.any(axis=1)))


def test_sketch_line_leaves_the_straight_path():
    """Kenar yumuşatma düz çizgiyi de birkaç satıra yayar; ölçüt aradaki fark."""
    assert spread_of_the_top_edge(wobble=4) > spread_of_the_top_edge(wobble=0) * 2


def test_wobble_zero_is_accepted():
    frame = blank()

    Theme(box_style="sketch", labels=False, wobble=0).annotate(
        frame, Detections(xyxy=[[40, 40, 180, 150]], class_id=[0])
    )

    assert painted(frame) > 0


def test_a_box_jittering_by_a_pixel_keeps_its_pattern():
    """Tespit kutusu her karede bir iki piksel oynar; desen bununla kaynamamalı."""
    theme = Theme(box_style="sketch", labels=False)
    steady, nudged = blank(), blank()

    theme.annotate(steady, Detections(xyxy=[[40, 40, 180, 150]], class_id=[0]))
    theme.annotate(nudged, Detections(xyxy=[[41, 41, 181, 151]], class_id=[0]))

    assert np.array_equal(steady[:-1, :-1], nudged[1:, 1:])


# -- nabız ------------------------------------------------------------------


def test_pulse_is_off_by_default():
    assert Theme()._pulse_annotator is None
    assert Theme(pulse=True)._pulse_annotator is not None


def test_ring_widens_with_time():
    """Faz ilerledikçe halka kutudan uzaklaşır."""
    early, late = blank(), blank()
    annotator = PulseAnnotator(color=["#00FF00"], reach=20, speed=1.0)
    boxes = Detections(xyxy=[[80, 70, 160, 130]], class_id=[0])

    annotator.annotate(early, boxes, moment=0.05)
    annotator.annotate(late, boxes, moment=0.8)

    reach = lambda frame: int(np.flatnonzero(frame.any(axis=(0, 2))).min())  # noqa: E731
    assert reach(late) < reach(early)


def test_ring_fades_as_it_grows():
    early, late = blank(), blank()
    annotator = PulseAnnotator(color=["#00FF00"], reach=20, speed=1.0)
    boxes = Detections(xyxy=[[80, 70, 160, 130]], class_id=[0])

    annotator.annotate(early, boxes, moment=0.05)
    annotator.annotate(late, boxes, moment=0.9)

    assert int(late.max()) < int(early.max())


def test_pulse_keeps_moving_without_a_moment():
    annotator = PulseAnnotator(color=["#00FF00"], speed=1.0)
    boxes = Detections(xyxy=[[80, 70, 160, 130]], class_id=[0])

    annotator.annotate(blank(), boxes)
    first = annotator._phase
    annotator.annotate(blank(), boxes, moment=first + 0.4)

    assert annotator._phase != first, "saat verilmediğinde faz kendi ilerlemeli"


def test_theme_passes_the_moment_through():
    early, late = blank(), blank()
    theme = Theme(pulse=True, labels=False, palette=["#00FF00"], pulse_reach=20)
    boxes = Detections(xyxy=[[80, 70, 160, 130]], class_id=[0])

    theme.annotate(early, boxes, moment=0.05)
    theme.annotate(late, boxes, moment=0.7)

    assert painted(late) != painted(early)


def test_ring_reaching_past_the_frame_does_not_crash():
    frame = blank()

    PulseAnnotator(color=["#00FF00"], reach=400).annotate(
        frame, Detections(xyxy=[[0, 0, 30, 30]], class_id=[0]), moment=0.95
    )


# -- iz ---------------------------------------------------------------------


def test_trace_is_off_by_default():
    assert Theme()._trace_annotator is None
    assert Theme(trace=True)._trace_annotator is not None


def test_path_appears_only_after_a_second_frame():
    annotator = TraceAnnotator(color=["#00FF00"])
    first, second = blank(), blank()

    annotator.annotate(first, moving(0))
    annotator.annotate(second, moving(1))

    assert painted(first) == 0, "tek noktadan çizgi çıkmaz"
    assert painted(second) > 0


def test_path_grows_with_the_object():
    annotator = TraceAnnotator(color=["#00FF00"])
    lengths = []
    for step in range(5):
        frame = blank()
        annotator.annotate(frame, moving(step))
        lengths.append(painted(frame))

    assert lengths[4] > lengths[2] > lengths[1]


def test_path_is_capped_at_its_length():
    annotator = TraceAnnotator(length=4)

    for step in range(20):
        annotator.annotate(blank(), moving(step))

    assert len(annotator._paths[7]) == 4


def test_detections_without_tracker_ids_draw_nothing():
    frame = blank()
    annotator = TraceAnnotator()

    for _ in range(4):
        annotator.annotate(frame, Detections(xyxy=[[20, 60, 60, 120]], class_id=[0]))

    assert painted(frame) == 0
    assert annotator._paths == {}


def test_each_track_keeps_its_own_path():
    annotator = TraceAnnotator()

    for step in range(3):
        annotator.annotate(blank(), moving(step, identity=1))
        annotator.annotate(blank(), moving(step, identity=2))

    assert set(annotator._paths) == {1, 2}


def test_lost_tracks_are_forgotten():
    annotator = TraceAnnotator(forget_after=3)

    for step in range(3):
        annotator.annotate(blank(), moving(step, identity=1))
    for step in range(10):  # 1 numara artık görünmüyor
        annotator.annotate(blank(), moving(step, identity=2))

    assert set(annotator._paths) == {2}, "kaybolan kimlik hafızada kalmamalı"


@pytest.mark.parametrize("anchor", ["bottom", "center"])
def test_anchor_picks_where_the_path_is_drawn(anchor):
    annotator = TraceAnnotator(color=["#00FF00"], anchor=anchor)
    frame = blank()

    annotator.annotate(blank(), moving(0))
    annotator.annotate(frame, moving(1))

    rows = np.flatnonzero(frame.any(axis=(1, 2)))
    assert (rows.mean() > 110) if anchor == "bottom" else (80 < rows.mean() < 100)


def test_unknown_anchor_is_rejected():
    with pytest.raises(ValueError, match="Use 'bottom' or 'center'"):
        TraceAnnotator(anchor="ust")


def test_reset_clears_the_history():
    theme = Theme(trace=True, labels=False)

    for step in range(4):
        theme.annotate(blank(), moving(step))
    theme.reset_trace()

    assert theme._trace_annotator._paths == {}


def test_broken_boxes_do_not_enter_the_path():
    annotator = TraceAnnotator()
    broken = Detections(xyxy=[[np.nan, 60, 60, 120]], class_id=[0], tracker_id=[7])

    annotator.annotate(blank(), moving(0))
    annotator.annotate(blank(), broken)

    assert len(annotator._paths[7]) == 1
