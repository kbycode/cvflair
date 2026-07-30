"""Tests for the outline annotators: geometry, colours and edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import ColorPalette, Detections
from cvflair.annotators import (
    BoxAnnotator,
    BracketBoxAnnotator,
    CrosshairAnnotator,
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    TargetBoxAnnotator,
)

ANNOTATORS = [
    DashedBoxAnnotator,
    DashedCornerAnnotator,
    BracketBoxAnnotator,
    CrosshairAnnotator,
    TargetBoxAnnotator,
]

RED = ColorPalette.from_hex(["#FF0000"])
BLUE = ColorPalette.from_hex(["#0000FF"])


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def colour_pixels(frame: np.ndarray, bgr: tuple[int, int, int]) -> int:
    return int(np.count_nonzero(np.all(frame == np.array(bgr, dtype=np.uint8), axis=2)))


@pytest.mark.parametrize("annotator_type", ANNOTATORS)
def test_annotator_draws_in_place(annotator_type, detections):
    frame = blank()

    result = annotator_type(color=RED).annotate(scene=frame, detections=detections)

    assert result is frame
    assert painted(frame) > 0


@pytest.mark.parametrize("annotator_type", ANNOTATORS)
def test_empty_detections_draw_nothing(annotator_type):
    frame = blank()

    annotator_type(color=RED).annotate(scene=frame, detections=Detections.empty())

    assert painted(frame) == 0


@pytest.mark.parametrize("annotator_type", ANNOTATORS)
def test_thicker_outline_covers_more(annotator_type, detections):
    thin, thick = blank(), blank()

    annotator_type(color=RED, thickness=1).annotate(scene=thin, detections=detections)
    annotator_type(color=RED, thickness=5).annotate(scene=thick, detections=detections)

    assert painted(thick) > painted(thin)


@pytest.mark.parametrize(
    "annotator_type",
    [DashedCornerAnnotator, BracketBoxAnnotator, CrosshairAnnotator, TargetBoxAnnotator],
)
def test_accent_colour_is_used(annotator_type, detections):
    frame = blank()

    annotator_type(color=RED, accent_color=BLUE, thickness=3).annotate(
        scene=frame, detections=detections
    )

    assert colour_pixels(frame, (0, 0, 255)) > 0, "detection colour missing"  # BGR red
    assert colour_pixels(frame, (255, 0, 0)) > 0, "accent colour missing"  # BGR blue


def test_dashes_leave_gaps(detections):
    solid, dashed = blank(), blank()

    BoxAnnotator(color=RED, thickness=2).annotate(scene=solid, detections=detections)
    DashedBoxAnnotator(color=RED, thickness=2, dash_length=6, gap_length=6).annotate(
        scene=dashed, detections=detections
    )

    assert painted(dashed) < painted(solid) * 0.75


def test_longer_dashes_cover_more(detections):
    short, long = blank(), blank()

    DashedBoxAnnotator(color=RED, dash_length=4, gap_length=10).annotate(
        scene=short, detections=detections
    )
    DashedBoxAnnotator(color=RED, dash_length=14, gap_length=10).annotate(
        scene=long, detections=detections
    )

    assert painted(long) > painted(short)


def test_dashed_corner_dashes_the_edges_between_the_corners(detections):
    frame = blank()

    DashedCornerAnnotator(
        color=RED, thickness=1, corner_length=12, dash_length=4, gap_length=4
    ).annotate(scene=frame, detections=detections)

    x1, y1, x2, y2 = detections.xyxy[0].astype(int)
    edge = frame[y1, x1 + 20 : x2 - 20].any(axis=1)
    assert edge.any(), "the dashed edge is missing between the corners"
    assert not edge.all(), "the edge should be dashed, not solid"


def test_dashed_corner_keeps_the_corners_solid(detections):
    frame = blank()

    DashedCornerAnnotator(
        color=RED, thickness=1, corner_length=16, dash_length=3, gap_length=6
    ).annotate(scene=frame, detections=detections)

    x1, y1 = detections.xyxy[0].astype(int)[:2]
    corner_arm = frame[y1, x1 : x1 + 16].any(axis=1)
    assert corner_arm.all(), "the corner arm must stay unbroken"


def test_dashed_corner_covers_more_than_the_plain_dashed_frame(detections):
    dashed, hybrid = blank(), blank()

    DashedBoxAnnotator(color=RED, thickness=2, dash_length=5, gap_length=5).annotate(
        scene=dashed, detections=detections
    )
    DashedCornerAnnotator(color=RED, thickness=2, dash_length=5, gap_length=5).annotate(
        scene=hybrid, detections=detections
    )

    assert painted(hybrid) > painted(dashed)


def test_bracket_draws_straight_arms_next_to_the_elbow(detections):
    frame = blank()

    BracketBoxAnnotator(color=RED, corner_length=24, roundness=1.0, thickness=2).annotate(
        scene=frame, detections=detections
    )

    # Yay en fazla kolun %60'ını kaplar; kolun dış ucunda düz parça kalmalı.
    x1, y1 = detections.xyxy[0].astype(int)[:2]
    outer_arm = frame[y1 - 1 : y1 + 2, x1 + 18 : x1 + 24]
    assert outer_arm.any(), "rounded elbow swallowed the straight arm"


def test_crosshair_marks_the_centre(detections):
    frame = blank()

    CrosshairAnnotator(color=RED, center_size=12, thickness=2).annotate(
        scene=frame, detections=detections
    )

    x1, y1, x2, y2 = detections.xyxy[0].astype(int)
    centre = frame[(y1 + y2) // 2 - 1 : (y1 + y2) // 2 + 2, (x1 + x2) // 2 - 1 : (x1 + x2) // 2 + 2]
    assert centre.any(), "centre cross missing"


def test_crosshair_leaves_the_corners_empty(detections):
    frame = blank()

    CrosshairAnnotator(color=RED, arm_length=8).annotate(scene=frame, detections=detections)

    x1, y1 = detections.xyxy[0].astype(int)[:2]
    assert not frame[y1 : y1 + 4, x1 : x1 + 4].any(), "reticle should not draw corners"


def test_target_keeps_the_full_rectangle(detections):
    frame = blank()

    TargetBoxAnnotator(color=RED, thickness=3, edge_thickness=1).annotate(
        scene=frame, detections=detections
    )

    x1, y1, x2, y2 = detections.xyxy[0].astype(int)
    middle_of_top_edge = frame[y1, (x1 + x2) // 2]
    assert middle_of_top_edge.any(), "thin edge missing between the corners"


def test_degenerate_boxes_are_skipped():
    frame = blank()
    flat = Detections(
        xyxy=np.array([[30, 30, 30, 30]], dtype=np.float32), class_id=np.array([0])
    )

    for annotator_type in (BracketBoxAnnotator, TargetBoxAnnotator):
        annotator_type(color=RED).annotate(scene=frame, detections=flat)

    assert painted(frame) == 0
