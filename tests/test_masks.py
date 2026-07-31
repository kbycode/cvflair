"""Segmentasyon maskesi çizimi."""

from __future__ import annotations

import numpy as np
import pytest

from cvflair import Detections, MaskAnnotator, Theme

SHAPE = (120, 160)


def blank() -> np.ndarray:
    return np.zeros((*SHAPE, 3), dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def disc(cx: int = 80, cy: int = 60, radius: int = 30) -> np.ndarray:
    y, x = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def with_mask(mask: np.ndarray | None = None) -> Detections:
    return Detections(
        xyxy=[[50, 30, 110, 90]],
        class_id=[0],
        confidence=[0.9],
        mask=disc() if mask is None else mask,
    )


def test_mask_is_reshaped_and_validated():
    detections = with_mask()

    assert detections.mask.shape == (1, *SHAPE)
    assert detections.mask.dtype == bool

    with pytest.raises(ValueError, match="mask has 2 entries but xyxy has 1"):
        Detections(xyxy=[[0, 0, 10, 10]], mask=np.zeros((2, *SHAPE), dtype=bool))


def test_fill_and_outline_are_drawn():
    frame = blank()

    MaskAnnotator(color=["#00FF00"], opacity=0.5, outline=2).annotate(frame, with_mask())

    assert painted(frame) > 0


def test_fill_blends_the_scene_through():
    frame = blank()
    frame[:] = 200

    MaskAnnotator(color=["#000000"], opacity=0.5, outline=0).annotate(frame, with_mask())

    inside = frame[60, 80]
    assert 0 < int(inside.max()) < 200, "dolgu sahneyi tamamen örtmemeli"


def test_outline_stays_full_strength():
    """Kontur harmanlamadan sonra çizilmeli, yoksa solar."""
    frame = blank()

    MaskAnnotator(color=["#FF0000"], opacity=0.4, outline=3).annotate(frame, with_mask())

    red = np.array([0, 0, 255], dtype=np.uint8)
    assert int(np.count_nonzero(np.all(frame == red, axis=2))) > 0


def test_outline_only_leaves_the_middle_alone():
    frame = blank()
    frame[:] = 150

    MaskAnnotator(color=["#FF0000"], opacity=0.0, outline=2).annotate(frame, with_mask())

    assert int(frame[60, 80].max()) == 150, "dolgu istenmediği hâlde uygulanmış"


def test_wrong_sized_mask_is_skipped():
    frame = blank()
    odd = Detections(xyxy=[[10, 10, 40, 40]], class_id=[0], mask=np.ones((1, 40, 40), dtype=bool))

    MaskAnnotator(opacity=0.6, outline=2).annotate(frame, odd)

    assert painted(frame) == 0


def test_detections_without_masks_are_ignored():
    frame = blank()

    MaskAnnotator().annotate(frame, Detections(xyxy=[[10, 10, 40, 40]], class_id=[0]))

    assert painted(frame) == 0


def test_theme_draws_masks_under_the_box():
    frame = blank()

    Theme(palette=["#FF0000"], mask_opacity=0.4).annotate(frame, with_mask())

    assert painted(frame) > 0


def test_masks_can_be_switched_off():
    with_masks, without = blank(), blank()

    Theme(masks=True, mask_opacity=0.5).annotate(with_masks, with_mask())
    Theme(masks=False).annotate(without, with_mask())

    assert painted(with_masks) > painted(without)
    assert Theme(masks=False)._mask_annotator is None
