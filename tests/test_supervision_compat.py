"""
Interop with supervision.

cvflair does not depend on supervision, but people who already use it must be
able to hand their ``sv.Detections`` straight to a theme. These tests run only
when supervision happens to be installed (it is in the dev extra).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import Theme, get_theme
from cvflair.colors import resolve_palette
from cvflair.detections import detection_names, is_detections
from cvflair.themes import BOX_STYLES

sv = pytest.importorskip("supervision", reason="supervision is not installed")


@pytest.fixture
def sv_detections():
    return sv.Detections(
        xyxy=np.array([[20, 20, 90, 80], [100, 30, 150, 100]], dtype=np.float32),
        class_id=np.array([0, 1]),
        confidence=np.array([0.9, 0.7], dtype=np.float32),
        data={"class_name": np.array(["kisi", "kopek"], dtype=object)},
    )


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def test_supervision_detections_are_recognised(sv_detections):
    assert is_detections(sv_detections)


def test_class_names_are_read_from_the_data_dict(sv_detections):
    assert detection_names(sv_detections).tolist() == ["kisi", "kopek"]


@pytest.mark.parametrize("box_style", BOX_STYLES)
def test_every_style_draws_supervision_detections(box_style, sv_detections):
    frame = blank()

    Theme(box_style=box_style, thickness=2).annotate(frame, sv_detections)

    assert painted(frame) > 0


def test_labels_use_supervision_class_names(sv_detections):
    named, plain = blank(), blank()

    get_theme("minimal").annotate(named, sv_detections)
    get_theme("minimal").annotate(
        plain, sv.Detections(xyxy=sv_detections.xyxy, class_id=sv_detections.class_id)
    )

    assert not np.array_equal(named, plain), "class names were ignored"


def test_supervision_palette_is_accepted():
    palette = resolve_palette(sv.ColorPalette.from_hex(["#39FF14", "#FF00E5"]))

    assert [colour.as_hex() for colour in palette.colors] == ["#39FF14", "#FF00E5"]


def test_a_theme_can_be_built_from_supervision_colours():
    theme = Theme(
        palette=sv.ColorPalette.from_hex(["#00F0FF"]),
        accent_palette=sv.Color.WHITE,
        box_style="target",
    )

    assert theme.palette.colors[0].as_hex() == "#00F0FF"
    assert theme.accent_palette.colors[0].as_hex() == "#FFFFFF"


def test_ultralytics_conversion_matches_supervision():
    """Our from_ultralytics should agree with supervision's on plain boxes."""
    from test_models import FakeResults

    from cvflair import Detections

    ours = Detections.from_ultralytics(FakeResults())
    theirs = sv.Detections.from_ultralytics(FakeResults())

    assert np.allclose(ours.xyxy, theirs.xyxy)
    assert ours.class_id.tolist() == theirs.class_id.tolist()
    assert np.allclose(ours.confidence, theirs.confidence)
    assert list(ours.names) == list(theirs.data["class_name"])
