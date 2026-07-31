"""Nokta taşıyıcısı, iskelet topolojileri ve iskelet çizimi."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SHAPE

from cvflair import HAND_21, POSE_17, KeyPoints, Theme, available_themes, get_theme
from cvflair.annotators import EdgeAnnotator, VertexAnnotator
from cvflair.keypoints import SKELETONS, is_keypoints, resolve_skeleton


def blank() -> np.ndarray:
    return np.zeros(FRAME_SHAPE, dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


def hand_points(offset: float = 0.0) -> np.ndarray:
    """21 nokta; gerçek bir el değil ama topolojiye uygun ve kadraj içinde."""
    rng = np.random.default_rng(3)
    points = rng.uniform([20, 20], [140, 100], size=(21, 2)) + offset
    return points.astype(np.float32)


@pytest.fixture
def hand() -> KeyPoints:
    return KeyPoints(xy=hand_points(), confidence=np.ones((1, 21), dtype=np.float32))


# -- taşıyıcı ---------------------------------------------------------------


def test_single_skeleton_is_reshaped():
    points = KeyPoints(xy=hand_points())

    assert points.xy.shape == (1, 21, 2)
    assert len(points) == 1
    assert points.point_count == 21


def test_many_skeletons_are_kept():
    points = KeyPoints(xy=np.stack([hand_points(), hand_points(30)]))

    assert len(points) == 2
    assert points.point_count == 21


def test_bad_shape_is_rejected():
    with pytest.raises(ValueError, match=r"xy must be shaped"):
        KeyPoints(xy=np.zeros((21, 3), dtype=np.float32))


def test_confidence_length_is_checked():
    with pytest.raises(ValueError, match="confidence has 5 points but xy has 21"):
        KeyPoints(xy=hand_points(), confidence=np.ones((1, 5)))


def test_class_id_length_is_checked():
    with pytest.raises(ValueError, match="class_id has 2 entries but xy has 1"):
        KeyPoints(xy=hand_points(), class_id=[0, 1])


def test_empty_and_duck_check():
    empty = KeyPoints.empty(21)

    assert len(empty) == 0
    assert is_keypoints(empty)
    assert not is_keypoints("iskelet")


def test_normalised_points_are_scaled():
    points = KeyPoints.from_normalized([[0.5, 0.25], [1.0, 1.0]], width=200, height=80)

    assert points.xy[0][0].tolist() == [100.0, 20.0]
    assert points.xy[0][1].tolist() == [200.0, 80.0]


# -- iskelet tanımları ------------------------------------------------------


@pytest.mark.parametrize(("skeleton", "points"), [(HAND_21, 21), (POSE_17, 17)])
def test_shipped_skeletons_stay_in_range(skeleton, points):
    indices = {index for pair in skeleton for index in pair}

    assert max(indices) == points - 1, "en yüksek indeks nokta sayısıyla uyuşmuyor"
    assert min(indices) == 0
    assert all(first != second for first, second in skeleton), "nokta kendine bağlanmış"


def test_hand_topology_connects_every_finger():
    # Bilekten çıkan zincirler: her parmak bileğe doğrudan ya da dolaylı bağlı.
    reachable = {0}
    for _ in range(len(HAND_21)):
        for first, second in HAND_21:
            if first in reachable:
                reachable.add(second)
            if second in reachable:
                reachable.add(first)

    assert reachable == set(range(21)), "kopuk nokta var"


def test_skeleton_can_be_named():
    assert resolve_skeleton("hand") is HAND_21
    assert resolve_skeleton("  POSE ") is POSE_17
    assert resolve_skeleton([(0, 1), (1, 2)]) == ((0, 1), (1, 2))
    assert set(SKELETONS) == {"hand", "pose"}


def test_unknown_skeleton_lists_the_options():
    with pytest.raises(ValueError, match="Available: hand, pose"):
        resolve_skeleton("kanat")


# -- çizim ------------------------------------------------------------------


@pytest.mark.parametrize("name", available_themes())
def test_every_theme_draws_a_skeleton(name, hand):
    frame = blank()

    get_theme(name).annotate_keypoints(frame, hand, HAND_21)

    assert painted(frame) > 0


def test_drawing_is_in_place(hand):
    frame = blank()

    assert get_theme("neon").annotate_keypoints(frame, hand, HAND_21) is frame


def test_empty_keypoints_leave_the_frame_alone():
    frame = blank()

    get_theme("neon").annotate_keypoints(frame, KeyPoints.empty(21), HAND_21)

    assert painted(frame) == 0


def test_low_confidence_points_are_skipped():
    strong = KeyPoints(xy=hand_points(), confidence=np.ones((1, 21)))
    weak = KeyPoints(xy=hand_points(), confidence=np.full((1, 21), 0.05))
    theme = Theme(pose_confidence=0.3)

    busy, quiet = blank(), blank()
    theme.annotate_keypoints(busy, strong, HAND_21)
    theme.annotate_keypoints(quiet, weak, HAND_21)

    assert painted(busy) > 0
    assert painted(quiet) == 0


def test_non_finite_points_are_skipped_without_raising():
    points = hand_points()
    points[3] = [np.nan, np.nan]
    points[7] = [np.inf, 10]
    frame = blank()

    get_theme("neon").annotate_keypoints(frame, KeyPoints(xy=points), HAND_21)

    assert painted(frame) > 0, "diğer noktalar çizilmeye devam etmeli"


def test_skeleton_longer_than_the_data_is_tolerated():
    """17 noktalı veriye el topolojisi verilirse fazla kenarlar atlanmalı."""
    frame = blank()
    short = KeyPoints(xy=hand_points()[:17])

    get_theme("minimal").annotate_keypoints(frame, short, HAND_21)

    assert painted(frame) > 0


def test_thicker_bones_cover_more(hand):
    thin, thick = blank(), blank()

    Theme(pose_thickness=1, pose_radius=1).annotate_keypoints(thin, hand, HAND_21)
    Theme(pose_thickness=5, pose_radius=4).annotate_keypoints(thick, hand, HAND_21)

    assert painted(thick) > painted(thin)


def test_joints_take_the_accent_colour(hand):
    frame = blank()
    theme = Theme(palette=["#FF0000"], accent_palette=["#0000FF"], pose_radius=4)

    theme.annotate_keypoints(frame, hand, HAND_21)

    blue = int(np.count_nonzero(np.all(frame == np.array([255, 0, 0], np.uint8), axis=2)))
    red = int(np.count_nonzero(np.all(frame == np.array([0, 0, 255], np.uint8), axis=2)))
    assert red > 0, "kemikler palet renginde olmalı"
    assert blue > 0, "eklemler vurgu renginde olmalı"


def test_glow_puts_a_wider_pass_behind_the_bones(hand):
    plain, glowing = blank(), blank()

    Theme(pose_thickness=2).annotate_keypoints(plain, hand, HAND_21)
    Theme(pose_thickness=2, glow=True, glow_thickness=6).annotate_keypoints(
        glowing, hand, HAND_21
    )

    assert painted(glowing) > painted(plain)


def test_annotators_can_be_used_directly(hand):
    frame = blank()

    EdgeAnnotator(color=["#FFFFFF"], thickness=2).annotate(frame, hand, HAND_21)
    VertexAnnotator(color=["#FFFFFF"], radius=3).annotate(frame, hand)

    assert painted(frame) > 0


def test_camera_draws_skeletons_through_show(camera_factory, hand):
    camera, _ = camera_factory(frame_count=1, theme="neon")
    frame = blank()

    camera.annotate(frame, keypoints=hand, skeleton="hand")

    assert painted(frame) > 0
