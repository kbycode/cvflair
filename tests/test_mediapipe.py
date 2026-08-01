"""MediaPipe sonuçlarını okuma. MediaPipe kurulu değil; alanlar taklit ediliyor."""

from __future__ import annotations

import types

import numpy as np
import pytest

from cvflair import HAND_21, KeyPoints, Theme

WIDTH, HEIGHT = 200, 100


class Landmark:
    """`x`, `y` normalize; görünürlük modele göre dolu ya da sıfır."""

    def __init__(self, x: float, y: float, visibility: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = visibility


def points(count: int = 3, visibility: float = 0.0) -> list[Landmark]:
    return [Landmark(0.1 * (index + 1), 0.2 * (index + 1), visibility) for index in range(count)]


def solutions_result(field: str, groups: list[list[Landmark]]):
    """Eski `solutions` API'si: noktalar `.landmark` altında bir sarmalayıcıda."""
    wrapped = [types.SimpleNamespace(landmark=group) for group in groups]
    return types.SimpleNamespace(**{field: wrapped})


def tasks_result(field: str, groups: list[list[Landmark]]):
    """Yeni `tasks` API'si: düz liste."""
    return types.SimpleNamespace(**{field: groups})


# -- biçimler ---------------------------------------------------------------


def test_solutions_hands_result():
    result = solutions_result("multi_hand_landmarks", [points(), points()])

    keypoints = KeyPoints.from_mediapipe(result, WIDTH, HEIGHT)

    assert len(keypoints) == 2
    assert keypoints.point_count == 3


def test_tasks_hands_result():
    keypoints = KeyPoints.from_mediapipe(tasks_result("hand_landmarks", [points()]), WIDTH, HEIGHT)

    assert len(keypoints) == 1


def test_solutions_pose_result_is_a_single_skeleton():
    result = types.SimpleNamespace(pose_landmarks=types.SimpleNamespace(landmark=points(5)))

    keypoints = KeyPoints.from_mediapipe(result, WIDTH, HEIGHT)

    assert len(keypoints) == 1
    assert keypoints.point_count == 5


def test_bare_landmark_list_is_accepted():
    keypoints = KeyPoints.from_mediapipe(points(4), WIDTH, HEIGHT)

    assert len(keypoints) == 1 and keypoints.point_count == 4


def test_bare_list_of_lists_is_accepted():
    keypoints = KeyPoints.from_mediapipe([points(), points()], WIDTH, HEIGHT)

    assert len(keypoints) == 2


# -- ölçek ve güven ---------------------------------------------------------


def test_coordinates_are_scaled_to_pixels():
    keypoints = KeyPoints.from_mediapipe(tasks_result("hand_landmarks", [points(1)]), WIDTH, HEIGHT)

    assert keypoints.xy[0, 0] == pytest.approx([0.1 * WIDTH, 0.2 * HEIGHT])


def test_visibility_becomes_confidence():
    result = tasks_result("pose_landmarks", [points(3, visibility=0.8)])

    keypoints = KeyPoints.from_mediapipe(result, WIDTH, HEIGHT)

    assert keypoints.confidence is not None
    assert keypoints.confidence[0, 0] == pytest.approx(0.8)


def test_all_zero_visibility_is_dropped():
    """
    El modeli görünürlük doldurmuyor. Sıfırı güven diye aktarmak eşiğin altında
    kalır ve tek bir nokta bile çizilmezdi.
    """
    result = solutions_result("multi_hand_landmarks", [points(visibility=0.0)])

    keypoints = KeyPoints.from_mediapipe(result, WIDTH, HEIGHT)

    assert keypoints.confidence is None


def test_points_are_actually_drawn_after_conversion():
    """Dönüşümün asıl ölçütü: çıkan noktalar kareye çiziliyor mu."""
    hand = [Landmark(0.1 + 0.04 * index, 0.3 + 0.02 * index) for index in range(21)]
    keypoints = KeyPoints.from_mediapipe(solutions_result("multi_hand_landmarks", [hand]), 320, 240)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    Theme(palette=["#00FF00"]).annotate_keypoints(frame, keypoints, HAND_21)

    assert np.count_nonzero(frame.any(axis=2)) > 0


# -- boş sonuçlar -----------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        types.SimpleNamespace(multi_hand_landmarks=None),  # el bulunamadı
        types.SimpleNamespace(multi_hand_landmarks=[]),
        types.SimpleNamespace(pose_landmarks=None),
        [],
        None,
    ],
)
def test_nothing_found_gives_empty_points(result):
    keypoints = KeyPoints.from_mediapipe(result, WIDTH, HEIGHT)

    assert len(keypoints) == 0
