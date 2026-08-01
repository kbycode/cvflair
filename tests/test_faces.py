"""Yüz tespiti yolu: xywh kutular, MediaPipe tespiti ve beş noktalı iskelet."""

from __future__ import annotations

import types

import numpy as np
import pytest

from cvflair import FACE_5, SKELETONS, Detections, KeyPoints, Theme
from cvflair.keypoints import resolve_skeleton

WIDTH, HEIGHT = 640, 480


def blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def painted(frame: np.ndarray) -> int:
    return int(np.count_nonzero(frame.any(axis=2)))


# -- xywh -------------------------------------------------------------------


def test_corner_and_size_become_two_corners():
    """`detectMultiScale` çıktısının birebir biçimi."""
    faces = np.array([[100, 80, 60, 60], [300, 120, 90, 90]])

    detections = Detections.from_xywh(faces)

    assert np.array_equal(detections.xyxy, [[100, 80, 160, 140], [300, 120, 390, 210]])


def test_single_box_is_accepted():
    detections = Detections.from_xywh([10, 20, 30, 40])

    assert len(detections) == 1
    assert list(detections.xyxy[0]) == [10, 20, 40, 60]


def test_relative_boxes_are_scaled():
    """MediaPipe 0-1 aralığında veriyor; ölçek kare boyutundan geliyor."""
    detections = Detections.from_xywh([[0.5, 0.25, 0.25, 0.5]], WIDTH, HEIGHT)

    assert list(detections.xyxy[0]) == [320, 120, 480, 360]


def test_half_a_frame_size_is_refused():
    with pytest.raises(ValueError, match="width and height go together"):
        Detections.from_xywh([[0.5, 0.25, 0.25, 0.5]], WIDTH)


def test_other_fields_are_carried():
    detections = Detections.from_xywh(
        [[10, 10, 20, 20]], confidence=[0.8], class_id=[3], names=["yuz"]
    )

    assert detections.confidence[0] == pytest.approx(0.8)
    assert detections.class_id[0] == 3
    assert detections.names[0] == "yuz"


def test_no_faces_gives_no_boxes():
    """`detectMultiScale` hiçbir şey bulamayınca boş demet döndürür."""
    assert len(Detections.from_xywh(np.empty((0, 4)))) == 0


def test_boxes_are_drawn():
    frame = blank()

    Theme(palette=["#00E5FF"], labels=False).annotate(
        frame, Detections.from_xywh([[100, 80, 60, 60]])
    )

    assert painted(frame) > 0


# -- MediaPipe tespiti ------------------------------------------------------


def legacy_detection(xmin=0.2, ymin=0.15, width=0.15, height=0.2, score=0.93):
    """`solutions.face_detection` sonucu: oranlı kutu, liste hâlinde skor."""
    return types.SimpleNamespace(
        location_data=types.SimpleNamespace(
            relative_bounding_box=types.SimpleNamespace(
                xmin=xmin, ymin=ymin, width=width, height=height
            )
        ),
        score=[score],
    )


def tasks_detection(x=120, y=90, width=80, height=100, score=0.88):
    """`tasks.vision.FaceDetector` sonucu: piksel kutu, kategori skoru."""
    return types.SimpleNamespace(
        bounding_box=types.SimpleNamespace(origin_x=x, origin_y=y, width=width, height=height),
        categories=[types.SimpleNamespace(score=score)],
    )


def test_legacy_result_is_scaled_to_pixels():
    result = types.SimpleNamespace(detections=[legacy_detection()])

    detections = Detections.from_mediapipe(result, WIDTH, HEIGHT)

    assert list(detections.xyxy[0]) == pytest.approx([128, 72, 224, 168])
    assert detections.confidence[0] == pytest.approx(0.93)


def test_tasks_result_is_already_in_pixels():
    result = types.SimpleNamespace(detections=[tasks_detection()])

    detections = Detections.from_mediapipe(result, WIDTH, HEIGHT)

    assert list(detections.xyxy[0]) == pytest.approx([120, 90, 200, 190])
    assert detections.confidence[0] == pytest.approx(0.88)


def test_several_faces():
    result = types.SimpleNamespace(
        detections=[legacy_detection(), legacy_detection(xmin=0.6)]
    )

    assert len(Detections.from_mediapipe(result, WIDTH, HEIGHT)) == 2


def test_bare_detection_list_is_accepted():
    assert len(Detections.from_mediapipe([tasks_detection()], WIDTH, HEIGHT)) == 1


@pytest.mark.parametrize(
    "result",
    [types.SimpleNamespace(detections=None), types.SimpleNamespace(detections=[]), [], None],
)
def test_no_face_gives_empty_detections(result):
    assert len(Detections.from_mediapipe(result, WIDTH, HEIGHT)) == 0


def test_unknown_detection_shape_is_named():
    result = types.SimpleNamespace(detections=[types.SimpleNamespace(kutu=[1, 2, 3, 4])])

    with pytest.raises(TypeError, match="MediaPipe tespiti tanınmadı"):
        Detections.from_mediapipe(result, WIDTH, HEIGHT)


def test_missing_scores_leave_confidence_empty():
    detection = tasks_detection()
    detection.categories = []

    detections = Detections.from_mediapipe([detection], WIDTH, HEIGHT)

    assert detections.confidence is None, "skor yoksa uydurulmamalı"


def test_mediapipe_faces_are_drawn():
    frame = blank()
    result = types.SimpleNamespace(detections=[legacy_detection()])

    Theme(palette=["#00E5FF"], labels=False).annotate(
        frame, Detections.from_mediapipe(result, WIDTH, HEIGHT)
    )

    assert painted(frame) > 0


# -- beş noktalı iskelet ----------------------------------------------------


def face_points() -> np.ndarray:
    """Sol göz, sağ göz, burun, sol ağız köşesi, sağ ağız köşesi."""
    return np.array([[[120, 100], [180, 100], [150, 130], [128, 160], [172, 160]]])


def test_face_is_registered_by_name():
    assert SKELETONS["face"] is FACE_5
    assert resolve_skeleton("face") == FACE_5


def test_topology_links_the_five_points():
    used = {index for edge in FACE_5 for index in edge}

    assert used == {0, 1, 2, 3, 4}, "beş noktanın hepsi bağlanmalı"
    assert (0, 1) not in FACE_5, "gözler arası çizgi yüzü ikiye böler"


def test_face_skeleton_is_drawn():
    frame = blank()

    Theme(palette=["#00E5FF"]).annotate_keypoints(frame, KeyPoints(xy=face_points()), "face")

    assert painted(frame) > 0


def test_insightface_output_flows_through():
    """InsightFace: bbox zaten xyxy, kps beş nokta, det_score güven."""
    face = types.SimpleNamespace(
        bbox=np.array([100.0, 80.0, 200.0, 200.0]),
        det_score=0.97,
        kps=face_points()[0],
    )
    frame = blank()
    theme = Theme(palette=["#00E5FF"], labels=False)

    theme.annotate(
        frame,
        Detections(xyxy=np.array([face.bbox]), confidence=np.array([face.det_score])),
    )
    theme.annotate_keypoints(frame, KeyPoints(xy=np.array([face.kps])), FACE_5)

    assert painted(frame) > 0


def test_small_faces_still_draw():
    """Kalabalıkta yüz 20 piksele iner; iskelet o ölçekte de çizilmeli."""
    frame = blank()
    tiny = face_points() * 0.2 + np.array([40, 40])

    Theme(palette=["#00E5FF"], pose_radius=1, pose_thickness=1).annotate_keypoints(
        frame, KeyPoints(xy=tiny), FACE_5
    )

    assert painted(frame) > 0
