"""
El iskeleti: 21 noktalı bir el, kamera görüntüsünün üzerinde temayla çizilir.

Buradaki el sentetik — parmakları açılıp kapanan bir model, ek kurulum
gerektirmesin diye. Gerçek bir elde tek fark noktaların nereden geldiği:

    import mediapipe as mp

    hands = mp.solutions.hands.Hands(max_num_hands=2)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    for landmarks in result.multi_hand_landmarks or []:
        points = KeyPoints.from_normalized(
            [(point.x, point.y) for point in landmarks.landmark],
            width=frame.shape[1],
            height=frame.shape[0],
        )
        cam.show(frame, keypoints=points, skeleton=HAND_21)

Tuşlar:  1-5 tema · q veya ESC çıkış

Çalıştırmak için:  python examples/hand_skeleton.py
"""

import math
import time

import numpy as np

from cvflair import HAND_21, Camera, KeyPoints, available_themes

#: Parmak yönleri (radyan) ve boyları; MediaPipe sırasıyla aynı.
FINGERS = (
    (-1.35, 0.62),   # başparmak
    (-0.42, 1.00),   # işaret
    (-0.12, 1.08),   # orta
    (0.18, 1.00),    # yüzük
    (0.50, 0.82),    # serçe
)


def hand(center: tuple[float, float], curl: float, scale: float = 1.0) -> np.ndarray:
    """
    21 noktalı bir el üretir.

    ``curl`` 0'da parmaklar açık, 1'de kapalı: eklemler ilerledikçe yön açısı
    artıyor, böylece parmak avuca doğru kıvrılıyor.
    """
    cx, cy = center
    points = [(cx, cy)]  # 0: bilek

    for angle, length in FINGERS:
        base = np.array([cx, cy]) + np.array([math.sin(angle), -math.cos(angle)]) * 46 * length
        position = base
        heading = angle
        points.append(tuple(position))
        for joint in range(3):
            heading += curl * 0.75
            step = np.array([math.sin(heading), -math.cos(heading)]) * (22 - joint * 2) * length
            position = position + step * scale
            points.append(tuple(position))

    return np.array(points, dtype=np.float32)


def main() -> None:
    themes = available_themes()
    print("Temalar: " + "  ".join(f"[{i + 1}] {n}" for i, n in enumerate(themes)))
    print("[q] çıkış")

    cam = Camera(source=0, theme="neon")
    started = time.monotonic()

    for frame in cam.stream():
        height, width = frame.shape[:2]
        elapsed = time.monotonic() - started
        curl = (math.sin(elapsed * 1.6) + 1) / 2          # açılıp kapanma
        drift = math.sin(elapsed * 0.7) * width * 0.12    # sağa sola salınım

        points = KeyPoints(
            xy=hand((width / 2 + drift, height * 0.75), curl, scale=height / 480),
            confidence=np.ones((1, 21), dtype=np.float32),
        )
        cam.show(frame, keypoints=points, skeleton=HAND_21)

        for index, name in enumerate(themes):
            if cam.pressed(str(index + 1)):
                cam.theme = name
                print(f"tema: {name}")


if __name__ == "__main__":
    main()
