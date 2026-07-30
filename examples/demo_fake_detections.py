"""
Modelsiz canlı demo: kamera görüntüsü üzerine hareketli sahte tespitler çizer.

Faz 1'de model bağlama yok; bu betik tespitlerin yerine hareket eden kutular
üreterek temayı canlı gösterir. Temalar birkaç saniyede bir sırayla değişir.

Çalıştırmak için:  python examples/demo_fake_detections.py
Çıkmak için:      pencere seçiliyken 'q' veya ESC
"""

import math
import time

import numpy as np
import supervision as sv

from cvflair import Camera, available_themes

#: Kaç saniyede bir sıradaki temaya geçileceği. Kare sayısı yerine süre
#: kullanılıyor, çünkü kare hızı cihaza göre değişiyor.
THEME_SWITCH_SECONDS = 3.0

CLASS_NAMES = ("kutu", "hedef")


def _pulse(tick: int, period: int, low: float, high: float) -> float:
    """(low, high) aralığında ileri geri salınan bir değer."""
    phase = (math.sin(2 * math.pi * tick / period) + 1) / 2
    return low + (high - low) * phase


def fake_detections(width: int, height: int, tick: int) -> sv.Detections:
    """İki kutu: biri yatay, diğeri dikey salınıyor."""
    box_w, box_h = width * 0.26, height * 0.42
    left = _pulse(tick, 140, 0.04, 0.42) * width
    top = height * 0.18

    small_w, small_h = width * 0.20, height * 0.28
    small_left = width * 0.62
    small_top = _pulse(tick, 100, 0.10, 0.55) * height

    return sv.Detections(
        xyxy=np.array(
            [
                [left, top, left + box_w, top + box_h],
                [small_left, small_top, small_left + small_w, small_top + small_h],
            ],
            dtype=np.float32,
        ),
        class_id=np.array([0, 1]),
        confidence=np.array(
            [_pulse(tick, 70, 0.55, 0.99), _pulse(tick, 55, 0.40, 0.95)], dtype=np.float32
        ),
    )


def main() -> None:
    themes = available_themes()
    print(f"Temalar sırayla: {', '.join(themes)} — çıkmak için 'q' veya ESC")

    cam = Camera(source=0, theme=themes[0])
    started = time.monotonic()
    for tick, frame in enumerate(cam.stream()):
        elapsed = time.monotonic() - started
        wanted = themes[int(elapsed // THEME_SWITCH_SECONDS) % len(themes)]
        if wanted != cam.theme.name:
            # Tema atamak annotator'ları yeniden kurar; sadece ad değişince yapılıyor.
            cam.theme = wanted

        height, width = frame.shape[:2]
        detections = fake_detections(width, height, tick)
        labels = [
            f"{name} {confidence:.2f}"
            for name, confidence in zip(CLASS_NAMES, detections.confidence, strict=True)
        ]
        cam.show(frame, detections, labels=labels)


if __name__ == "__main__":
    main()
