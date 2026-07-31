"""
Sayaç panelini kendi verinle beslemek.

`Camera` panele kare hızını ve tespit sayısını kendisi yazar; `stats` ile
istediğin satırı eklersin. Aynı anahtarı verirsen seninki geçerli olur.

Bu örnek hareketi sayar: kadrajın sol yarısına giren her nesne için puan.

Çalıştırmak için:  python examples/hud_stats.py
Çıkmak için:      pencere seçiliyken 'q' veya ESC
"""

import time

from motion_detection import MotionDetector

from cvflair import Camera


def main() -> None:
    cam = Camera(source=0, theme="hud")
    detector = MotionDetector()

    started = time.monotonic()
    score = 0
    seen_left = False

    for frame, detections in cam.stream(model=detector):
        middle = frame.shape[1] / 2
        on_the_left = any(box[2] < middle for box in detections.xyxy)

        # Yalnızca geçişte say: kutu sol yarıya ilk girdiğinde bir puan.
        if on_the_left and not seen_left:
            score += 1
        seen_left = on_the_left

        cam.show(
            frame,
            detections,
            stats={
                "Skor": score,
                "Sure": f"{time.monotonic() - started:.0f}s",
            },
        )


if __name__ == "__main__":
    main()
