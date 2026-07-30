"""
Kamerasız tema önizlemesi: sahte tespitlerle her temayı bir PNG'ye çizer.

Çalıştırmak için:  python examples/theme_preview.py
Çıktı:             examples/output/theme-<ad>.png
"""

from pathlib import Path

import cv2
import numpy as np

from cvflair import Detections, available_themes, get_theme

OUTPUT_DIR = Path(__file__).parent / "output"

# Tespit üreten bir model yerine sabit kutular: amaç temanın görünümü.
DETECTIONS = Detections(
    xyxy=np.array(
        [[60, 90, 300, 380], [340, 140, 560, 330], [600, 60, 760, 260]], dtype=np.float32
    ),
    class_id=np.array([0, 1, 2]),
    confidence=np.array([0.94, 0.81, 0.66], dtype=np.float32),
)
LABELS = ["kisi 0.94", "bisiklet 0.81", "kopek 0.66"]


def backdrop(width: int = 840, height: int = 460) -> np.ndarray:
    """Koyu gri, hafif degradeli bir zemin — kutuların kontrastını görmek için."""
    column = np.linspace(28, 58, width, dtype=np.uint8)
    return np.repeat(column[None, :, None], height, axis=0).repeat(3, axis=2)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in available_themes():
        frame = get_theme(name).annotate(backdrop(), DETECTIONS, labels=LABELS)
        path = OUTPUT_DIR / f"theme-{name}.png"
        cv2.imwrite(str(path), frame)
        print(f"{name:>8} -> {path}")


if __name__ == "__main__":
    main()
