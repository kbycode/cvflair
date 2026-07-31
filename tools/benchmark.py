"""
Çizim maliyetini ölçer: tema başına kare süresi, elle yazılmış OpenCV döngüsüne
göre ek yük, ve annotator'ları döngü içinde kurmanın bedeli.

Çalıştırmak için:
    python tools/benchmark.py
    python tools/benchmark.py --width 1920 --height 1080 --boxes 20 --repeat 300

Sayılar makineye özgüdür; karşılaştırma aynı koşuda anlamlıdır.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable

import cv2
import numpy as np

from cvflair import Detections, Theme, available_themes, get_theme

FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_frame(width: int, height: int) -> np.ndarray:
    column = np.linspace(30, 90, width, dtype=np.uint8)
    return np.repeat(column[None, :, None], height, axis=0).repeat(3, axis=2).copy()


def make_detections(count: int, width: int, height: int) -> Detections:
    rng = np.random.default_rng(0)
    boxes = []
    for _ in range(count):
        box_width = rng.integers(width // 12, width // 5)
        box_height = rng.integers(height // 10, height // 3)
        x1 = rng.integers(0, width - box_width)
        y1 = rng.integers(0, height - box_height)
        boxes.append([x1, y1, x1 + box_width, y1 + box_height])

    return Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=rng.integers(0, 4, size=count),
        confidence=rng.uniform(0.5, 0.99, size=count).astype(np.float32),
        names=np.array(["nesne"] * count, dtype=object),
    )


def with_masks(detections: Detections, width: int, height: int) -> Detections:
    """Aynı kutulara içten teğet elips maskeler; segmentasyon çıktısını taklit eder."""
    rows, columns = np.ogrid[:height, :width]
    masks = []
    for x1, y1, x2, y2 in detections.xyxy:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = max((x2 - x1) / 2, 1), max((y2 - y1) / 2, 1)
        masks.append(((columns - cx) / rx) ** 2 + ((rows - cy) / ry) ** 2 <= 1)
    return Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        names=detections.names,
        mask=np.stack(masks),
    )


def bare_opencv(frame: np.ndarray, detections: Detections, labels: list[str]) -> None:
    """Kütüphanesiz karşılaştırma tabanı: kutu + etiket, elle."""
    for index in range(len(detections)):
        x1, y1, x2, y2 = detections.xyxy[index].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), _ = cv2.getTextSize(labels[index], FONT, 0.5, 1)
        plate = (x1 + text_width + 12, y1)
        cv2.rectangle(frame, (x1, y1 - text_height - 12), plate, (0, 255, 0), -1)
        cv2.putText(frame, labels[index], (x1 + 6, y1 - 6), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def measure(work: Callable[[], None], repeat: int) -> tuple[float, float]:
    """Ortanca ve en iyi kare süresini milisaniye olarak döndürür."""
    for _ in range(5):  # ısınma: ilk çağrılar önbellek ve tahsis maliyeti taşıyor
        work()

    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        work()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples), min(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--boxes", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    frame = make_frame(args.width, args.height)
    detections = make_detections(args.boxes, args.width, args.height)
    labels = [f"nesne {score:.2f}" for score in detections.confidence]

    print(f"{platform.python_implementation()} {platform.python_version()} | "
          f"OpenCV {cv2.__version__} | {platform.processor() or platform.machine()}")
    print(f"{args.width}x{args.height}, {args.boxes} kutu, {args.repeat} tekrar\n")

    baseline, _ = measure(lambda: bare_opencv(frame, detections, labels), args.repeat)
    print("| Tema | ms/kare | kare/sn | elle çizime göre |")
    print("|---|---|---|---|")
    print(f"| _elle OpenCV_ | {baseline:.2f} | {1000 / baseline:.0f} | — |")

    def draw_with(theme: Theme, data: Detections | None = None) -> Callable[[], None]:
        boxes = detections if data is None else data
        return lambda: theme.annotate(frame, boxes, labels=labels)

    for name in available_themes():
        median, _ = measure(draw_with(get_theme(name)), args.repeat)
        overhead = (median - baseline) / baseline * 100
        print(f"| `{name}` | {median:.2f} | {1000 / median:.0f} | {overhead:+.0f}% |")

    # Maske çizimi ayrı ölçülüyor: yalnızca tespitte maske varsa çalışıyor ve
    # maliyeti piksel başına, yani kapladığı alanla orantılı.
    masked = with_masks(detections, args.width, args.height)
    mask_theme = Theme(box_style="box", mask_opacity=0.4, mask_outline=2)
    outline_theme = Theme(box_style="box", mask_opacity=0.0, mask_outline=2)
    fill_ms, _ = measure(lambda: mask_theme.annotate(frame, masked, labels=labels), args.repeat)
    line_ms, _ = measure(lambda: outline_theme.annotate(frame, masked, labels=labels), args.repeat)
    print(f"\nMaske (aynı sahne, {args.boxes} maske):")
    print(f"  dolgu + kontur : {fill_ms:.2f} ms  (+{fill_ms - baseline:.2f})")
    print(f"  yalnız kontur  : {line_ms:.2f} ms  (+{line_ms - baseline:.2f})")

    # Kutunun üstüne binen çizimler: her biri aynı sahnede, düz kutuya göre.
    plain = Theme(box_style="box")
    plain_ms, _ = measure(lambda: plain.annotate(frame, detections, labels=labels), args.repeat)
    tracked = Detections(
        xyxy=detections.xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        names=detections.names,
        tracker_id=np.arange(len(detections)),
    )
    extras: list[tuple[str, Theme, Detections]] = [
        ("sketch (çerçeve)", Theme(box_style="sketch"), detections),
        ("pulse", Theme(box_style="box", pulse=True), detections),
        ("trace (32 nokta)", Theme(box_style="box", trace=True), tracked),
    ]
    print(f"\nEk çizimler (düz kutu {plain_ms:.2f} ms üzerine):")
    for name, theme, data in extras:
        for _ in range(40):  # iz dolsun, ölçüm tam uzunlukta yapılsın
            theme.annotate(frame, data, labels=labels)
        median, _ = measure(draw_with(theme, data), args.repeat)
        print(f"  {name:18s}: {median:5.2f} ms  (+{median - plain_ms:.2f})")

    # Tema kurulumunun kendi maliyeti: annotator'lar burada hazırlanıyor. Döngü
    # içinde tema kurmak bu süreyi her kareye ekler.
    def build_theme(name: str) -> Callable[[], None]:
        return lambda: get_theme(name)

    print("\n| Tema | kurulum ms | döngü içinde kurulursa çizime eklenen |")
    print("|---|---|---|")
    for name in available_themes():
        build, _ = measure(build_theme(name), args.repeat)
        draw, _ = measure(draw_with(get_theme(name)), args.repeat)
        print(f"| `{name}` | {build:.3f} | {build / draw * 100:+.0f}% |")


if __name__ == "__main__":
    main()
