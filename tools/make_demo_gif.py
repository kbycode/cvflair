"""
README'deki demo GIF'ini üretir: sentetik bir sahne, hareketli iki nesne ve
sırayla değişen temalar. Kamera gerektirmez, çıktısı her çalıştırmada aynıdır.

Çalıştırmak için:  python tools/make_demo_gif.py
Çıktı:             docs/demo.gif

Pillow gerekir (dev bağımlılığı): pip install -e ".[dev]"
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

from cvflair import available_themes, get_theme

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "demo.gif"
WIDTH, HEIGHT = 480, 270
FRAMES_PER_THEME = 20
FRAME_MS = 80

OBJECT_FILL = ((92, 96, 104), (104, 92, 96))
CLASS_NAMES = ("kisi", "top")


def scene() -> np.ndarray:
    """Koyu, hafif degradeli bir zemin — sabit, sadece nesneler hareket eder."""
    column = np.linspace(26, 52, WIDTH, dtype=np.uint8)
    return np.repeat(column[None, :, None], HEIGHT, axis=0).repeat(3, axis=2).copy()


def objects(tick: int) -> tuple[np.ndarray, np.ndarray]:
    """Sahnedeki iki nesnenin kutuları (xyxy). Tespitler bunlardan türetilir."""
    phase = 2 * math.pi * tick / (FRAMES_PER_THEME * len(available_themes()))

    person_w, person_h = 62, 130
    person_x = 40 + (math.sin(phase) + 1) / 2 * (WIDTH * 0.42)
    person_y = HEIGHT - person_h - 40

    ball_r = 26
    ball_x = WIDTH * 0.74
    ball_y = 60 + (math.sin(phase * 1.6) + 1) / 2 * (HEIGHT * 0.45)

    person = np.array([person_x, person_y, person_x + person_w, person_y + person_h])
    ball = np.array([ball_x - ball_r, ball_y - ball_r, ball_x + ball_r, ball_y + ball_r])
    return person, ball


def draw_objects(frame: np.ndarray, person: np.ndarray, ball: np.ndarray) -> None:
    """Nesneleri sahneye çizer; kutular bunların üstüne oturur."""
    x1, y1, x2, y2 = person.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_FILL[0], thickness=-1)
    center = ((ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2)
    radius = int((ball[2] - ball[0]) / 2)
    cv2.circle(frame, (int(center[0]), int(center[1])), radius, OBJECT_FILL[1], thickness=-1)


def main() -> None:
    themes = available_themes()
    frames: list[Image.Image] = []

    for index, name in enumerate(themes):
        theme = get_theme(name)
        for step in range(FRAMES_PER_THEME):
            tick = index * FRAMES_PER_THEME + step
            frame = scene()
            person, ball = objects(tick)
            draw_objects(frame, person, ball)

            confidence = np.array(
                [0.90 + 0.08 * math.sin(tick / 5), 0.70 + 0.20 * math.sin(tick / 3)],
                dtype=np.float32,
            )
            detections = sv.Detections(
                xyxy=np.vstack([person, ball]).astype(np.float32),
                class_id=np.array([0, 1]),
                confidence=confidence,
            )
            labels = [f"{n} {c:.2f}" for n, c in zip(CLASS_NAMES, confidence, strict=True)]
            theme.annotate(frame, detections, labels=labels)

            cv2.putText(
                frame,
                f"tema: {name}",
                (14, HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    OUTPUT.parent.mkdir(exist_ok=True)
    palette = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=96) for frame in frames]
    palette[0].save(
        OUTPUT,
        save_all=True,
        append_images=palette[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"{OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB, {len(frames)} kare)")


if __name__ == "__main__":
    main()
