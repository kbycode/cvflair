"""
README'deki demo GIF'ini üretir: bir sokak sahnesi, hareket eden üç nesne ve
sırayla değişen temalar. Kamera gerektirmez, çıktısı her çalıştırmada aynıdır.

Çalıştırmak için:  python tools/make_demo_gif.py
Kendi arka planınla: python tools/make_demo_gif.py --background docs/sokak.jpg
Çıktı:             docs/demo.gif

Arka plan verilmezse sahne çizilerek üretilir (gökyüzü, silüet, yol, sokak
lambaları). Verilirse görsel 16:9'a kırpılıp ölçeklenir; figürler ve kutular
üstüne çizilir. Pillow gerekir (dev bağımlılığı): pip install -e ".[dev]"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cvflair import Detections, available_themes, get_theme

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "demo.gif"
WIDTH, HEIGHT = 480, 270
GROUND = 0.88
FRAMES_PER_THEME = 18
FRAME_MS = 80

FIGURE = (178, 161, 152)       # BGR
FIGURE_DARK = (136, 116, 108)


# -- sahne ------------------------------------------------------------------


def sky(width: int, height: int) -> np.ndarray:
    """Alacakaranlık geçişi: üstte lacivert, ufka doğru sıcak turuncu."""
    top = np.array([54, 34, 24], dtype=np.float32)
    bottom = np.array([120, 132, 176], dtype=np.float32)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None] ** 2.2
    column = top + (bottom - top) * ramp
    return np.repeat(column[:, None, :], width, axis=1).astype(np.uint8)


def skyline(frame: np.ndarray, rng: np.random.Generator, horizon: int) -> None:
    """İki katmanlı bina silüeti; arkadaki daha açık, öndeki koyu."""
    for layer, (colour, min_h, max_h, step) in enumerate(
        ((( 78, 62, 52), 0.22, 0.44, 46), ((46, 36, 32), 0.30, 0.58, 62))
    ):
        x = -20
        while x < WIDTH:
            block_w = step + int(rng.integers(-12, 14))
            block_h = int(HEIGHT * rng.uniform(min_h, max_h))
            top = horizon - block_h
            cv2.rectangle(frame, (x, top), (x + block_w, horizon), colour, -1)

            # Pencereler: bazıları yanık, bazıları karanlık.
            for wy in range(top + 8, horizon - 8, 12):
                for wx in range(x + 7, x + block_w - 7, 11):
                    if rng.random() < (0.22 if layer else 0.14):
                        glow = (150, 205, 235) if rng.random() < 0.75 else (110, 150, 205)
                        cv2.rectangle(frame, (wx, wy), (wx + 4, wy + 6), glow, -1)
            x += block_w + int(rng.integers(2, 10))


def street(frame: np.ndarray, ground_y: int) -> None:
    """Kaldırım, asfalt ve orta çizgi."""
    cv2.rectangle(frame, (0, ground_y - 16), (WIDTH, ground_y), (74, 70, 68), -1)
    cv2.rectangle(frame, (0, ground_y), (WIDTH, HEIGHT), (48, 46, 46), -1)
    cv2.line(frame, (0, ground_y), (WIDTH, ground_y), (96, 92, 90), 1)

    lane_y = ground_y + (HEIGHT - ground_y) // 2
    for x in range(-10, WIDTH, 46):
        cv2.line(frame, (x, lane_y), (x + 24, lane_y), (120, 122, 124), 2)


def street_lamps(frame: np.ndarray, ground_y: int) -> None:
    for x in (86, 300, 500):
        top = ground_y - 96
        cv2.line(frame, (x, ground_y - 16), (x, top), (70, 66, 64), 3)
        cv2.line(frame, (x, top), (x + 20, top), (70, 66, 64), 3)
        cv2.circle(frame, (x + 22, top + 2), 4, (170, 210, 240), -1)
        halo = frame.copy()
        cv2.circle(halo, (x + 22, top + 2), 18, (120, 170, 215), -1)
        cv2.addWeighted(halo, 0.10, frame, 0.90, 0, frame)


def build_backdrop(background: Path | None) -> np.ndarray:
    ground_y = int(HEIGHT * GROUND)
    if background is not None:
        image = cv2.imread(str(background))
        if image is None:
            raise SystemExit(f"Arka plan okunamadı: {background}")
        return fit_to_frame(image)

    rng = np.random.default_rng(7)
    frame = sky(WIDTH, HEIGHT)
    skyline(frame, rng, ground_y - 16)
    street(frame, ground_y)
    street_lamps(frame, ground_y)
    return frame


def fit_to_frame(image: np.ndarray) -> np.ndarray:
    """Görseli 16:9'a ortadan kırpıp kare boyutuna ölçekler."""
    height, width = image.shape[:2]
    target = WIDTH / HEIGHT
    if width / height > target:
        new_width = int(height * target)
        start = (width - new_width) // 2
        image = image[:, start : start + new_width]
    else:
        new_height = int(width / target)
        start = (height - new_height) // 2
        image = image[start : start + new_height, :]
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


# -- figürler ---------------------------------------------------------------


class Pen:
    """Yerel figür koordinatlarını sahneye taşıyan küçük yardımcı."""

    def __init__(self, frame: np.ndarray, x: float, ground_y: float, scale: float, facing: int):
        self.frame = frame
        self.x = x
        self.ground_y = ground_y
        self.scale = scale
        self.facing = facing

    def point(self, px: float, py: float) -> tuple[int, int]:
        return (
            int(round(self.x + self.facing * px * self.scale)),
            int(round(self.ground_y + py * self.scale)),
        )

    def line(self, a: tuple[float, float], b: tuple[float, float], width: float, colour) -> None:
        cv2.line(
            self.frame, self.point(*a), self.point(*b), colour,
            max(1, int(round(width * self.scale))), cv2.LINE_AA,
        )

    def dot(self, centre: tuple[float, float], radius: float, colour) -> None:
        cv2.circle(
            self.frame, self.point(*centre), max(1, int(round(radius * self.scale))),
            colour, -1, cv2.LINE_AA,
        )

    def ring(self, centre: tuple[float, float], radius: float, width: float, colour) -> None:
        cv2.circle(
            self.frame, self.point(*centre), max(1, int(round(radius * self.scale))),
            colour, max(1, int(round(width * self.scale))), cv2.LINE_AA,
        )


def draw_person(pen: Pen, time: float) -> None:
    swing = math.sin(time * 3.6)
    pen.line((-5, -72), (-5 - swing * 20, -1), 12, FIGURE_DARK)
    pen.line((-11, -112), (-11 - swing * 17, -82), 9, FIGURE_DARK)
    pen.line((0, -113), (0, -74), 24, FIGURE)
    pen.line((-11, -114), (11, -114), 10, FIGURE)
    pen.line((0, -122), (0, -116), 8, FIGURE)
    pen.line((5, -72), (5 + swing * 20, -1), 12, FIGURE)
    pen.line((11, -112), (11 + swing * 17, -82), 9, FIGURE)
    pen.dot((1, -136), 13, FIGURE)


def draw_bicycle(pen: Pen, time: float) -> None:
    spin = time * 5.2
    for hub in (-46, 46):
        pen.ring((hub, -31), 30, 4, FIGURE)
        for i in range(6):
            angle = spin + i * math.pi / 3
            pen.line(
                (hub, -31),
                (hub + math.cos(angle) * 30, -31 + math.sin(angle) * 30),
                2, FIGURE_DARK,
            )
    for a, b in (
        ((-46, -31), (-8, -70)), ((-46, -31), (6, -31)), ((-8, -70), (6, -31)),
        ((-8, -70), (34, -68)), ((34, -68), (46, -31)), ((6, -31), (34, -68)),
    ):
        pen.line(a, b, 5, FIGURE)
    pen.line((28, -74), (44, -74), 4, FIGURE)
    pen.line((-16, -73), (-2, -73), 6, FIGURE)


def draw_dog(pen: Pen, time: float) -> None:
    step = math.sin(time * 4.8)
    tail = math.sin(time * 11) * 9
    pen.line((-24, -32), (-24 - step * 13, -1), 7, FIGURE_DARK)
    pen.line((26, -32), (26 - step * 13, -1), 7, FIGURE_DARK)
    pen.line((-18, -32), (-18 + step * 13, -1), 7, FIGURE)
    pen.line((32, -32), (32 + step * 13, -1), 7, FIGURE)
    pen.line((-30, -44), (-46, -50 + tail), 6, FIGURE)
    pen.line((-24, -44), (26, -44), 30, FIGURE)
    pen.line((28, -46), (42, -62), 13, FIGURE)
    pen.dot((44, -66), 12, FIGURE)
    pen.line((48, -64), (60, -64), 8, FIGURE)


ACTORS = (
    {
        "label": "kisi",
        "draw": draw_person,
        "bounds": (-32, -152, 32, 3),
        "scale": 1.0,
        "lift": 0,
        "path": lambda t: (0.17 + 0.10 * math.sin(t * 0.5), math.cos(t * 0.5) >= 0),
        "confidence": lambda t: 0.90 + 0.08 * math.sin(t * 1.6),
    },
    {
        "label": "bisiklet",
        "draw": draw_bicycle,
        "bounds": (-80, -84, 80, 3),
        "scale": 0.82,
        # Yolda, kaldırımdaki figürlerin önünde: kendi şeridi olduğu için
        # onlarla sürekli çakışmıyor.
        "lift": 20,
        "path": lambda t: (1.25 - ((t * 0.16) % 1.55), False),
        "confidence": lambda t: 0.78 + 0.14 * math.sin(t * 2.1 + 1),
    },
    {
        "label": "kopek",
        "draw": draw_dog,
        "bounds": (-54, -82, 66, 3),
        "scale": 0.85,
        "lift": 0,
        "path": lambda t: (0.72 + 0.14 * math.sin(t * 0.75 + 2), math.cos(t * 0.75 + 2) >= 0),
        "confidence": lambda t: 0.66 + 0.20 * math.sin(t * 2.6),
    },
)


def actor_frame(actor: dict, time: float) -> dict:
    position, forward = actor["path"](time)
    facing = 1 if forward else -1
    x = position * WIDTH
    lane = HEIGHT * GROUND + actor["lift"]
    scale = actor["scale"]

    min_x, min_y, max_x, max_y = actor["bounds"]
    if facing < 0:
        min_x, max_x = -max_x, -min_x

    return {
        "actor": actor,
        "x": x,
        "lane": lane,
        "facing": facing,
        "scale": scale,
        "box": (
            x + min_x * scale, lane + min_y * scale,
            x + max_x * scale, lane + max_y * scale,
        ),
    }


def draw_actors(frame: np.ndarray, placements: list[dict]) -> None:
    for placement in sorted(placements, key=lambda item: item["lane"]):
        actor = placement["actor"]
        width = (actor["bounds"][2] - actor["bounds"][0]) * placement["scale"]

        shadow = frame.copy()
        cv2.ellipse(
            shadow, (int(placement["x"]), int(placement["lane"] + 2)),
            (int(width * 0.34), 5), 0, 0, 360, (0, 0, 0), -1,
        )
        cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)

        pen = Pen(frame, placement["x"], placement["lane"], placement["scale"], placement["facing"])
        actor["draw"](pen, placement["time"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background", type=Path, default=None,
        help="Sahne yerine kullanılacak arka plan görseli (16:9'a kırpılır).",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    backdrop = build_backdrop(args.background)
    themes = available_themes()
    frames: list[Image.Image] = []

    for index, name in enumerate(themes):
        theme = get_theme(name)
        for step in range(FRAMES_PER_THEME):
            time = (index * FRAMES_PER_THEME + step) / 14
            frame = backdrop.copy()

            placements = [actor_frame(actor, time) for actor in ACTORS]
            for placement in placements:
                placement["time"] = time
            draw_actors(frame, placements)

            boxes = np.array([placement["box"] for placement in placements], dtype=np.float32)
            confidences = np.array(
                [placement["actor"]["confidence"](time) for placement in placements],
                dtype=np.float32,
            )
            detections = Detections(
                xyxy=boxes,
                class_id=np.arange(len(placements)),
                confidence=confidences,
            )
            labels = [
                f"{placement['actor']['label']} {score:.2f}"
                for placement, score in zip(placements, confidences, strict=True)
            ]
            theme.annotate(frame, detections, labels=labels)

            cv2.putText(
                frame, f"tema: {name}", (14, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (225, 225, 225), 1, cv2.LINE_AA,
            )
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    args.output.parent.mkdir(exist_ok=True)
    # Kare başına ayrı palet kareler arası farkı bozup dosyayı şişiriyor; tek
    # ortak palet kullanılıyor. Palet her temadan birer kareye bakılarak
    # çıkarılıyor, yoksa ilk temada bulunmayan renkler en yakın tona düşüyor.
    sample = Image.new("RGB", (WIDTH, HEIGHT * len(themes)))
    for index in range(len(themes)):
        middle = index * FRAMES_PER_THEME + FRAMES_PER_THEME // 2
        sample.paste(frames[middle], (0, index * HEIGHT))
    base = sample.convert("P", palette=Image.Palette.ADAPTIVE, colors=96)
    paletted = [frame.quantize(palette=base, dither=Image.Dither.NONE) for frame in frames]
    paletted[0].save(
        args.output,
        save_all=True,
        append_images=paletted[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=True,
    )
    print(f"{args.output}  ({args.output.stat().st_size / 1024:.0f} KB, {len(frames)} kare)")


if __name__ == "__main__":
    main()
