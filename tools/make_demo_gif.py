"""
README'deki demo GIF'ini üretir: gerçek bir sokak fotoğrafı üzerine cvflair
kutuları, sırayla değişen temalarla.

Çalıştırmak için:  python tools/make_demo_gif.py
Başka bir görselle: python tools/make_demo_gif.py --background yol/gorsel.jpg
Çıktı:             docs/demo.gif

Kutu koordinatları `docs/city.png` için elle ayarlandı ve kaynak görselin
oranlarına göre saklanıyor; kırpma ile ölçekleme sırasında birlikte taşınıyor.
Başka bir arka plan verilirse kutular oturmaz, `DETECTIONS` yeniden ayarlanmalı.

Kareden kareye küçük bir salınım var: gerçek bir tespit modelinin kutusu da
sabit durmaz, ayrıca bu hareket GIF'in donmuş görünmesini engelliyor.

Pillow gerekir (dev bağımlılığı): pip install -e ".[dev]"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cvflair import Detections, Theme, available_themes, get_theme

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = ROOT / "docs" / "city.png"
OUTPUT = ROOT / "docs" / "demo.gif"

WIDTH, HEIGHT = 640, 360
FRAMES_PER_THEME = 16
FRAME_MS = 90

#: Kaynak görseldeki nesneler: (etiket, x1, y1, x2, y2) — 0-1 aralığında.
DETECTIONS = (
    ("bisiklet", 0.103, 0.648, 0.328, 0.945),
    ("kopek", 0.604, 0.617, 0.672, 0.748),
    ("kisi", 0.812, 0.447, 0.906, 0.836),
)

#: Çerçeve ayarları; temalar yalnızca renk ve etiket biçimiyle ayrışır.
BOX_SETTINGS = {
    "box_style": "dashed_corner",
    "thickness": 2,
    "corner_length": 8,
    "dash_length": 5,
    "gap_length": 7,
    "accent_palette": None,
}


def fit_to_frame(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """
    Görseli 16:9'a ortadan kırpıp kare boyutuna ölçekler.

    Kutuların birlikte taşınabilmesi için kırpma başlangıcı ve ölçek de döner.
    """
    height, width = image.shape[:2]
    target = WIDTH / HEIGHT

    if width / height > target:
        crop_width = int(round(height * target))
        left = (width - crop_width) // 2
        top = 0
        image = image[:, left : left + crop_width]
    else:
        crop_height = int(round(width / target))
        left = 0
        top = (height - crop_height) // 2
        image = image[top : top + crop_height, :]

    scale = WIDTH / image.shape[1]
    resized = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return resized, left, top, scale


def frame_boxes(
    source_size: tuple[int, int], left: float, top: float, scale: float, time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Kaynak orandaki kutuları kare koordinatlarına taşır ve hafifçe oynatır."""
    source_width, source_height = source_size
    boxes = []
    confidences = []

    for index, (_, x1, y1, x2, y2) in enumerate(DETECTIONS):
        phase = index * 2.1
        # Kenar başına farklı fazda küçük salınım.
        wobble = [math.sin(time * 3.1 + phase + corner) * 1.6 for corner in range(4)]
        boxes.append(
            [
                (x1 * source_width - left) * scale + wobble[0],
                (y1 * source_height - top) * scale + wobble[1],
                (x2 * source_width - left) * scale + wobble[2],
                (y2 * source_height - top) * scale + wobble[3],
            ]
        )
        confidences.append(0.86 + 0.11 * math.sin(time * 1.7 + phase))

    return np.array(boxes, dtype=np.float32), np.array(confidences, dtype=np.float32)


def styled(name: str) -> Theme:
    """Temanın rengini koru, çerçeve ayarlarını ortak hale getir."""
    return replace(get_theme(name), **BOX_SETTINGS)


def caption(frame: np.ndarray, text: str) -> None:
    """Koyu konturlu yazı: hem açık hem koyu arka planda okunuyor."""
    for colour, thickness in (((20, 20, 20), 3), ((245, 245, 245), 1)):
        cv2.putText(
            frame, text, (14, HEIGHT - 14), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, colour, thickness, cv2.LINE_AA,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", type=Path, default=BACKGROUND)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    source = cv2.imread(str(args.background))
    if source is None:
        raise SystemExit(f"Arka plan okunamadı: {args.background}")

    source_size = (source.shape[1], source.shape[0])
    backdrop, left, top, scale = fit_to_frame(source)

    themes = available_themes()
    frames: list[Image.Image] = []

    for index, name in enumerate(themes):
        theme = styled(name)
        for step in range(FRAMES_PER_THEME):
            time = (index * FRAMES_PER_THEME + step) / 12
            frame = backdrop.copy()

            xyxy, confidences = frame_boxes(source_size, left, top, scale, time)
            detections = Detections(
                xyxy=xyxy,
                class_id=np.arange(len(DETECTIONS)),
                confidence=confidences,
            )
            labels = [
                f"{item[0]} {score:.2f}"
                for item, score in zip(DETECTIONS, confidences, strict=True)
            ]
            theme.annotate(frame, detections, labels=labels)
            caption(frame, f"tema: {name}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    # Tek ortak palet: kare başına palet, kareler arası farkı bozup dosyayı
    # şişiriyor. Palet her temadan birer kareye bakılarak çıkarılıyor, yoksa
    # ilk temada bulunmayan renkler en yakın tona düşüyor.
    sample = Image.new("RGB", (WIDTH, HEIGHT * len(themes)))
    for index in range(len(themes)):
        middle = index * FRAMES_PER_THEME + FRAMES_PER_THEME // 2
        sample.paste(frames[middle], (0, index * HEIGHT))
    base = sample.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    paletted = [frame.quantize(palette=base, dither=Image.Dither.NONE) for frame in frames]

    args.output.parent.mkdir(exist_ok=True)
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
