"""
README'deki iki görseli üretir:

* `docs/box-styles.png` — aynı tespit, sekiz çerçeve biçimi yan yana.
* `docs/theme-<ad>.png` — her tema, kendi çerçeve biçimiyle, tam kare.

Çalıştırmak için:  python tools/make_style_sheet.py
Kaynak `docs/city.png` ve kutu oranları `tools/make_demo_gif.py` ile ortak.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from make_demo_gif import BACKGROUND, DETECTIONS, fit_to_frame, frame_boxes

from cvflair import Detections, Theme, available_themes, get_theme
from cvflair.themes import BOX_STYLES

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUTPUT = DOCS / "box-styles.png"
COLUMNS = 4
CAPTION_H = 26
PAD = 26

BASE = ["#00F0FF"]
ACCENT = ["#FF206E"]

#: İkinci rengi kullanan biçimler; diğerlerinde vurgu ayarı görünmez.
ACCENT_STYLES = ("dashed_corner", "bracket", "crosshair", "target")

#: Karşılaştırma tek nesne üzerinden: bisiklet, geniş ve ayrıntılı.
SUBJECT = DETECTIONS[0]


def cell(style: str, source: np.ndarray) -> np.ndarray:
    label, x1, y1, x2, y2 = SUBJECT
    height, width = source.shape[:2]
    box = np.array([x1 * width, y1 * height, x2 * width, y2 * height])

    left = int(max(0, box[0] - PAD * 2))
    top = int(max(0, box[1] - PAD * 3))
    right = int(min(width, box[2] + PAD * 2))
    bottom = int(min(height, box[3] + PAD))
    crop = source[top:bottom, left:right].copy()

    theme = Theme(
        palette=BASE,
        accent_palette=ACCENT if style in ACCENT_STYLES else None,
        box_style=style,
        thickness=3,
        corner_length=22,
        text_color="#000000",
        text_scale=0.5,
        text_padding=6,
    )
    theme.annotate(
        crop,
        Detections(
            xyxy=[[box[0] - left, box[1] - top, box[2] - left, box[3] - top]],
            class_id=[0],
            confidence=[0.94],
        ),
        labels=[f"{label} 0.94"],
    )

    scaled = cv2.resize(crop, (360, 260), interpolation=cv2.INTER_AREA)
    caption = np.full((CAPTION_H, 360, 3), 24, dtype=np.uint8)
    cv2.putText(
        caption, style, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA
    )
    return np.vstack([scaled, caption])


def theme_preview(name: str, source: np.ndarray) -> np.ndarray:
    """Tam kare, temanın kendi çerçeve biçimiyle."""
    frame, left, top, scale = fit_to_frame(source)
    xyxy, confidences = frame_boxes((source.shape[1], source.shape[0]), left, top, scale, 0.0)
    detections = Detections(
        xyxy=xyxy, class_id=np.arange(len(DETECTIONS)), confidence=confidences
    )
    labels = [f"{item[0]} {score:.2f}" for item, score in zip(DETECTIONS, confidences, strict=True)]
    get_theme(name).annotate(frame, detections, labels=labels)
    return frame


def main() -> None:
    source = cv2.imread(str(BACKGROUND))
    if source is None:
        raise SystemExit(f"Kaynak görsel okunamadı: {BACKGROUND}")

    cells = [cell(style, source) for style in BOX_STYLES]
    rows = [np.hstack(cells[i : i + COLUMNS]) for i in range(0, len(cells), COLUMNS)]
    DOCS.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT), np.vstack(rows))
    print(f"{OUTPUT}  ({len(cells)} biçim)")

    for name in available_themes():
        path = DOCS / f"theme-{name}.png"
        cv2.imwrite(str(path), theme_preview(name, source))
        print(f"{path}")


if __name__ == "__main__":
    main()
