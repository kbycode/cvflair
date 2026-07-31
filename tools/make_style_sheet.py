"""
README ve belgelerdeki görselleri üretir:

* `docs/box-styles.png` — aynı tespit, sekiz çerçeve biçimi yan yana.
* `docs/theme-<ad>.png` — her tema, kendi çerçeve biçimiyle, tam kare.
* `docs/skeletons.png` — el ve poz iskeleti, her temada.

Çalıştırmak için:  python tools/make_style_sheet.py
Kaynak `docs/city.png` ve kutu oranları `tools/make_demo_gif.py` ile ortak.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from make_demo_gif import BACKGROUND, DETECTIONS, fit_to_frame, frame_boxes

from cvflair import HAND_21, POSE_17, Detections, KeyPoints, Theme, available_themes, get_theme
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
    # Panelli temalar için örnek sayaçlar; panelsiz temalarda yok sayılıyor.
    stats = {"FPS": 30, "Objects": len(DETECTIONS)}
    get_theme(name).annotate(frame, detections, labels=labels, stats=stats)
    return frame


#: Sentetik el ve poz: gerçek bir modele gerek kalmadan topolojiyi göstermek için.
FINGERS = ((-1.35, 0.62), (-0.42, 1.0), (-0.12, 1.08), (0.18, 1.0), (0.5, 0.82))


def hand_points(cx: float, cy: float, curl: float = 0.35) -> np.ndarray:
    points = [(cx, cy)]
    for angle, length in FINGERS:
        x = cx + np.sin(angle) * 46 * length
        y = cy - np.cos(angle) * 46 * length
        heading = angle
        points.append((x, y))
        for joint in range(3):
            heading += curl * 0.75
            x += np.sin(heading) * (22 - joint * 2) * length
            y -= np.cos(heading) * (22 - joint * 2) * length
            points.append((x, y))
    return np.array(points, dtype=np.float32)


def pose_points(cx: float, cy: float) -> np.ndarray:
    return np.array([
        (cx, cy - 152),
        (cx - 8, cy - 159), (cx + 8, cy - 159),
        (cx - 17, cy - 153), (cx + 17, cy - 153),
        (cx - 28, cy - 124), (cx + 28, cy - 124),
        (cx - 40, cy - 88), (cx + 40, cy - 88),
        (cx - 44, cy - 54), (cx + 44, cy - 54),
        (cx - 18, cy - 76), (cx + 18, cy - 76),
        (cx - 22, cy - 40), (cx + 14, cy - 40),
        (cx - 26, cy - 2), (cx + 10, cy - 2),
    ], dtype=np.float32)


def skeleton_cell(theme_name: str) -> np.ndarray:
    width, height = 380, 300
    column = np.linspace(26, 52, width, dtype=np.uint8)
    frame = np.repeat(column[None, :, None], height, axis=0).repeat(3, axis=2).copy()

    theme = get_theme(theme_name)
    theme.annotate_keypoints(frame, KeyPoints(xy=hand_points(105, 215)), HAND_21)
    theme.annotate_keypoints(frame, KeyPoints(xy=pose_points(275, 190), class_id=[1]), POSE_17)
    cv2.putText(
        frame, theme_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (230, 230, 230), 1, cv2.LINE_AA,
    )
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

    skeletons = [skeleton_cell(name) for name in available_themes()]
    while len(skeletons) % 3:
        skeletons.append(np.full_like(skeletons[0], 18))
    rows = [np.hstack(skeletons[i : i + 3]) for i in range(0, len(skeletons), 3)]
    path = DOCS / "skeletons.png"
    cv2.imwrite(str(path), np.vstack(rows))
    print(f"{path}")


if __name__ == "__main__":
    main()
