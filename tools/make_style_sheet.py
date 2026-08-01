"""
README ve belgelerdeki görselleri üretir:

* `docs/box-styles.png` — aynı tespit, bütün çerçeve biçimleri yan yana.
* `docs/theme-<ad>.png` — her tema, kendi çerçeve biçimiyle, tam kare.
* `docs/skeletons.png` — el, poz ve yüz iskeleti, her temada.
* `docs/motion.png` — nabız halkasının evreleri ve takip izi.

Çalıştırmak için:  python tools/make_style_sheet.py
Kaynak `docs/city.png` ve kutu oranları `tools/make_demo_gif.py` ile ortak.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from make_demo_gif import BACKGROUND, DETECTIONS, fit_to_frame, frame_boxes

from cvflair import (
    FACE_5,
    HAND_21,
    POSE_17,
    Detections,
    KeyPoints,
    Theme,
    available_themes,
    get_theme,
)
from cvflair.themes import BOX_STYLES

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUTPUT = DOCS / "box-styles.png"
COLUMNS = 3  # dokuz biçim tam üç satır
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


def face_points(cx: float, cy: float, scale: float) -> np.ndarray:
    """Beş nokta, yüz modellerinin sırasıyla: göz, göz, burun, ağız, ağız."""
    layout = ((-0.22, -0.20), (0.22, -0.20), (0.0, 0.02), (-0.16, 0.28), (0.16, 0.28))
    return np.array([[cx + dx * scale, cy + dy * scale] for dx, dy in layout])


#: Sahnedeki figürlerin grisi; playground'daki FIGURE ile aynı ton.
FIGURE = (178, 161, 152)
FIGURE_DARK = (136, 116, 108)


def draw_subject(frame: np.ndarray, points: np.ndarray, edges, weight: int) -> None:
    """
    İskeletin altına aynı geometriden kalın bir kütle çizer.

    OpenCV çizgileri düz uçlu; eklemlere daire konmazsa parmak uçları ve dirsekler
    köşeli kalıyor, kütle çizim gibi değil şema gibi görünüyor.
    """
    for first, second in edges:
        cv2.line(
            frame, tuple(points[first].astype(int)), tuple(points[second].astype(int)),
            FIGURE, weight, cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(frame, tuple(point.astype(int)), weight // 2, FIGURE, -1, cv2.LINE_AA)


def fill_blob(frame: np.ndarray, points: np.ndarray, indices, weight: int) -> None:
    """Verilen noktaların dışbükey örtüsünü kalın konturla doldurur: köşeler yuvarlanır."""
    hull = cv2.convexHull(points[list(indices)].astype(np.int32))
    cv2.fillPoly(frame, [hull], FIGURE, cv2.LINE_AA)
    cv2.polylines(frame, [hull], True, FIGURE, weight, cv2.LINE_AA)
    for point in hull.reshape(-1, 2):
        cv2.circle(frame, tuple(point), weight // 2, FIGURE, -1, cv2.LINE_AA)


#: Baş ölçüleri göz aralığının katı olarak. İnsan yüzünde gözler arası mesafe
#: kafa genişliğinin kabaca yarısı; daha dar tutulunca gözler kulaklara,
#: ağız da çenenin altına düşüyor.
HEAD_WIDTH = 1.15
HEAD_HEIGHT = 1.45
HEAD_DROP = 0.12       # göz çizgisi kafa merkezinin biraz üstünde
SHOULDER_DROP = 2.2


def draw_face_subject(frame: np.ndarray, points: np.ndarray) -> None:
    """Baş ve omuz: beş nokta tek başına yüz olduğunu anlatmıyor."""
    left, right = points[0], points[1]
    span = float(np.hypot(*(right - left)))
    angle = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))

    along = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    down = np.array([-along[1], along[0]])
    eyes = (left + right) / 2
    center = eyes + down * span * HEAD_DROP

    shoulder = tuple((eyes + down * span * SHOULDER_DROP).astype(int))
    cv2.ellipse(frame, shoulder, (int(span * 1.8), int(span * 0.8)),
                angle, 180, 360, FIGURE_DARK, -1, cv2.LINE_AA)
    for side in (-1, 1):
        ear = tuple((center + along * side * span * HEAD_WIDTH).astype(int))
        cv2.ellipse(frame, ear, (int(span * 0.16), int(span * 0.28)),
                    angle, 0, 360, FIGURE, -1, cv2.LINE_AA)
    cv2.ellipse(frame, tuple(center.astype(int)),
                (int(span * HEAD_WIDTH), int(span * HEAD_HEIGHT)),
                angle, 0, 360, FIGURE, -1, cv2.LINE_AA)


def skeleton_cell(theme_name: str) -> np.ndarray:
    width, height = 380, 300
    column = np.linspace(26, 52, width, dtype=np.uint8)
    frame = np.repeat(column[None, :, None], height, axis=0).repeat(3, axis=2).copy()

    theme = get_theme(theme_name)
    hand = hand_points(78, 232)
    pose = pose_points(206, 205)
    face = face_points(312, 128, 62)

    # Siluetler önce: çizim bir gövdenin üstünde dursun, boşlukta kalmasın.
    draw_subject(frame, hand, HAND_21, 15)
    fill_blob(frame, hand, (0, 1, 5, 9, 13, 17), 15)
    draw_subject(frame, pose, POSE_17, 17)
    fill_blob(frame, pose, (5, 6, 12, 11), 17)
    cv2.circle(frame, tuple(pose[0].astype(int)),
               int(np.hypot(*(pose[5] - pose[6])) * 0.42), FIGURE, -1, cv2.LINE_AA)
    draw_face_subject(frame, face)

    theme.annotate_keypoints(frame, KeyPoints(xy=hand[None, ...]), HAND_21)
    theme.annotate_keypoints(frame, KeyPoints(xy=pose[None, ...], class_id=[1]), POSE_17)
    theme.annotate_keypoints(frame, KeyPoints(xy=face[None, ...], class_id=[2]), FACE_5)
    cv2.putText(
        frame, theme_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (230, 230, 230), 1, cv2.LINE_AA,
    )
    return frame


def pad_to_grid(cells: list[np.ndarray], columns: int) -> np.ndarray:
    """Son satırı boş hücreyle tamamlar; eksikse `vstack` genişlik uyuşmazlığı verir."""
    cells = list(cells)  # çağıranın listesi büyümesin
    while len(cells) % columns:
        cells.append(np.full_like(cells[0], 24))
    rows = [np.hstack(cells[index : index + columns]) for index in range(0, len(cells), columns)]
    return np.vstack(rows)


def pulse_cell(moment: float) -> np.ndarray:
    """Nabız halkasının tek bir anı; şerit hâlinde döngü okunur hâle geliyor."""
    width, height = 240, 220
    frame = np.full((height, width, 3), 22, dtype=np.uint8)
    theme = Theme(
        box_style="corner", palette=BASE, thickness=3, corner_length=22,
        pulse=True, pulse_reach=22, pulse_speed=1.0, labels=False,
    )
    theme.annotate(
        frame,
        Detections(xyxy=[[60, 50, 180, 170]], class_id=[0], confidence=[0.94]),
        moment=moment,
    )
    cv2.putText(
        frame, f"{moment:.2f}", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (170, 170, 170), 1, cv2.LINE_AA,
    )
    return frame


def trace_cell(width: int = 960) -> np.ndarray:
    """Kırk kare boyunca dolaşan bir nesnenin bıraktığı iz."""
    height = 220
    frame = np.full((height, width, 3), 22, dtype=np.uint8)
    theme = Theme(
        box_style="round", palette=BASE, accent_palette=ACCENT, thickness=3,
        trace=True, trace_length=36, labels=False,
    )
    for step in range(40):
        left = 40 + step * (width - 140) // 40
        top = 95 - int(45 * np.sin(step / 7))  # iz kare içinde kalsın
        # Ara kareler yalnızca izi besliyor; ekrana yalnız sonuncusu çiziliyor.
        canvas = frame if step == 39 else frame.copy()
        theme.annotate(
            canvas,
            Detections(
                xyxy=[[left, top, left + 56, top + 56]],
                class_id=[0], confidence=[0.9], tracker_id=[1],
            ),
        )
    cv2.putText(
        frame, "trace", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (170, 170, 170), 1, cv2.LINE_AA,
    )
    return frame


def main() -> None:
    source = cv2.imread(str(BACKGROUND))
    if source is None:
        raise SystemExit(f"Kaynak görsel okunamadı: {BACKGROUND}")

    cells = [cell(style, source) for style in BOX_STYLES]
    DOCS.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT), pad_to_grid(cells, COLUMNS))
    print(f"{OUTPUT}  ({len(cells)} biçim)")

    for name in available_themes():
        path = DOCS / f"theme-{name}.png"
        cv2.imwrite(str(path), theme_preview(name, source))
        print(f"{path}")

    skeletons = [skeleton_cell(name) for name in available_themes()]
    path = DOCS / "skeletons.png"
    cv2.imwrite(str(path), pad_to_grid(skeletons, 3))
    print(f"{path}")

    phases = np.hstack([pulse_cell(moment) for moment in (0.05, 0.35, 0.65, 0.95)])
    motion = np.vstack([phases, trace_cell(phases.shape[1])])
    path = DOCS / "motion.png"
    cv2.imwrite(str(path), motion)
    print(f"{path}")


if __name__ == "__main__":
    main()
