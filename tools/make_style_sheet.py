"""
README'deki çerçeve biçimleri tablosunun görselini üretir.

Çalıştırmak için:  python tools/make_style_sheet.py
Çıktı:             docs/box-styles.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from cvflair import Theme
from cvflair.themes import BOX_STYLES

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "box-styles.png"
CELL_W, CELL_H = 300, 200
COLUMNS = 4

BASE = sv.ColorPalette.from_hex(["#00F0FF"])
ACCENT = sv.ColorPalette.from_hex(["#FF206E"])

DETECTIONS = sv.Detections(
    xyxy=np.array([[45, 55, 195, 165]], dtype=np.float32),
    class_id=np.array([0]),
    confidence=np.array([0.92], dtype=np.float32),
)


def cell(style: str, accent: bool) -> np.ndarray:
    column = np.linspace(26, 52, CELL_W, dtype=np.uint8)
    frame = np.repeat(column[None, :, None], CELL_H, axis=0).repeat(3, axis=2).copy()

    theme = Theme(
        palette=BASE,
        accent_palette=ACCENT if accent else None,
        box_style=style,
        thickness=3,
        text_color=sv.Color.BLACK,
        text_scale=0.4,
        text_padding=5,
    )
    theme.annotate(frame, DETECTIONS, labels=["kisi 0.92"])
    cv2.putText(
        frame,
        style + (" + accent" if accent else ""),
        (12, CELL_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    cells = [cell(style, accent=False) for style in BOX_STYLES]
    accented = ("dashed_corner", "bracket", "crosshair", "target")
    cells += [cell(style, accent=True) for style in accented]
    while len(cells) % COLUMNS:
        cells.append(np.full((CELL_H, CELL_W, 3), 20, dtype=np.uint8))

    rows = [np.hstack(cells[i : i + COLUMNS]) for i in range(0, len(cells), COLUMNS)]
    OUTPUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT), np.vstack(rows))
    print(f"{OUTPUT}  ({len(cells)} hücre)")


if __name__ == "__main__":
    main()
