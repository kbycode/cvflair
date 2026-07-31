"""
Video dosyasını işleyip işaretlenmiş kopyasını yazar.

`Camera` kaynak olarak dosya yolu da kabul eder; kamera olmayan bir makinede
ya da CI'da denemek için en pratik yol bu.

Çalıştırmak için:
    python examples/video_file.py girdi.mp4
    python examples/video_file.py girdi.mp4 --output cikti.mp4 --theme cyberpunk

Ekranda pencere açılmaz; kareler dosyaya yazılır.
"""

import argparse
from pathlib import Path

import cv2
from motion_detection import MotionDetector

from cvflair import Camera


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="İşlenecek video dosyası")
    parser.add_argument("--output", type=Path, default=None, help="Varsayılan: <girdi>-cvflair.mp4")
    parser.add_argument("--theme", default="neon")
    args = parser.parse_args()

    output = args.output or args.source.with_name(f"{args.source.stem}-cvflair.mp4")
    # drop_frames=False: dosyada her kare gerekli, okuyucu tüketiciyi bekler.
    cam = Camera(source=str(args.source), theme=args.theme, drop_frames=False)
    writer = None
    written = 0

    for frame, detections in cam.stream(timeout=2.0, model=MotionDetector()):
        cam.annotate(frame, detections)

        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output), cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height)
            )
            if not writer.isOpened():
                raise SystemExit(f"Çıktı dosyası açılamadı: {output}")
        writer.write(frame)
        written += 1

    if writer is not None:
        writer.release()
    print(f"{written} kare yazıldı -> {output}")


if __name__ == "__main__":
    main()
