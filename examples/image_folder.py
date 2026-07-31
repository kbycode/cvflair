"""
Bir klasördeki görselleri işaretleyip başka bir klasöre yazar.

Kamera yok, akış yok: yalnızca `Theme.annotate`. Toplu işlemede temanın nasıl
kurulup tekrar tekrar kullanıldığını gösterir — annotator'lar döngü dışında
bir kez hazırlanır.

Buradaki `detect` bir yer tutucudur: görselin ortasına tek kutu koyar. Gerçek
bir tespitle değiştirmek için `cvflair.Detections` döndüren herhangi bir
fonksiyon yeterli, bkz. examples/motion_detection.py.

Çalıştırmak için:
    python examples/image_folder.py girdi_klasoru
    python examples/image_folder.py girdi_klasoru --output isaretli --theme hud
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from cvflair import Detections, get_theme

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def detect(image: np.ndarray) -> Detections:
    """Yer tutucu tespit: görselin ortasında tek bir kutu."""
    height, width = image.shape[:2]
    return Detections(
        xyxy=[[width * 0.25, height * 0.25, width * 0.75, height * 0.75]],
        class_id=[0],
        confidence=[1.0],
        names=["ornek"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Görsellerin bulunduğu klasör")
    parser.add_argument("--output", type=Path, default=None, help="Varsayılan: <klasör>/cvflair")
    parser.add_argument("--theme", default="neon")
    args = parser.parse_args()

    images = sorted(path for path in args.folder.iterdir() if path.suffix.lower() in SUFFIXES)
    if not images:
        raise SystemExit(f"{args.folder} içinde görsel bulunamadı.")

    output = args.output or args.folder / "cvflair"
    output.mkdir(parents=True, exist_ok=True)
    theme = get_theme(args.theme)  # döngü dışında bir kez

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"atlandı (okunamadı): {path.name}")
            continue

        detections = detect(image)
        theme.annotate(image, detections, stats={"Objects": len(detections)})
        cv2.imwrite(str(output / path.name), image)
        print(f"{path.name}: {len(detections)} kutu")

    print(f"\nçıktı klasörü: {output}")


if __name__ == "__main__":
    main()
