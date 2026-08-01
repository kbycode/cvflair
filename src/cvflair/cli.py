"""
The ``cvflair`` command: annotate a camera, a video, an image or a folder
without writing any Python.

    cvflair 0 --theme neon --model yolov8n.pt
    cvflair girdi.mp4 --theme cyberpunk --model yolov8n.pt -o cikti.mp4
    cvflair fotograflar/ --model yolov8n.pt -o isaretli/

Detection still comes from a model of your choosing; without ``--model``
nothing is drawn, because cvflair does not detect anything itself.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .camera import Camera, CameraError
from .detections import Detections
from .models import resolve_detector
from .themes import available_themes, get_theme
from .video import VideoWriteError, VideoWriter

__all__ = ["main"]

#: OpenCV'nin okuyabildiği yaygın görsel uzantıları; klasör taramasında kullanılıyor.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvflair",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Kamera numarası (0), video dosyası, görsel ya da görsel klasörü",
    )
    parser.add_argument("-o", "--output", type=Path, help="Çıktı dosyası ya da klasörü")
    parser.add_argument("--theme", default="neon", help="Tema adı (varsayılan: neon)")
    parser.add_argument("--model", help="Ağırlık dosyası, örn. yolov8n.pt")
    parser.add_argument("--fps", type=float, help="Çıktı hızı; verilmezse kaynaktan alınır")
    parser.add_argument("--codec", default="mp4v", help="Video codec'i (varsayılan: mp4v)")
    parser.add_argument(
        "--no-window", action="store_true", help="Pencere açma; yalnızca dosyaya yaz"
    )
    parser.add_argument("--themes", action="store_true", help="Tema adlarını listele ve çık")
    parser.add_argument("--version", action="version", version=f"cvflair {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.themes:
        print("\n".join(available_themes()))
        return 0
    if not args.source:
        parser.error("kaynak gerekli (kamera numarası, video, görsel ya da klasör)")

    try:
        theme = get_theme(args.theme)
    except (ValueError, KeyError) as error:
        print(f"cvflair: {error}", file=sys.stderr)
        print(f"Tema adları: {', '.join(available_themes())}", file=sys.stderr)
        return 2

    detector = resolve_detector(args.model) if args.model else None
    if detector is None:
        print(
            "cvflair: --model verilmedi, tespit yapılmayacak; kareler olduğu gibi geçecek.",
            file=sys.stderr,
        )

    images = collect_images(args.source)
    try:
        if images is not None:
            return run_images(images, args, theme, detector)
        return run_stream(args, theme, detector)
    except (CameraError, VideoWriteError, OSError) as error:
        print(f"cvflair: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ncvflair: durduruldu", file=sys.stderr)
        return 130


def collect_images(source: str) -> list[Path] | None:
    """Kaynak görsel ya da görsel klasörüyse dosya listesi, değilse ``None``."""
    path = Path(source)
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not found:
            raise OSError(f"Klasörde görsel bulunamadı: {path}")
        return found
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    return None


def run_images(images: list[Path], args, theme, detector) -> int:
    output = args.output
    if output is not None and len(images) > 1:
        output.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"cvflair: okunamadı, atlanıyor: {path}", file=sys.stderr)
            continue

        detections = detector(frame) if detector else Detections.empty()
        # Durağan görselde kare hızı diye bir şey yok; panelli temada anlamlı olan
        # tek sayaç nesne adedi.
        theme.annotate(frame, detections, stats={"Objects": len(detections)})
        target = image_target(path, output, len(images))
        if not cv2.imwrite(str(target), frame):
            raise OSError(f"Yazılamadı: {target}")
        written += 1
        print(f"{path.name} -> {target}")

    print(f"{written} görsel işaretlendi")
    return 0 if written else 1


def image_target(source: Path, output: Path | None, count: int) -> Path:
    """Tek görselde çıktı bir dosya, çoklu görselde klasör; verilmezse yanına yazılır."""
    if output is None:
        return source.with_name(f"{source.stem}-cvflair{source.suffix}")
    if count > 1 or output.is_dir():
        return output / source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def run_stream(args, theme, detector) -> int:
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    live = isinstance(source, int)
    if not live and not Path(args.source).exists():
        raise OSError(f"Kaynak bulunamadı: {args.source}")

    # Dosyaya yazarken pencere açmak gereksiz; canlı kamerada ise pencere olmadan
    # ne olup bittiği görünmüyor, o yüzden yazarken bile açık kalıyor.
    show = not args.no_window and (args.output is None or live)
    # Dosyada her kare gerekli; canlı kamerada eski kareyi düşürmek doğru.
    cam = Camera(source=source, theme=theme, drop_frames=live)
    writer: VideoWriter | None = None

    try:
        for frame, detections in stream_pairs(cam, detector):
            # Tek çizim: show() kareyi kendisi işaretliyor ve panelli temada kare
            # hızını dolduruyor. İkisini birden çağırmak paneli ikinci kez, bu kez
            # sıfır nesneyle çizerdi.
            if show:
                running = cam.show(frame, detections)
            else:
                cam.annotate(frame, detections)
                running = True

            if args.output is not None:
                if writer is None:
                    writer = VideoWriter(
                        args.output, fps=args.fps or cam.source_fps or 25.0, codec=args.codec
                    )
                writer.write(frame)
            if not running:
                break
    finally:
        if writer is not None:
            writer.close()
        cam.close()

    if writer is not None:
        print(f"{writer.frames_written} kare yazıldı -> {writer.path}")
    else:
        print(f"{cam.frames_read} kare okundu")
    return 0


def stream_pairs(cam: Camera, detector) -> Iterator[tuple[np.ndarray, Detections | None]]:
    """Modelli ve modelsiz akışı tek biçime indirir."""
    if detector is None:
        for frame in cam.stream():
            yield frame, None
    else:
        yield from cam.stream(model=detector)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
