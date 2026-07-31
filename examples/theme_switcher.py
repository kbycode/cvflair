"""
Çalışırken tema değiştirme: rakam tuşları temalar arasında geçiyor.

`show()` basılan tuşu `cam.key` içinde tutar, `cam.pressed(...)` de onu okur.
Playground'da fareyle yaptığın şeyin kamera karşısındaki hâli.

Tuşlar:  1-5 tema · b panel aç/kapa · q veya ESC çıkış

Çalıştırmak için:  python examples/theme_switcher.py
"""

from dataclasses import replace

from motion_detection import MotionDetector

from cvflair import Camera, available_themes


def main() -> None:
    themes = available_themes()
    print("Temalar: " + "  ".join(f"[{i + 1}] {name}" for i, name in enumerate(themes)))
    print("[b] sayaç paneli   [q] çıkış")

    cam = Camera(source=0, theme=themes[0])
    for frame, detections in cam.stream(model=MotionDetector()):
        cam.show(frame, detections)

        for index, name in enumerate(themes):
            if cam.pressed(str(index + 1)):
                cam.theme = name
                print(f"tema: {name}")

        # Panel her temaya açılabiliyor; mevcut temanın kopyasını değiştiriyoruz.
        if cam.pressed("b"):
            cam.theme = replace(cam.theme, hud=not cam.theme.hud)
            print(f"panel: {'açık' if cam.theme.hud else 'kapalı'}")


if __name__ == "__main__":
    main()
