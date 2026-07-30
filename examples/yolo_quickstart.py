"""
YOLO ile üç satırlık akış.

Gereken: pip install "cvflair[yolo]"
Ağırlık dosyası ilk çalıştırmada Ultralytics tarafından indirilir.
Ultralytics AGPL-3.0 lisanslıdır; cvflair'in zorunlu bağımlılığı değildir.

Etiketler sınıf adlarından gelir (supervision dönüşümü class_name alanını doldurur).
Güven skorunu da yazdırmak için:
    labels = [f"{n} {c:.2f}" for n, c in zip(detections["class_name"], detections.confidence)]
    cam.show(frame, detections, labels=labels)

Çalıştırmak için:  python examples/yolo_quickstart.py
Çıkmak için:      pencere seçiliyken 'q' veya ESC
"""

from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)
