"""
Üç satırlık başlangıç: kamerayı aç, kareleri neon temasıyla göster.

Tespit verilmediği için ekranda ham kare görünür — tema yalnızca `show()`'a bir
`Detections` geldiğinde çizim yapar. Temayı modelsiz görmek için:
examples/demo_fake_detections.py

Çalıştırmak için:  python examples/quickstart.py
Çıkmak için:      pencere seçiliyken 'q' veya ESC
"""

from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame in cam.stream():
    cam.show(frame)
