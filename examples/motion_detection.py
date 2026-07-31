"""
Modelsiz gerçek tespit: hareket eden bölgeleri arka plan çıkarımıyla bulur.

Sinir ağı yok, indirilecek ağırlık yok — yalnızca OpenCV. cvflair'in model
bağımsızlığı tam olarak bu: kutu üreten her şey tema tarafından çizilir.

Çalıştırmak için:  python examples/motion_detection.py
Çıkmak için:      pencere seçiliyken 'q' veya ESC
"""

import cv2
import numpy as np

from cvflair import Camera, Detections

#: Bu alandan küçük lekeler gürültü sayılıp atlanıyor.
MIN_AREA = 1200


class MotionDetector:
    """Arka planı öğrenir, ondan ayrışan bölgelerin kutusunu döndürür."""

    def __init__(self) -> None:
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=32, detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def __call__(self, frame: np.ndarray) -> Detections:
        mask = self.subtractor.apply(frame)
        # Açma işlemi tek piksellik parazitleri siler, kapama delikleri doldurur.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        scores = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append([x, y, x + width, y + height])
            # Güven yerine doluluk oranı: kutuyu ne kadar doldurduğu.
            scores.append(min(1.0, area / (width * height)))

        if not boxes:
            return Detections.empty()
        return Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            confidence=np.array(scores, dtype=np.float32),
            names=np.array(["hareket"] * len(boxes), dtype=object),
        )


def main() -> None:
    print("Arka planın öğrenilmesi birkaç saniye sürer — çıkmak için 'q'")
    cam = Camera(source=0, theme="hud")
    for frame, detections in cam.stream(model=MotionDetector()):
        cam.show(frame, detections)


if __name__ == "__main__":
    main()
