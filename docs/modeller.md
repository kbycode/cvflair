# Model bağlama ve tespitler

cvflair tespit **yapmaz**, tespit **çizer**. Kutuyu üreten şey senin seçtiğin
modeldir; kütüphane onu tek bir arayüze indirger.

## `stream(model=...)`

Model verildiğinde akış her adımda `(kare, tespitler)` çifti döndürür:

```python
from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)
```

`model` üç şeyden biri olabilir:

| Değer | Anlamı |
|---|---|
| `"yolov8n.pt"` (ağırlık yolu) | Ultralytics ile yüklenir — `pip install "cvflair[yolo]"` gerekir |
| Hazır bir Ultralytics modeli | `YOLO(...)` nesnesi doğrudan verilir, çıktısı dönüştürülür |
| Herhangi bir çağrılabilir | Kutu döndüren kendi fonksiyonun |

Ağırlık, akışın ilk yinelemesinde yüklenir; `stream()` çağrısında değil.

## Kendi detektörün

Kütüphaneyi model-agnostik yapan yer burası: kareyi alıp `Detections` döndüren
her çağrılabilir işe yarar.

```python
from cvflair import Camera, Detections

def detect(frame) -> Detections:
    ...  # kendi modelin, kendi kütüphanen
    return Detections(
        xyxy=[[10, 20, 120, 260]],
        class_id=[0],
        confidence=[0.91],
        names=["kisi"],
    )

cam = Camera(source=0, theme="pastel")
for frame, detections in cam.stream(model=detect):
    cam.show(frame, detections)
```

Çalışan bir örnek için
[`examples/motion_detection.py`](https://github.com/kbycode/cvflair/blob/main/examples/motion_detection.py):
arka plan çıkarımıyla hareket eden bölgeleri buluyor — sinir ağı yok, indirilecek
ağırlık yok, yalnızca OpenCV.

## `Detections`

| Alan | Zorunlu | Ne |
|---|---|---|
| `xyxy` | evet | `(N, 4)` piksel koordinatı: `[x1, y1, x2, y2]` |
| `class_id` | hayır | sınıf numarası; renk dağıtımı buna bakar |
| `confidence` | hayır | güven skoru |
| `names` | hayır | sınıf adı; etiket metni verilmezse buradan gelir |
| `tracker_id` | hayır | takip kimliği; `ColorLookup.TRACK` bunu kullanır |
| `mask` | hayır | `(N, H, W)` boolean segmentasyon maskesi; kare boyutunda olmalı |

Yardımcılar:

```python
Detections.empty()                       # boş
Detections.from_arrays(xyxy=..., ...)    # açık kurucu
Detections.from_ultralytics(result)      # Ultralytics Results nesnesinden
```

`from_ultralytics` yalnızca kutuları taşır; segmentasyon maskeleri ve döndürülmüş
kutular aktarılmaz — maskeleri `mask=` alanına kendin verebilir ya da başka bir
kütüphanenin tespit nesnesini doğrudan geçebilirsin.

### Maskeler

Maske verilirse çizilir: renkli yarı saydam dolgu ve kenarında kontur.

```python
Theme(mask_opacity=0.4, mask_outline=2)   # varsayılan
Theme(mask_opacity=0.0)                   # yalnız kontur, daha ucuz
Theme(masks=False)                        # maskeleri hiç çizme
```

Maske kutunun altında, çerçeve ve etiketin üstünde kalır. Maliyeti piksel
başına olduğu için kapladığı alanla orantılıdır; ölçülmüş rakamlar
[API belgesinde](api.md#ne-kadar-sürüyor).

### Bozuk kutular

Model çıktısı her zaman temiz gelmez: sıfıra bölme NaN üretir, ıraksayan bir
takipçi sonsuz koordinat verebilir. Böyle bir kutu çizilemez — o tespit atlanır,
aynı karedeki diğerleri normal çizilir. Akış bir bozuk kutu yüzünden durmaz.

Sonlu ama uçuk değerler (negatif, ters, kadrajı taşan) çizilir; koordinatlar
yalnızca OpenCV'nin sınırına kırpılır, kadraja değil.

### Başka kütüphanelerin tespitleri

Çizim, tespit nesnesini alan adlarına göre okur. Bu yüzden aynı alanları taşıyan
yabancı nesneler de dönüştürülmeden çalışır — örneğin `supervision.Detections`:

```python
import supervision as sv

detections = sv.Detections.from_ultralytics(result)
cam.show(frame, detections)               # dönüştürme yok
```

Aynısı paletler için de geçerli: `sv.ColorPalette` ve `sv.Color` doğrudan
`Theme(palette=...)` içine verilebilir. Bu uyum
[`tests/test_supervision_compat.py`](https://github.com/kbycode/cvflair/blob/main/tests/test_supervision_compat.py)
ile doğrulanıyor — ama `supervision` cvflair'in bağımlılığı değildir.

## Ultralytics ayarları

`UltralyticsDetector` ek parametreleri her çağrıya taşır:

```python
from cvflair import Camera, UltralyticsDetector
from ultralytics import YOLO

detector = UltralyticsDetector(YOLO("yolov8n.pt"), conf=0.4, device="cpu", classes=[0])
cam = Camera(source=0, theme="neon")
for frame, detections in cam.stream(model=detector):
    cam.show(frame, detections)
```

`verbose` varsayılan olarak kapalı, yoksa her kare için satır basılır.
Maskeleri de taşımak istersen dönüştürücüyü değiştirebilirsin:
`UltralyticsDetector(model, convert=sv.Detections.from_ultralytics)`.

## Çıkarım nerede çalışır

Döngünün içinde, okuma thread'inde değil. Bir kare işlenirken okuyucu kuyruktaki
kareyi tazelemeye devam eder; dolayısıyla bir sonraki tur birikmiş kareyle değil
en güncel kareyle başlar. Model yavaşsa ara kareler düşer, gecikme birikmez.

Video dosyasında bu istenmez — orada her kare gerekir:

```python
cam = Camera(source="girdi.mp4", theme="neon", drop_frames=False)
```

Bu kipte okuyucu tüketiciyi bekler, hiçbir kare atlanmaz.

## Lisans notu

YOLO ağırlıkları veya Ultralytics kodu pakete gömülü değildir; `yolo` extra'sı
bilinçli olarak isteğe bağlıdır. Ultralytics AGPL-3.0 lisanslıdır ve kullanılması
hâlinde koşulları onu kuran projenin sorumluluğundadır.
