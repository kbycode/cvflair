# cvflair

[![PyPI](https://img.shields.io/pypi/v/cvflair)](https://pypi.org/project/cvflair/)
[![CI](https://github.com/kbycode/cvflair/actions/workflows/ci.yml/badge.svg)](https://github.com/kbycode/cvflair/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/cvflair)](https://pypi.org/project/cvflair/)
[![İndirme](https://img.shields.io/pypi/dm/cvflair)](https://pypi.org/project/cvflair/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/kbycode/cvflair/blob/main/LICENSE)

Bilgisayarlı görü tespitlerini üç satırda, hazır temalarla ekrana çizen ince bir katman.

Çizim işini [supervision](https://github.com/roboflow/supervision) yapar; cvflair kamera
döngüsünü ve tema ayarlarını üstlenir. Model bağımsızdır: `supervision`'ın `Detections`
nesnesini üreten her kaynak (YOLO, MediaPipe, InsightFace veya özel bir model) tema
tarafından çizilebilir.

![cvflair demo](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/demo.gif)

*Aynı tespitler, üç tema. Animasyon `tools/make_demo_gif.py` ile üretildi — sentetik
sahne, kamera gerekmiyor.*

> **Durum:** Faz 1 tamam (kamera döngüsü, üç tema, testler), Faz 2 başladı (model
> bağlama). İlk sürüm PyPI'da: [cvflair 0.2.0](https://pypi.org/project/cvflair/).

## Kurulum

Python 3.10 veya üzeri gerekir (bu alt sınır `supervision`'dan geliyor).

```bash
pip install cvflair
```

YOLO ile kullanmak için Ultralytics extra'sı — ayrıntı ve lisans notu için aşağıdaki
[Lisans](#lisans) bölümü:

```bash
pip install "cvflair[yolo]"
```

Depodan geliştirme kurulumu:

```bash
git clone https://github.com/kbycode/cvflair.git
cd cvflair
pip install -e ".[dev]"
```

## Hızlı başlangıç

```python
from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame in cam.stream():
    cam.show(frame)
```

Kamera açılır, kareler ayrı bir thread'de okunur, pencere `q` veya ESC ile kapanır —
`release()` çağırmaya, `while True` kurmaya gerek yok.

> Model verilmeyen bu akışta ekranda ham kare görünür: tema ancak ortada tespit
> varken çizim yapar. Temayı modelsiz, canlı görmek için:
> `python examples/demo_fake_detections.py` — kamera görüntüsü üzerine hareketli sahte
> kutular çizer ve temaları 3 saniyede bir değiştirir.

## Modelle kullanım

`stream()`'e bir model verildiğinde her adım `(kare, tespitler)` çifti döndürür ve
tema otomatik uygulanır:

```python
from cvflair import Camera

cam = Camera(source=0, theme="neon")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)
```

`model` üç şeyden biri olabilir:

| Değer | Anlamı |
|---|---|
| `"yolov8n.pt"` (ağırlık yolu) | Ultralytics ile yüklenir — `cvflair[yolo]` gerekir |
| Hazır bir Ultralytics modeli | `YOLO(...)` nesnesi doğrudan verilebilir, çıktısı dönüştürülür |
| Herhangi bir çağrılabilir | `sv.Detections` döndüren herhangi bir fonksiyon — MediaPipe, InsightFace, özel model |

Son seçenek kütüphaneyi model-agnostik yapan yer:

```python
import supervision as sv
from cvflair import Camera

def detect(frame) -> sv.Detections:
    ...  # özel model çağrısı
    return sv.Detections(xyxy=..., class_id=..., confidence=...)

cam = Camera(source=0, theme="pastel")
for frame, detections in cam.stream(model=detect):
    cam.show(frame, detections)
```

Çıkarım bu döngüde çalışır, okuma thread'inde değil: bir kare işlenirken okuyucu
kuyruktaki kareyi tazelemeye devam eder, dolayısıyla bir sonraki tur birikmiş
kareyle değil en güncel kareyle başlar.

Ultralytics'e ek ayar geçirmek için `UltralyticsDetector` doğrudan kullanılabilir:

```python
from cvflair import Camera, UltralyticsDetector
from ultralytics import YOLO

detector = UltralyticsDetector(YOLO("yolov8n.pt"), conf=0.4, device="cpu", classes=[0])
cam = Camera(source=0, theme="neon")
for frame, detections in cam.stream(model=detector):
    cam.show(frame, detections)
```

Etiket metni doğrudan da verilebilir: `cam.show(frame, detections, labels=[...])`.
Pencere yönetimi uygulamaya aitse `cam.annotate(frame, detections)` yalnızca çizim yapar.

## Temalar

| Tema | Görünüm | |
|---|---|---|
| `minimal` | ince beyaz çerçeve, sade etiket — ekran kaydı ve profesyonel demo için | ![minimal](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-minimal.png) |
| `neon` | sınıf başına canlı renk, yuvarlak köşe, koyu hâle ile parlama hissi | ![neon](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-neon.png) |
| `pastel` | yumuşak tonlar, geniş yuvarlama, koyu etiket yazısı — atölye/projeksiyon | ![pastel](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-pastel.png) |

Yol haritasındaki `cyberpunk` ve `hud` temaları Faz 2'de gelecek.

Özel bir tema, `Theme` doğrudan kurulup `Camera`'ya verilerek tanımlanır:

```python
import supervision as sv
from cvflair import Camera, Theme

my_theme = Theme(
    name="my-theme",
    palette=sv.ColorPalette.from_hex(["#39FF14", "#FF00E5"]),
    box_style="corner",     # "box" | "round" | "corner"
    thickness=2,
    glow=True,
    text_scale=0.6,
)
cam = Camera(source=0, theme=my_theme)
```

Temaları görmenin iki yolu:

```bash
python examples/demo_fake_detections.py   # canlı kamera + hareketli sahte tespitler
python examples/theme_preview.py          # kamerasız, her temayı bir PNG'ye çizer
```

## API özeti

| Üye | Ne yapar |
|---|---|
| `Camera(source, theme, width, height, fps, window_name, capture_factory)` | Kaynağı ve temayı bağlar; kamerayı henüz açmaz |
| `cam.start()` / `cam.close()` | Cihazı açar ve okuma thread'ini başlatır / her şeyi bırakır |
| `cam.stream(timeout, model=None)` | Kareleri üretir; model verilirse `(kare, tespitler)` çifti. İlk kullanımda `start()`, bitince `close()` eder |
| `cam.read(timeout)` | En güncel tek kareyi döndürür, kaynak bittiyse `None` |
| `cam.show(frame, detections, labels)` | Temayı uygular, pencerede gösterir; çıkış istendiğinde `False` döner |
| `cam.annotate(frame, detections, labels)` | Sadece çizer, pencere açmaz |
| `cam.theme` | Okunur/yazılır; `cam.theme = "minimal"` çalışır |
| `cam.frames_read` / `cam.frames_dropped` | Okunan ve tüketici yetişemediği için atılan kare sayısı |
| `get_theme(ad)` / `available_themes()` | Tema adını çözer / mevcut adları listeler |
| `UltralyticsDetector(model, **kwargs)` | Ultralytics çıktısını `sv.Detections`'a çevirir; `conf`, `iou`, `device` gibi ayarları taşır |
| `resolve_detector(model)` | Ağırlık yolu / model / çağrılabilir → detektör; `stream()` bunu kullanır |

`Camera` bağlam yöneticisi olarak da kullanılabilir: `with Camera() as cam: ...`

## Nasıl çalışıyor

- **Kareler ayrı thread'de okunur.** Okuyucu, tüketiciyi beklemez.
- **Kuyruk tek slotlu.** Yeni kare gelince bekleyen eski kare düşürülür
  (`frames_dropped` ile sayılır). Böylece işleme yavaşladığında gecikme birikmez;
  ekranda hep en güncel kare olur.
- **Annotator'lar bir kere kurulur.** `Theme` nesnesi oluşturulurken `supervision`
  annotator'ları hazırlanır ve her karede yeniden kullanılır — döngü içinde annotator
  kurmak bu tür işlerde en sık görülen gereksiz maliyettir.
- **Çizim matematiği yeniden yazılmadı.** Her piksel `supervision` tarafından çiziliyor;
  cvflair sadece yapılandırma ve akış katmanı.
- **Model paketin dışında.** `stream(model=...)` verilen şeyi bir çağrılabilire çevirir;
  ağırlıklar ilk yinelemede yüklenir. Hiçbir model kodu veya ağırlığı pakete gömülü değil.

## Geliştirme

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check .
```

Testler kamera gerektirmez: `Camera`'ya `capture_factory` üzerinden sahte bir
`VideoCapture` verilir, temalar da sentetik kareler üzerinde doğrulanır.

Dokümantasyon görselleri de kamerasız üretilir:

```bash
python tools/make_demo_gif.py    # docs/demo.gif
python examples/theme_preview.py # examples/output/theme-*.png
```

Yayın adımları: [docs/pypi-yayin-rehberi.md](docs/pypi-yayin-rehberi.md).

## Yol haritası

| Faz | İçerik | Durum |
|---|---|---|
| Faz 1 | Kamera döngüsü, `minimal`/`neon`/`pastel` temaları, README, demo GIF, testler | tamam |
| Faz 2 | Model bağlama (`stream(model=...)`) | tamam |
| Faz 2 | PyPI paketi, Türkçe dokümantasyon sitesi, tema playground, `cyberpunk`/`hud` | sırada |
| Faz 3 | GitHub Actions (lint + test), issue şablonları, örnek galerisi | planlandı |

## Lisans

MIT — bkz. [LICENSE](LICENSE). Bağımlılıkların hepsi izin verici lisanslı
(`supervision` MIT, `opencv-python` Apache 2.0, `numpy` BSD).

YOLO ağırlıkları veya Ultralytics kodu bu pakete gömülü değildir; Ultralytics'in
kullanılması hâlinde AGPL-3.0 koşulları onu kullanan projenin sorumluluğundadır.
