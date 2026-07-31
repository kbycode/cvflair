# cvflair

[![PyPI](https://img.shields.io/pypi/v/cvflair)](https://pypi.org/project/cvflair/)
[![CI](https://github.com/kbycode/cvflair/actions/workflows/ci.yml/badge.svg)](https://github.com/kbycode/cvflair/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/cvflair)](https://pypi.org/project/cvflair/)
[![İndirme](https://img.shields.io/pypi/dm/cvflair)](https://pypi.org/project/cvflair/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/kbycode/cvflair/blob/main/LICENSE)

Bilgisayarlı görü tespitlerini üç satırda, hazır temalarla ekrana çizen ince bir katman.

Kamera döngüsü, temalar ve çizim tek pakette; **numpy ve opencv dışında bağımlılığı
yok**. Model bağımsızdır: kutu üreten her kaynak (YOLO, MediaPipe, InsightFace veya
özel bir model) aynı temayla çizilir.

![cvflair demo](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/demo.gif)

*Aynı tespitler, dört tema. `tools/make_demo_gif.py` ile üretildi: `docs/city.png`
üzerine çizilen kutular. Başka bir görsel için `--background <yol>`.*

**Sekiz çerçeve biçimi, dört hazır tema — hepsi tek `Theme(...)` satırıyla.**
Hiçbirini kurmadan denemek için: **[tema playground →](https://kbycode.github.io/cvflair/)**
Ayarları oynat, hazır Python kodunu kopyala; sayfa tamamen tarayıcıda çalışır.

![çerçeve biçimleri](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/box-styles.png)

## Kurulum

Python 3.10 veya üzeri gerekir.

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
| Herhangi bir çağrılabilir | Kutu döndüren herhangi bir fonksiyon — MediaPipe, InsightFace, özel model |

Son seçenek kütüphaneyi model-agnostik yapan yer:

```python
from cvflair import Camera, Detections

def detect(frame) -> Detections:
    ...  # özel model çağrısı
    return Detections(xyxy=[[10, 20, 120, 260]], class_id=[0], names=["kisi"])

cam = Camera(source=0, theme="pastel")
for frame, detections in cam.stream(model=detect):
    cam.show(frame, detections)
```

Çizim alan adlarına göre okuduğu için başka kütüphanelerin tespit nesneleri de
(örneğin `supervision.Detections`) dönüştürülmeden verilebilir.

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

`show()` basılan tuşu da tutar, böylece demolar klavyeye cevap verebilir:

```python
cam.show(frame, detections)
if cam.pressed("1"):
    cam.theme = "neon"
```

## Temalar

| Tema | Görünüm | |
|---|---|---|
| `minimal` | ince beyaz çerçeve, sade etiket — ekran kaydı ve profesyonel demo için | ![minimal](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-minimal.png) |
| `neon` | sınıf başına canlı renk, yuvarlak köşe, koyu hâle ile parlama hissi | ![neon](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-neon.png) |
| `pastel` | yumuşak tonlar, geniş yuvarlama, koyu etiket yazısı — atölye/projeksiyon | ![pastel](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-pastel.png) |
| `cyberpunk` | ince çerçeve + kalın beyaz köşeler, yüksek kontrast — hedef kilitleme görünümü | ![cyberpunk](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-cyberpunk.png) |
| `hud` | ince köşe çentikleri + köşede sayaç paneli — oyun ve robotik demoları | ![hud](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-hud.png) |

### Çerçeve biçimleri

Görselleri yukarıda. `box_style` sekiz değerden birini alır:

| Değer | Görünüm | Ayarları |
|---|---|---|
| `box` | düz dikdörtgen | `thickness` |
| `round` | yuvarlak köşeli dikdörtgen | `roundness` |
| `corner` | yalnızca köşe çentikleri | `corner_length` |
| `dashed` | kesikli çerçeve | `dash_length`, `gap_length` |
| `dashed_corner` | kesikli çerçevenin üstüne dolu köşe ayraçları (kesikli + köşe karışımı) | `corner_length`, `dash_length`, `gap_length` |
| `bracket` | yuvarlak dirsekli köşe ayracı (köşe + yuvarlak karışımı) | `corner_length`, `roundness` |
| `crosshair` | kenar ortası çentikleri + merkez artısı | `arm_length`, `center_size` |
| `target` | ince çerçeve + kalın köşeler | `corner_length`, `edge_thickness` |

Hepsi `cvflair.annotators` içinde, OpenCV çizim çağrılarıyla tanımlıdır; renk
paleti ve `ColorLookup` davranışı sekizinde de aynıdır.

`dashed_corner`, `bracket`, `crosshair` ve `target` ikinci bir renk kabul eder; köşe
ayraçları, dirsekler ve merkez artısı o renge geçer — hibrit biçimlerin iki katmanı
böyle ayrışır:

```python
theme = Theme(
    palette=["#00F0FF"],
    accent_palette="#FF206E",
    box_style="target",
    thickness=3,
)
```

### Sayaç paneli

`hud` teması köşeye küçük bir panel çizer. Sayılar kutulardan değil döngüden gelir,
bu yüzden ayrı bir yoldan veriliyor — `Camera` kare hızını ve tespit sayısını
kendisi dolduruyor:

```python
cam = Camera(source=0, theme="hud")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)          # FPS ve Objects panelde
```

Kendi satırlarını eklemek için `stats`; aynı anahtar verilirse seninki kazanır:

```python
cam.show(frame, detections, stats={"Skor": score, "Tur": lap})
```

Panel her temaya açılabilir: `Theme(hud=True, hud_position="bottom_right", hud_opacity=0.5)`.
Konumlar `cvflair.HUD_POSITIONS` içinde. Ölçülen kare hızına `cam.measured_fps` ile
doğrudan da erişilebilir — cihazdan istenen `fps` değil, döngünün gerçekte ulaştığı hız.

Özel bir tema, `Theme` doğrudan kurulup `Camera`'ya verilerek tanımlanır:

```python
from cvflair import Camera, Theme

my_theme = Theme(
    name="my-theme",
    palette=["#39FF14", "#FF00E5"],
    box_style="corner",     # bkz. Çerçeve biçimleri
    thickness=2,
    glow=True,
    text_scale=0.6,
)
cam = Camera(source=0, theme=my_theme)
```

Ayarları tarayıcıda deneyip hazır `Theme(...)` kodunu kopyalamak için:
**[tema playground](https://kbycode.github.io/cvflair/)** — kurulum gerektirmez,
tamamen istemci tarafında çalışır, ayarlar bağlantıda taşınır.

Playground İngilizce açılır, sağ üstteki **TR** düğmesiyle Türkçeye geçer.

Temaları yerelde görmek ve kendi modelini bağlamak için sekiz örnek var —
[örnek galerisi](examples/README.md):

```bash
python examples/motion_detection.py       # gerçek tespit, sinir ağı yok
python examples/hud_stats.py              # sayaç panelini kendi verinle besle
python examples/theme_preview.py          # kamerasız, her temayı bir PNG'ye çizer
```

## API özeti

| Üye | Ne yapar |
|---|---|
| `Camera(source, theme, width, height, fps, window_name, capture_factory)` | Kaynağı ve temayı bağlar; kamerayı henüz açmaz |
| `cam.start()` / `cam.close()` | Cihazı açar ve okuma thread'ini başlatır / her şeyi bırakır |
| `cam.stream(timeout, model=None)` | Kareleri üretir; model verilirse `(kare, tespitler)` çifti. İlk kullanımda `start()`, bitince `close()` eder |
| `cam.read(timeout)` | En güncel tek kareyi döndürür, kaynak bittiyse `None` |
| `cam.show(frame, detections, labels, stats)` | Temayı uygular, pencerede gösterir; çıkış istendiğinde `False` döner |
| `cam.annotate(frame, detections, labels)` | Sadece çizer, pencere açmaz |
| `cam.theme` | Okunur/yazılır; `cam.theme = "minimal"` çalışır |
| `cam.frames_read` / `cam.frames_dropped` | Okunan ve tüketici yetişemediği için atılan kare sayısı |
| `cam.measured_fps` | Son 30 karenin ortalamasıyla ölçülen gerçek kare hızı |
| `cam.key` / `cam.pressed(tuş)` | Son `show()` sırasında basılan tuş; yoksa `-1` |
| `get_theme(ad)` / `available_themes()` | Tema adını çözer / mevcut adları listeler |
| `Detections(xyxy, class_id, confidence, names, tracker_id)` | Kutu taşıyıcısı; `from_ultralytics` ve `from_arrays` yardımcılarıyla |
| `UltralyticsDetector(model, **kwargs)` | Ultralytics çıktısını `Detections`'a çevirir; `conf`, `iou`, `device` gibi ayarları taşır |
| `resolve_detector(model)` | Ağırlık yolu / model / çağrılabilir → detektör; `stream()` bunu kullanır |

`Camera` bağlam yöneticisi olarak da kullanılabilir: `with Camera() as cam: ...`

## Nasıl çalışıyor

- **Kareler ayrı thread'de okunur.** Okuyucu, tüketiciyi beklemez.
- **Kuyruk tek slotlu.** Yeni kare gelince bekleyen eski kare düşürülür
  (`frames_dropped` ile sayılır). Böylece işleme yavaşladığında gecikme birikmez;
  ekranda hep en güncel kare olur.
- **Annotator'lar bir kere kurulur.** `Theme` nesnesi oluşturulurken çizim
  nesneleri hazırlanır ve her karede yeniden kullanılır. Ölçüldüğünde kurulum
  0,02–0,10 ms; döngü içinde tema kurmak çizim süresine %1–8 ekliyor — gerçek
  ama küçük bir maliyet.
- **Bağımlılık yüzeyi kasten dar.** Yalnızca numpy ve opencv. `import cvflair`
  ~0.3 saniye sürüyor; kurulum ~170 MB (neredeyse tamamı opencv + numpy).
- **Model paketin dışında.** `stream(model=...)` verilen şeyi bir çağrılabilire çevirir;
  ağırlıklar ilk yinelemede yüklenir. Hiçbir model kodu veya ağırlığı pakete gömülü değil.

### Ne kadar sürüyor

1280x720, 8 kutu, `tools/benchmark.py` (i5-3xxx, OpenCV 5.0). Sayılar makineye
özgü; oranlar aynı koşuda anlamlı.

| Tema | ms/kare | 30 fps bütçesinin |
|---|---|---|
| `minimal` | 0,35 | %1 |
| `hud` | 0,51 | %2 |
| `cyberpunk` | 0,64 | %2 |
| `pastel` | 2,8 | %8 |
| `neon` | 4,1 | %12 |

Karşılaştırma tabanı, elle yazılmış bir OpenCV kutu+etiket döngüsü: 0,34 ms.
`minimal` onunla aynı hızda; aradaki fark tamamen çizilen şeyden geliyor.
Pahalı olan yuvarlak köşe: kenar yumuşatmalı yay, düz çizgiye göre dört kat
maliyetli ve her kutuda dört tane var. `neon` bunun üstüne bir de hâle geçişi
ekliyor. Hâle sönük ve ana çizginin altında kaldığı için orada kenar yumuşatma
kapalı — bu tek değişiklik `neon`'u 7,1 ms'den 4,1 ms'ye indirdi.

## Geliştirme

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest              # kamera gerektirmez
ruff check .
mypy                # paket py.typed gönderiyor, tip iddiası denetleniyor
```

Katkı akışı, kapsam sınırları ve yeni tema ekleme adımları:
[CONTRIBUTING.md](CONTRIBUTING.md)

Testler kamera gerektirmez: `Camera`'ya `capture_factory` üzerinden sahte bir
`VideoCapture` verilir, temalar da sentetik kareler üzerinde doğrulanır.

Dokümantasyon görselleri de kamerasız üretilir:

```bash
python tools/make_demo_gif.py     # docs/demo.gif
python tools/make_style_sheet.py  # docs/box-styles.png + docs/theme-*.png
python tools/benchmark.py         # yukarıdaki tabloyu üretir
```

## Lisans

MIT — bkz. [LICENSE](LICENSE). Bağımlılıkların ikisi de izin verici lisanslı
(`opencv-python` Apache 2.0, `numpy` BSD).

YOLO ağırlıkları veya Ultralytics kodu bu pakete gömülü değildir; Ultralytics'in
kullanılması hâlinde AGPL-3.0 koşulları onu kullanan projenin sorumluluğundadır.
