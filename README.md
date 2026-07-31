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

**Dokuz çerçeve biçimi, beş hazır tema, el ve poz iskeleti — hepsi tek `Theme(...)` satırıyla.**
Hiçbirini kurmadan denemek için: **[tema playground →](https://kbycode.github.io/cvflair/)**
Önizlemede kutu ve iskelet kipleri ayrı ayrı denenebiliyor.
Ayarları oynat, hazır Python kodunu kopyala; sayfa tamamen tarayıcıda çalışır.

![çerçeve biçimleri](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/box-styles.png)

Kutunun üstüne binen çizimler de var: kilitlenme nabzı ve takip edilen nesnenin
bıraktığı iz.

![nabız ve iz](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/motion.png)

## Kurulum

Python 3.10 veya üzeri gerekir.

```bash
pip install cvflair
```

YOLO ile kullanmak için Ultralytics extra'sı (lisans notu aşağıda):

```bash
pip install "cvflair[yolo]"
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

Bu akışta ekranda ham kare görünür: tema ancak ortada tespit varken çizim yapar.
Bir model bağlandığında her adım `(kare, tespitler)` çiftine dönüşür ve tema
otomatik uygulanır:

```python
cam = Camera(source=0, theme="hud")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)
```

`model` yerine `Detections` döndüren kendi fonksiyonunu da verebilirsin —
kütüphaneyi model-agnostik yapan yer orası.

Kutuların yanında **el ve poz iskeleti** de çizilir; noktalar yine senin
modelinden gelir:

```python
from cvflair import HAND_21, KeyPoints

cam.show(frame, keypoints=KeyPoints(xy=el_noktalari), skeleton=HAND_21)
```

## Belgeler

| | |
|---|---|
| [Temalar ve çerçeve biçimleri](https://github.com/kbycode/cvflair/blob/main/docs/temalar.md) | Beş tema, dokuz biçim, nabız ve iz, ikinci renk, renk paleti, sayaç paneli, kendi temanı yazmak |
| [Nokta ve iskelet çizimi](https://github.com/kbycode/cvflair/blob/main/docs/noktalar.md) | El ve poz iskeletleri, `KeyPoints`, hazır topolojiler, kendi iskeletin |
| [Model bağlama ve tespitler](https://github.com/kbycode/cvflair/blob/main/docs/modeller.md) | `stream(model=...)`, kendi detektörün, `Detections`, Ultralytics ayarları, video dosyaları |
| [API özeti ve iç işleyiş](https://github.com/kbycode/cvflair/blob/main/docs/api.md) | Bütün genel arayüz, thread ve kuyruk davranışı, ölçülmüş performans |
| [Örnek galerisi](https://github.com/kbycode/cvflair/blob/main/examples/README.md) | On çalışan örnek; hangisi kamera istiyor, hangisi istemiyor |
| [Katkı rehberi](https://github.com/kbycode/cvflair/blob/main/CONTRIBUTING.md) | Kurulum, kapsam sınırları, yeni tema ekleme adımları |

Kameran yoksa bile çalışan üç örnek var:

```bash
python examples/motion_detection.py       # gerçek tespit, sinir ağı yok (kamera ister)
python examples/theme_preview.py          # kamerasız, her temayı bir PNG'ye çizer
python examples/video_file.py girdi.mp4   # dosyayı işleyip işaretlenmiş kopyasını yazar
```

## Neden bu tasarım

- **Kuyruk tek slotlu.** Yeni kare gelince bekleyen eski kare düşürülür; işleme
  yavaşladığında gecikme birikmez, ekranda hep en güncel kare olur. Video dosyası
  için `drop_frames=False` bunu tersine çevirir.
- **Çizim nesneleri bir kere kurulur**, her karede yeniden kullanılır.
- **Bağımlılık yüzeyi kasten dar.** `import cvflair` ~0,3 saniye, kurulum ~170 MB
  (neredeyse tamamı opencv + numpy).
- **Model paketin dışında.** Hiçbir ağırlık veya model kodu pakete gömülü değil.

Ölçülmüş rakamlar ve gerekçeleri:
[API ve iç işleyiş](https://github.com/kbycode/cvflair/blob/main/docs/api.md#ne-kadar-sürüyor).

## Geliştirme

```bash
git clone https://github.com/kbycode/cvflair.git
cd cvflair
pip install -e ".[dev]"

pytest              # kamera gerektirmez
ruff check .
mypy                # paket py.typed gönderiyor, tip iddiası denetleniyor
```

Ayrıntılar ve katkı akışı:
[CONTRIBUTING.md](https://github.com/kbycode/cvflair/blob/main/CONTRIBUTING.md)

## Lisans

MIT — bkz. [LICENSE](https://github.com/kbycode/cvflair/blob/main/LICENSE).
Bağımlılıkların ikisi de izin verici lisanslı (`opencv-python` Apache 2.0,
`numpy` BSD).

YOLO ağırlıkları veya Ultralytics kodu bu pakete gömülü değildir; Ultralytics'in
kullanılması hâlinde AGPL-3.0 koşulları onu kullanan projenin sorumluluğundadır.
