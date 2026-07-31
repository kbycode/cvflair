# Örnekler

Hepsi depodan doğrudan çalışır. Kamera gerektirenler ayrıca işaretli; gerisi
kamerasız bir makinede de çalışır.

```bash
git clone https://github.com/kbycode/cvflair.git
cd cvflair
pip install -e .
python examples/quickstart.py
```

| Örnek | Ne gösteriyor | Kamera | Ek kurulum |
|---|---|---|---|
| [quickstart.py](quickstart.py) | Üç satırlık akış: kamerayı aç, kareleri göster | gerekli | — |
| [demo_fake_detections.py](demo_fake_detections.py) | Modelsiz canlı demo: hareketli sahte kutular, 3 saniyede bir tema değişimi | gerekli | — |
| [motion_detection.py](motion_detection.py) | **Gerçek tespit, sinir ağı yok:** arka plan çıkarımıyla hareket eden bölgeler | gerekli | — |
| [theme_switcher.py](theme_switcher.py) | Çalışırken tuşla tema değiştirme (`cam.pressed`) | gerekli | — |
| [hand_skeleton.py](hand_skeleton.py) | 21 noktalı el iskeleti, açılıp kapanan parmaklarla | gerekli | — |
| [hud_stats.py](hud_stats.py) | `hud` temasının sayaç panelini kendi verinle beslemek | gerekli | — |
| [yolo_quickstart.py](yolo_quickstart.py) | YOLO ağırlığıyla akış; etiketler sınıf adlarından | gerekli | `pip install "cvflair[yolo]"` |
| [video_file.py](video_file.py) | Video dosyasını işleyip işaretlenmiş kopyasını yazmak (`drop_frames=False`) | hayır | video dosyası |
| [image_folder.py](image_folder.py) | Bir klasördeki görselleri toplu işaretlemek | hayır | görsel klasörü |
| [theme_preview.py](theme_preview.py) | Her temayı bir PNG'ye çizmek | hayır | — |

## Nereden başlamalı

- **Kütüphaneyi ilk kez deniyorsan:** `quickstart.py` → `demo_fake_detections.py`.
- **Tuş kısayolu ekleyeceksen:** `theme_switcher.py` — `show()` sonrası `cam.pressed("1")`.
- **Kendi modelini bağlayacaksan:** `motion_detection.py` şablonu tam olarak bu —
  `Detections` döndüren bir çağrılabilir yaz, `cam.stream(model=...)` içine ver.
- **Kameran yoksa:** `theme_preview.py`, `image_folder.py`, `video_file.py`.

## Temaları önce tarayıcıda denemek

Kurulum yapmadan ayar denemek için: **[tema playground](https://kbycode.github.io/cvflair/)**
Ayarları oynat, hazır `Theme(...)` kodunu kopyala, buradaki örneklerin içine yapıştır.
