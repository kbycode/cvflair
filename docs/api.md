# API özeti ve iç işleyiş

## Camera

| Üye | Ne yapar |
|---|---|
| `Camera(source, theme, width, height, fps, window_name, drop_frames, capture_factory)` | Kaynağı ve temayı bağlar; kamerayı henüz açmaz |
| `cam.start()` / `cam.close()` | Cihazı açar ve okuma thread'ini başlatır / her şeyi bırakır |
| `cam.stream(timeout, model=None)` | Kareleri üretir; model verilirse `(kare, tespitler)` çifti. İlk kullanımda `start()`, bitince `close()` eder |
| `cam.read(timeout)` | En güncel tek kareyi döndürür, kaynak bittiyse `None` |
| `cam.show(frame, detections, labels, stats, keypoints, skeleton)` | Temayı uygular, pencerede gösterir; çıkış istendiğinde `False` döner |
| `cam.annotate(frame, detections, labels, stats, keypoints, skeleton)` | Sadece çizer, pencere açmaz |
| `cam.theme` | Okunur/yazılır; `cam.theme = "minimal"` çalışır |
| `cam.key` / `cam.pressed(tuş)` | Son `show()` sırasında basılan tuş; basılmadıysa `-1` |
| `cam.measured_fps` | Son 30 karenin ortalamasıyla ölçülen gerçek kare hızı |
| `cam.frames_read` / `cam.frames_dropped` | Okunan ve tüketici yetişemediği için atılan kare sayısı |
| `cam.drop_frames` | `True` canlı kamera için, `False` video dosyası için |

`Camera` bağlam yöneticisi olarak da kullanılabilir: `with Camera() as cam: ...`

## Tema ve çizim

| Üye | Ne yapar |
|---|---|
| `Theme(...)` | Bütün çizim ayarları — bkz. [temalar](temalar.md) |
| `theme.annotate(scene, detections, labels, stats)` | Kareye yerinde çizer, aynı diziyi döndürür |
| `theme.annotate_keypoints(scene, keypoints, skeleton)` | İskelet çizer — bkz. [noktalar](noktalar.md) |
| `get_theme(ad)` / `available_themes()` | Tema adını çözer / mevcut adları listeler |
| `BOX_STYLES` / `HUD_POSITIONS` | Geçerli değerler |
| `Color`, `ColorPalette`, `ColorLookup` | Renk altyapısı; hex dizgeleri her yerde kabul edilir |
| `cvflair.annotators` | Sekiz çerçeve biçimi, etiket plakası ve sayaç paneli sınıfları |

## Tespit ve model

| Üye | Ne yapar |
|---|---|
| `Detections(xyxy, class_id, confidence, names, tracker_id)` | Kutu taşıyıcısı — bkz. [modeller](modeller.md) |
| `KeyPoints(xy, confidence, class_id)` | Nokta taşıyıcısı; `from_normalized` yardımcısıyla |
| `HAND_21` / `POSE_17` / `SKELETONS` | Hazır iskelet topolojileri |
| `UltralyticsDetector(model, **kwargs)` | Ultralytics çıktısını `Detections`'a çevirir |
| `load_ultralytics(weights, **kwargs)` | Ağırlık dosyasını yükler; extra eksikse açıklayıcı hata verir |
| `resolve_detector(model)` | Ağırlık yolu / model / çağrılabilir → detektör; `stream()` bunu kullanır |

## Nasıl çalışıyor

- **Kareler ayrı thread'de okunur.** Okuyucu, tüketiciyi beklemez.
- **Kuyruk tek slotlu.** Yeni kare gelince bekleyen eski kare düşürülür
  (`frames_dropped` ile sayılır). Böylece işleme yavaşladığında gecikme birikmez;
  ekranda hep en güncel kare olur. `drop_frames=False` bu davranışı tersine
  çevirir: okuyucu bekler, hiçbir kare atlanmaz.
- **Çizim nesneleri bir kere kurulur.** `Theme` oluşturulurken hazırlanır ve her
  karede yeniden kullanılır. Ölçüldüğünde kurulum 0,02–0,10 ms; döngü içinde tema
  kurmak çizim süresine %1–8 ekler — gerçek ama küçük bir maliyet.
- **Bağımlılık yüzeyi kasten dar.** Yalnızca numpy ve opencv. `import cvflair`
  ~0,3 saniye sürer; kurulum ~170 MB (neredeyse tamamı opencv + numpy).
- **Model paketin dışında.** Hiçbir model kodu veya ağırlığı pakete gömülü değil.

## Ne kadar sürüyor

1280x720, 8 kutu,
[`tools/benchmark.py`](https://github.com/kbycode/cvflair/blob/main/tools/benchmark.py)
(i5-3xxx, OpenCV 5.0). Sayılar makineye özgü; oranlar aynı koşuda anlamlı.

| Tema | ms/kare | 30 fps bütçesinin |
|---|---|---|
| `minimal` | 0,35 | %1 |
| `hud` | 0,51 | %2 |
| `cyberpunk` | 0,64 | %2 |
| `pastel` | 2,8 | %8 |
| `neon` | 4,1 | %12 |

Maske çizimi ayrı bir hikâye — piksel başına iş olduğu için maliyeti kapladığı
alanla orantılı. Aynı sahnede 8 maske (her biri kutusuna içten teğet elips):

| Çizim | ms/kare |
|---|---|
| dolgu + kontur | 10,7 |
| yalnız kontur (`mask_opacity=0`) | 7,8 |

Maske yalnızca tespitte varsa çizilir; olmayan sahnede ölçülebilir bir maliyeti
yok. İlk sürümde bu 46 ms'ydi: iş tüm kare yerine nesnenin penceresine indirildi
ve dizin atama yerine OpenCV'nin maskeli kopyası kullanıldı.

Karşılaştırma tabanı, elle yazılmış bir OpenCV kutu+etiket döngüsü: 0,34 ms.
`minimal` onunla aynı hızda; aradaki fark tamamen çizilen şeyden geliyor.
Pahalı olan yuvarlak köşe: kenar yumuşatmalı yay, düz çizgiye göre dört kat
maliyetli ve her kutuda dört tane var. `neon` bunun üstüne bir de hâle geçişi
ekliyor. Hâle sönük ve ana çizginin altında kaldığı için orada kenar yumuşatma
kapalı — bu tek değişiklik `neon`'u 7,1 ms'den 4,1 ms'ye indirdi.

## Tip bilgisi

Paket `py.typed` gönderir; genel arayüzün tipleri CI'da `mypy` ile denetlenir.
