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
| `cam.source_fps` / `cam.frame_count` | Kaynağın kendi bildirdiği hız ve kare sayısı; kamerada `0` |

`Camera` bağlam yöneticisi olarak da kullanılabilir: `with Camera() as cam: ...`

## Video yazma

| Üye | Ne yapar |
|---|---|
| `VideoWriter(path, fps, codec)` | İşaretli kareleri dosyaya yazar; dosya ilk kareyle açılır |
| `writer.write(frame)` / `writer.close()` | Kare yazar / dosyayı kapatır (`with` de çalışır) |
| `writer.frames_written` / `writer.size` | Yazılan kare sayısı, dosyanın boyutu |

Ayrıntı ve komut satırı karşılığı: [komut satırı](komut-satiri.md).

## Tema ve çizim

| Üye | Ne yapar |
|---|---|
| `Theme(...)` | Bütün çizim ayarları — bkz. [temalar](temalar.md) |
| `theme.annotate(scene, detections, labels, stats, moment)` | Kareye yerinde çizer, aynı diziyi döndürür |
| `theme.annotate_zone(scene, points, fill_opacity)` | Poligon ya da çizgi çizer |
| `theme.reset_trace()` | Biriken takip izlerini siler |
| `theme.annotate_keypoints(scene, keypoints, skeleton)` | İskelet çizer — bkz. [noktalar](noktalar.md) |
| `get_theme(ad \| yol \| sözlük)` / `available_themes()` | Temayı çözer (ad, `.json` yolu ya da sözlük) / adları listeler |
| `theme.save(yol)` / `Theme.load(yol)` | Temayı JSON olarak yazar / okur |
| `theme.to_dict()` / `Theme.from_dict(...)` | Düz veriye çevirir / geri kurar; yalnızca varsayılandan farklılar yazılır |
| `BOX_STYLES` / `HUD_POSITIONS` | Geçerli değerler |
| `Color`, `ColorPalette`, `ColorLookup` | Renk altyapısı; hex dizgeleri her yerde kabul edilir |
| `cvflair.annotators` | Dokuz çerçeve biçimi, etiket plakası, iz, nabız ve panel sınıfları |

## Not defteri

`cv2.imshow` Jupyter ve Colab'da çalışmaz: pencere açacak bir masaüstü yok, çağrı
ya hiçbir şey yapmaz ya da çekirdeği düşürür. Bilinen çözüm matplotlib artı kanal
çevirmedir; OpenCV pikselleri BGR tuttuğu için çevirmeyi atlayan kırmızıyla maviyi
takas etmiş bir görüntü alır.

```python
import cvflair

theme.annotate(frame, detections)
cvflair.notebook.show(frame)
```

| Üye | Ne yapar |
|---|---|
| `notebook.show(frame, bgr=True, width=None)` | Kareyi hücrede gösterir, görüntü nesnesini döndürür |
| `notebook.to_png(frame, bgr=True)` | PNG baytları; dosyaya yazmak ya da başka yere göndermek için |
| `notebook.in_notebook()` | Çekirdek içinde miyiz |

Not defteri dışında `show` çizmez, yalnızca nesneyi döndürür — aynı betik iki
yerde de çalışır. IPython bir bağımlılık değil; yalnızca çağrıldığında import
edilir.

## Tespit ve model

| Üye | Ne yapar |
|---|---|
| `Detections(xyxy, class_id, confidence, names, tracker_id)` | Kutu taşıyıcısı — bkz. [modeller](modeller.md) |
| `Detections.from_xywh(kutular, genislik, yukseklik)` | Köşe+boyut kutuları; oranlıysa kare boyutuyla ölçekler |
| `Detections.from_mediapipe(sonuc, genislik, yukseklik)` | MediaPipe yüz/nesne tespiti; iki API sürümü de |
| `KeyPoints(xy, confidence, class_id)` | Nokta taşıyıcısı; `from_normalized` ve `from_mediapipe` yardımcılarıyla |
| `HAND_21` / `POSE_17` / `FACE_5` / `SKELETONS` | Hazır iskelet topolojileri (`"hand"`, `"pose"`, `"face"`) |
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
| `minimal` | 0,64 | %2 |
| `hud` | 0,80 | %2 |
| `cyberpunk` | 0,88 | %3 |
| `pastel` | 3,1 | %9 |
| `neon` | 4,2 | %13 |

Ölçü, üç koşunun en iyi karesi: ortanca arka plan yüküyle koşudan koşuya %20
oynuyor, en iyi süre oynamıyor.

Etiket plakaları bu sürenin küçük olmayan bir parçası: aynı sahnede etiketler
kapalıyken düz kutu 0,64 yerine 0,29 ms. Kutu başına metin ölçümü, plaka dolgusu
ve `putText` var; plakaların çakışmadan yerleştirilmesi bunun üstüne 0,06 ms
ekliyor.

Kutunun üstüne binen çizimler, aynı sahnede düz kutuya (0,74 ms) göre:

| Çizim | ms/kare | ek |
|---|---|---|
| `pulse` | 0,98 | +0,24 |
| `trace` (32 nokta, 8 iz) | 1,8 | +1,1 |
| `sketch` çerçeve | 4,4 | +3,7 |

`sketch` pahalı olan: kutu başına sekiz titrek çoklu-çizgi, hepsi kenar
yumuşatmalı. Kenar başına ayrı hesap ve ayrı OpenCV çağrısı yapan ilk hâli
9,4 ms'ydi; kutunun bütün kenarları tek dizide üretilip tek çağrıda çizilince
5 ms'nin altına indi. Geri kalanı rasterleştirmenin kendisi.

Maske çizimi ayrı bir hikâye — piksel başına iş olduğu için maliyeti kapladığı
alanla orantılı. Aynı sahnede 8 maske (her biri kutusuna içten teğet elips):

| Çizim | ms/kare |
|---|---|
| dolgu + kontur | 10,4 |
| yalnız kontur (`mask_opacity=0`) | 7,6 |

Maske yalnızca tespitte varsa çizilir; olmayan sahnede ölçülebilir bir maliyeti
yok. İlk sürümde bu 46 ms'ydi: iş tüm kare yerine nesnenin penceresine indirildi
ve dizin atama yerine OpenCV'nin maskeli kopyası kullanıldı.

Karşılaştırma tabanı, elle yazılmış bir OpenCV kutu+etiket döngüsü: 0,32 ms.
`minimal` onunla aynı hızda; aradaki fark tamamen çizilen şeyden geliyor.
Pahalı olan yuvarlak köşe: kenar yumuşatmalı yay, düz çizgiye göre dört kat
maliyetli ve her kutuda dört tane var. `neon` bunun üstüne bir de hâle geçişi
ekliyor. Hâle sönük ve ana çizginin altında kaldığı için orada kenar yumuşatma
kapalı — bu tek değişiklik `neon`'u 7,1 ms'den 4,1 ms'ye indirdi.

## Tip bilgisi

Paket `py.typed` gönderir; genel arayüzün tipleri CI'da `mypy` ile denetlenir.
