# Komut satırı

Paket kurulunca `cvflair` komutu da geliyor. Kamerayı, bir videoyu, tek bir
görseli ya da bir klasör dolusu görseli tek satırda işaretler:

```bash
cvflair 0 --theme neon --model yolov8n.pt
```

`python -m cvflair` de aynı komuttur; sanal ortam yolunu karıştırmadan çalıştırmak
için pratiktir.

## Kaynaklar

| Kaynak | Yazılışı | Varsayılan davranış |
|---|---|---|
| Kamera | `0`, `1`, ... | Pencere açar |
| Video dosyası | `girdi.mp4` | Pencere açar |
| Görsel | `kare.png` | Yanına `kare-cvflair.png` yazar |
| Görsel klasörü | `fotograflar/` | Her görselin yanına yazar |

`-o` verildiğinde çıktı oraya gider: video için dosya, çoklu görsel için klasör.

```bash
cvflair girdi.mp4 --model yolov8n.pt -o cikti.mp4 --theme cyberpunk
cvflair fotograflar/ --model yolov8n.pt -o isaretli/
```

Video yazarken pencere açılmaz; canlı kamerada ise yazarken de açık kalır, yoksa
ne kaydedildiği görünmez. `--no-window` ikisinde de pencereyi kapatır.

## Seçenekler

| Seçenek | Ne yapar |
|---|---|
| `--theme <ad>` | Tema (varsayılan `neon`); adları `--themes` listeler |
| `--model <yol>` | Ağırlık dosyası, örn. `yolov8n.pt` |
| `-o, --output <yol>` | Çıktı dosyası ya da klasörü |
| `--fps <sayı>` | Çıktı hızı; verilmezse kaynağın kendi hızı kullanılır |
| `--codec <dört harf>` | Video codec'i (varsayılan `mp4v`) |
| `--no-window` | Pencere açma |
| `--themes` | Tema adlarını listele ve çık |

`--model` isteğe bağlıdır ama onsuz çizilecek bir şey olmaz: cvflair tespit
yapmaz, tespiti çizer. Model verilmediğinde komut bunu söyler ve kareleri olduğu
gibi geçirir.

Çıkış kodları: `0` başarılı, `1` çalışma hatası (kaynak yok, dosya açılamadı),
`2` kullanım hatası (bilinmeyen tema, eksik argüman).

## Tema dosyası

`--theme` hazır bir tema adı ya da bir `.json` dosyası alır. Playground'da
ayarladığın görünüm bayrağa sığmadığı için dosyayla taşınır: sayfadaki
**theme.json indir** düğmesi tam bunun için.

```bash
cvflair 0 --theme theme.json --model yolov8n.pt
```

Aynı dosya Python'dan da okunur ve yazılır:

```python
from cvflair import Theme, get_theme

Theme(box_style="sketch", palette=["#39FF14"], glow=True).save("theme.json")
theme = get_theme("theme.json")
```

Dosyaya yalnızca varsayılandan farklı ayarlar yazılır, yani paylaşılan dosya
neyin seçildiğini anlatır. `Theme.to_dict()` ve `Theme.from_dict()` aynı biçimi
sözlük olarak verir; `get_theme` sözlüğü de doğrudan kabul eder.

## Video yazma

Komut satırının altındaki yazıcı kütüphaneden de kullanılabilir:

```python
from cvflair import Camera, VideoWriter

cam = Camera(source="girdi.mp4", drop_frames=False, theme="neon")
with VideoWriter("cikti.mp4", fps=cam.source_fps or 25) as writer:
    for frame, detections in cam.stream(model="yolov8n.pt"):
        writer.write(cam.annotate(frame, detections))
```

Dosya ilk kareyle açılır, yani boyutu önceden bilmek gerekmez. Sonraki kareler
farklı boyutta gelirse yazıcı hata verir: OpenCV bu kareleri sessizce düşürür ve
dosya nedeni görünmeden eksik çıkar.

`drop_frames=False` dosya kaynağında önemlidir — canlı kamerada eski kareyi
düşürmek doğrudur, dosyada her kare gerekir. Kaynağın kendi hızı
`cam.source_fps`, kare sayısı `cam.frame_count` ile okunur; kamera bunları
bildirmezse ikisi de sıfır döner.

Codec kurulu OpenCV'den gelir, cvflair'den değil. `mp4v` hemen her yerde vardır;
dosya açılamazsa ilk denenecek yer orasıdır (`.avi` uzantısıyla `MJPG` de yaygın
bir alternatiftir).
