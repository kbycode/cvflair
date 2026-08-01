# Değişiklik günlüğü

Sürümleme [Semantic Versioning](https://semver.org/lang/tr/) izler.

## 0.13.0 — 2026-08-01

### Eklendi
- **`Detections.from_xywh(...)`** — köşe ve boyut veren kutuları okur. OpenCV
  cascade'leri ve MediaPipe kutuyu böyle bildiriyor; dönüşüm herkesin kendi
  yazdığı satırdı. Model 0-1 aralığında veriyorsa kare boyutu geçilerek
  ölçekleniyor.
- **`Detections.from_mediapipe(sonuc, genislik, yukseklik)`** — MediaPipe'ın yüz
  ve nesne tespitini okur. Eski `solutions` sonucu oranlı kutu, yeni `tasks`
  sonucu piksel veriyor; ikisi de tanınıyor. MediaPipe import edilmiyor.
- **`FACE_5` iskeleti** — yüz modellerinin ortak beş noktası (iki göz, burun,
  iki ağız köşesi). InsightFace'in `kps` alanı, RetinaFace ve MediaPipe aynı
  sırayı kullanıyor. `skeleton="face"` ile de çağrılıyor; gözler arasına çizgi
  çekilmiyor, yüzü ortadan bölüyor.
- Playground'un iskelet önizlemesine yüz eklendi; belgelerdeki iskelet görseli de
  üç topolojiyi gösteriyor.
- İskelet önizlemesindeki çizimlerin altına silüet kondu. Kutu kipinde çizim bir
  figürün üstünde duruyordu, iskelet kipinde çizgiler boşluktaydı ve özellikle
  yüz beş noktadan ibaret bir işarete benziyordu. Silüetler kutulardaki
  figürlerle aynı gri tonda ve iskeletin altında: çizimin kendisi değişmiyor,
  yalnızca oturacağı bir gövde kazanıyor.

### Değişti
- İskelet listesine bağlı iki test artık listeyi sabit saymıyor; yeni bir
  topoloji eklenince kendiliğinden kapsıyor.

## 0.12.0 — 2026-08-01

### Eklendi
- **Tema dosyası.** `theme.save("theme.json")` ve `get_theme("theme.json")` —
  `--theme` de artık hazır bir ad ya da bir `.json` yolu alıyor. Dosyaya
  yalnızca varsayılandan farklı ayarlar yazılıyor, yani paylaşılan dosya neyin
  seçildiğini anlatıyor. `Theme.to_dict()` / `Theme.from_dict()` aynı biçimi
  sözlük olarak veriyor; `get_theme` sözlüğü de kabul ediyor.
- **Playground'da komut satırı.** Üretilen Python kodunun altında aynı görünümü
  veren `cvflair` komutu duruyor. Hazır bir temada `--theme neon` yetiyor; özel
  görünüm bir bayrağa sığmadığı için **theme.json indir** düğmesi dosyayı
  üretiyor ve komut onu kullanıyor.

### Depo
- Playground sayfası CI'da denetleniyor
  ([`tools/check_playground.py`](tools/check_playground.py)): betik `node --check`
  ile ayrıştırılıyor, sayfadaki çerçeve biçimi / panel konumu / tema listeleri
  kütüphanedekilerle karşılaştırılıyor ve sayfanın ürettiği tema alanlarının
  `Theme`'de gerçekten bulunduğu doğrulanıyor. Sayfanın derleme adımı yok: bozuk
  bir sözdizimi hiçbir yerde hata vermeden yayınlanıyor ve sayfa sessizce boş
  açılıyordu.

## 0.11.0 — 2026-08-01

### Eklendi
- **`cvflair` komutu.** Kamerayı, bir videoyu, tek bir görseli ya da bir klasör
  dolusu görseli kod yazmadan işaretler: `cvflair girdi.mp4 --model yolov8n.pt
  -o cikti.mp4 --theme neon`. `python -m cvflair` de aynı komut.
  Bkz. [komut satırı](docs/komut-satiri.md).
- **`VideoWriter`** — işaretli kareleri dosyaya yazar. Dosya ilk kareyle açılır,
  boyutu önceden bilmek gerekmez; sonraki kareler farklı boyutta gelirse hata
  verir (OpenCV bunları sessizce düşürüp dosyayı eksik bırakıyor).
- `Camera.source_fps` ve `Camera.frame_count` — kaynağın kendi bildirdiği hız ve
  kare sayısı. İşaretli kopyayı doğru hızda yazmak için gerekiyordu.
- **Not defteri gösterimi:** `cvflair.notebook.show(frame)` kareyi Jupyter ve
  Colab hücresinde gösterir; `to_png` de baytları verir. `cv2.imshow` orada
  çalışmıyor, matplotlib'e geçmek ise BGR/RGB takasını kullanıcıya bırakıyordu.
  IPython bağımlılık değil, yalnızca çağrıldığında import ediliyor.
- **`KeyPoints.from_mediapipe(...)`** — MediaPipe'in el ve poz sonucunu okur.
  Eski `solutions` ve yeni `tasks` API'leri ile düz nokta listesi kabul edilir;
  ölçekleme burada yapılır. MediaPipe import edilmiyor, alanlar adlarıyla
  okunuyor.

### Düzeltildi
- **Video dosyasının son karesi kayboluyordu.** Okuyucu tüketiciden bir adım
  önde bittiği için, kaynak tükendiğinde kuyrukta bekleyen kare işlenmeden
  döngü kapanıyordu; 40 karelik bir dosya 39 kare olarak yazılıyordu. Artık
  kaynak bittiğinde kuyruk boşaltılıyor, kullanıcı çıktığında ise
  boşaltılmıyor — o durumda bekleyen kare işlenmemeli.

### Belgeler
- README İngilizceye çevrildi; Türkçesi [README.tr.md](README.tr.md) olarak
  duruyor ve ikisi birbirine bağlı. Ayrıntılı belgeler Türkçe kalıyor. Paketin
  PyPI sayfası artık İngilizce.

### Depo
- `__all__` ile gerçekten dışa aktarılanları karşılaştıran test. Bir ad `__all__`
  listesine eklenip import'u unutulduğunda hata ancak kullanıcı o adı çağırınca
  çıkıyordu; ruff bunu `__init__.py` içinde yakalamıyor.

## 0.10.0 — 2026-08-01

### Eklendi
- **Etiket çakışması çözülüyor.** Kutular üst üste bindiğinde plakalar birbirini
  eziyordu; artık boş bir yere kaydırılıyor (kutu içi, altı, yanı, bir kat yukarı)
  ve uzaklaşan plakaya kutuya giden ince bir işaretçi çizgisi çekiliyor. Hiçbir
  yer boş değilse varsayılana dönülüyor. `Theme(avoid_label_overlap=False)` ile
  kapatılabilir.
- **Güven barı:** `Theme(confidence_bar=True)` kutunun altına skorla orantılı ince
  bir çubuk çizer. Güveni olmayan tespitlerde çizilmez.
- **Kutu içini gizleme:** `Theme(hide="blur")` ya da `"pixelate"`. Çerçeveden önce
  uygulanıyor, böylece kutu çizgisi ve etiket gizlemenin üstünde net kalıyor.
- **Bölge çizimi:** `theme.annotate_zone(scene, points, fill_opacity=...)` poligon
  ya da açık çizgi çizer. Yalnızca çizim: bir tespitin bölgenin içinde olup
  olmadığına cvflair karar vermiyor.
- **Segmentasyon maskesi.** `Detections.mask` maskeleri taşıyor, `Theme(masks=...,
  mask_opacity=..., mask_outline=...)` çiziyor; maske yoksa yok sayılıyor. Maliyeti
  46 ms'den 10,4 ms'ye indi: iş tüm kare yerine nesnenin penceresine indirildi ve
  dizin atama yerine OpenCV'nin maskeli kopyası kullanıldı.
- **`sketch` çerçeve biçimi** — elle çizilmiş gibi titrek, iki kere üstünden
  geçilmiş dikdörtgen. Titreşim kutunun konumundan türetilen tohumla üretiliyor:
  aynı nesne kare kare aynı çiziliyor, desen ekranda kaynamıyor. `wobble` ve
  `sketch_passes` ile ayarlanır.
- **Nabız:** `Theme(pulse=True)` kutunun çevresinde açılıp sönen bir halka çizer.
  Evre saatten okunuyor, duran karede bile hareket ediyor; kaydedilen videoda ya
  da testte tekrarlanabilir olması için `annotate(..., moment=...)` verilebilir.
- **Takip izi:** `Theme(trace=True, trace_length=...)` takip edilen nesnelerin
  geçtiği yolu çiziyor, iz geriye doğru inceliyor ve soluyor. Yalnızca
  `tracker_id` taşıyan tespitlerde çalışıyor — cvflair takip etmiyor. Bir süre
  görünmeyen kimlikler unutuluyor; `theme.reset_trace()` hepsini siliyor.

### Değişti
- Ölçüm sayıları yenilendi ve ölçüt en iyi kare süresine çevrildi: ortanca, arka
  plan yüküyle koşudan koşuya %20 oynuyordu.

## 0.9.0

### Eklendi
- **Nokta ve iskelet çizimi.** `KeyPoints` taşıyıcısı, `EdgeAnnotator` (kemikler)
  ve `VertexAnnotator` (eklemler); hazır topolojiler `HAND_21` (MediaPipe Hands
  sırası) ve `POSE_17` (COCO / YOLO-pose). Kendi kenar listeni de verebilirsin.
  Poz tahmini yapılmıyor — noktalar kullanıcının modelinden geliyor.
- `Theme.annotate_keypoints(...)` ve `Camera.show(..., keypoints=..., skeleton=...)`.
  Kemikler paletin rengini, eklemler vurgu rengini alıyor; `glow` kemiklerin
  arkasına koyu geçiş çiziyor. Yeni tema alanları: `pose_thickness`,
  `pose_radius`, `pose_confidence`.
- `KeyPoints.from_normalized(...)` — MediaPipe gibi 0-1 aralığında veren
  kütüphaneler için ölçekleme.
- [`examples/hand_skeleton.py`](examples/hand_skeleton.py) ve
  [belgesi](docs/noktalar.md).
- Playground'da **iskelet önizleme kipi**: el ve poz iskeleti canlı çiziliyor,
  kemik/eklem ayarları oradan yapılıyor. Kutu ve etiket ayarları o kipte
  gizleniyor, üretilen kod da iskelet kullanımını gösteriyor — iki aile
  birbirine karışmıyor. Palet altındaki adlar da kipe göre değişiyor
  (kutularda sahnedeki nesneler, iskelette el ve poz).

## 0.8.1

### Düzeltildi
- Bozuk kutu artık akışı düşürmüyor. `NaN`, sonsuz ve int32 sınırını aşan
  koordinatlarda OpenCV hata fırlatıyordu; tek bir bozuk tespit bütün döngüyü
  durduruyordu. Çizilemeyen kutu atlanıyor, aynı karedeki diğerleri çiziliyor;
  sonlu ama uçuk koordinatlar güvenli aralığa kırpılıyor (kadraja değil).

### Belgeler
- README vitrin haline getirildi (304 → 119 satır); ayrıntı `docs/` altında üç
  markdown dosyasına bölündü: [temalar](docs/temalar.md), [modeller](docs/modeller.md),
  [api](docs/api.md). Site üreteci eklenmedi, GitHub kendisi render ediyor.
- README'deki bütün bağlantılar mutlak GitHub adresine çevrildi; göreli yollar
  aynı dosya PyPI sayfası olarak kullanıldığında kırılıyordu.

## 0.8.0

### Eklendi
- `Camera.key` ve `Camera.pressed(tuş)` — `show()` artık basılan tuşu yutmuyor,
  demolar klavyeye cevap verebiliyor. Çıkış tuşları (`q`, `Q`, ESC) aynı şekilde
  çalışmaya devam ediyor.
- `examples/theme_switcher.py` — rakam tuşlarıyla tema, `b` ile sayaç paneli.

### Düzeltildi
- `Camera.close()` artık pencere kapatılamadığında hata fırlatmıyor (başsız
  ortamda ya da pencere zaten gitmişse).

### Depo
- PyPI yayını otomatikleşti: `v*` etiketi push edilince paket derlenip yükleniyor.
  Kimlik doğrulama token değil, GitHub'ın imzaladığı kısa ömürlü OIDC kimliği
  (trusted publishing); depoda ve diskte tutulan bir sır yok. Yükleme `pypi`
  environment'ındaki onay kuralını bekliyor.

## 0.7.0

### Hızlandı
- `neon` teması 7,1 ms/kare yerine 4,1 ms (1280x720, 8 kutu). Hâle geçişinde kenar
  yumuşatma kapatıldı: yay çizimi düz çizgiye göre dört kat pahalı, ama hâle sönük
  ve ana çizginin altında kaldığı için fark gözle görünmüyor (belirgin farklı
  piksel %0,4).

### Eklendi
- `tools/benchmark.py` — tema başına kare süresi, elle yazılmış OpenCV döngüsüne
  göre ek yük ve tema kurulum maliyeti. Ölçüm README'deki tabloyu üretiyor.

### Değişti
- Playground'daki renk paleti artık hangi rengin neye gittiğini gösteriyor: her renk
  kutusunun altında önizlemedeki nesnenin adı, kullanılmayanlar soluk, altında da
  dağıtım kuralı yazıyor.
- Playground'daki kamera önizlemesi kaldırıldı. Tarayıcıda tespit çalışmadığı
  için kutular sahte kalıyordu; ayarları görmeye bir katkısı yoktu, kamera izni
  istemesi ise gereksizdi.

### Depo
- Tip denetimi CI'ya eklendi (`mypy`). Paket `py.typed` gönderiyordu ama tip iddiası
  denetlenmiyordu; ilk koşuda üç gerçek sorun çıktı ve düzeltildi (annotator
  kurucularına giden sözlüğün tiplerini kaybetmesi, okuma döngüsünün `capture`
  değerini `None` olmayan varsayması, isteğe bağlı `ultralytics` importu).
- Issue şablonları (hata / öneri), PR şablonu ve [CONTRIBUTING.md](CONTRIBUTING.md).

## 0.6.0

### Eklendi
- `Camera(drop_frames=False)` — okuyucu tüketiciyi bekler. Canlı kamerada eski kareyi
  düşürmek doğru, ama video dosyasında her kare gerekli; dosya işlerken kare kaybı
  bununla ortadan kalkıyor.
- Dört yeni örnek ve [örnek galerisi](examples/README.md): `motion_detection.py`
  (arka plan çıkarımıyla gerçek tespit, sinir ağı ve indirme yok), `hud_stats.py`
  (paneli kendi verinle beslemek), `video_file.py` (dosyayı işleyip işaretlenmiş
  kopyasını yazmak), `image_folder.py` (toplu görsel işaretleme).
- Playground'da İngilizce/Türkçe seçeneği; varsayılan İngilizce, seçim tarayıcıda
  saklanıyor ve `?lang=tr` ile de verilebiliyor. Sayfa başlığı her zaman İngilizce.

## 0.5.0

### Eklendi
- **`hud` teması** — ince köşe çentikleri ve köşede yarı saydam bir sayaç paneli.
  Panel kutulardan değil döngüden beslenir: `Camera` kare hızını ve tespit sayısını
  kendisi doldurur, `cam.show(frame, detections, stats={"Skor": 12})` ile satır
  eklenebilir.
- `HudAnnotator` ve `Theme.hud` / `hud_position` / `hud_opacity`. Panel dört köşeden
  birine konabiliyor (`cvflair.HUD_POSITIONS`); plaka sahneye harmanlanıyor, yazı tam
  opaklıkta kalıyor.
- `Camera.measured_fps` — son 30 karenin ortalamasıyla ölçülen gerçek hız; cihazdan
  istenen `fps` değil, döngünün ulaştığı değer.
- `Theme.annotate(..., stats=...)`: panel verisi buradan geçiyor, veri yoksa panel de
  çizilmiyor. Tespit olmasa bile panel görünür.

## 0.4.0

### Belgeler
- Demo GIF'i artık gerçek bir sokak fotoğrafı (`docs/city.png`) üzerinde: kutular
  görseldeki bisiklet, köpek ve yayanın üzerine oturuyor, kareden kareye hafifçe
  oynuyor. `tools/make_demo_gif.py --background <görsel>` ile başka bir arka plan
  verilebiliyor (kutu oranlarının yeniden ayarlanması gerekir).
- Playground'un açılış ayarı: kesikli köşe çerçeve, kalınlık 2, köşe uzunluğu 8,
  çizgi 5, boşluk 7. Demo GIF'i de aynı çerçeve ayarlarını kullanıyor.
- Playground'un önizleme tuvali artık gösterildiği boyutta çiziliyor (devicePixelRatio
  dahil). Önceden sabit 880 px'lik tuval panele küçültülerek sığdırıldığı için etiket
  yazısı ekranda ~9 px'e düşüp okunmaz hâle geliyordu.

### Değişti — bağımlılıklar
- **Kurulum artık yalnızca `numpy` ve `opencv-python` çekiyor.** `import cvflair`
  2.9 saniyeden ~0.3 saniyeye, kurulum ~380 MB'tan ~170 MB'a indi; scipy, matplotlib,
  pillow ve fonttools artık hiç gelmiyor. Sürüm kilidi de ortadan kalktı.
- Çizim, tespit nesnesini alan adlarına göre okuduğu için başka kütüphanelerin
  tespitleri ve paletleri dönüştürülmeden çalışmaya devam ediyor;
  `tests/test_supervision_compat.py` bunu doğruluyor.

### Eklendi
- `cvflair.Detections` — kutu taşıyıcısı; `from_ultralytics` ve `from_arrays`.
  Alan adlarına göre okuma yapıldığı için yabancı tespit nesneleri de çalışıyor.
- `cvflair.colors`: `Color`, `ColorPalette`, `ColorLookup`. Paletler artık düz hex
  listesi olarak yazılabiliyor: `Theme(palette=["#39FF14"], text_color="#101010")`.
- `BoxAnnotator`, `RoundBoxAnnotator`, `BoxCornerAnnotator`, `LabelAnnotator` —
  temel çizim de artık `cvflair.annotators` içinde.

### Kırıcı olabilir
- `Theme.palette` / `accent_palette` / `text_color` alanları artık `cvflair` renk
  tiplerine dönüştürülüyor; yabancı palet vermek çalışıyor, ancak geri okunduğunda
  `cvflair.ColorPalette` dönüyor.
- `Detections.from_ultralytics` segmentasyon maskelerini ve döndürülmüş kutuları
  taşımıyor; o çıktılar dışarıdan hazır tespit nesnesi olarak verilebilir.

## 0.3.0

### Eklendi
- Beş yeni çerçeve biçimi (`cvflair.annotators`): `dashed` (kesikli), `dashed_corner`
  (kesikli çerçeve + üstüne dolu köşe ayraçları), `bracket` (yuvarlak dirsekli köşe ayracı),
  `crosshair` (kenar ortası çentikleri + merkez artı), `target` (ince çerçeve + kalın
  köşeler). Palet ve `ColorLookup` davranışı diğer biçimlerle aynı; yalnızca
  çizgi geometrisi farklı.
- `Theme.accent_palette`: köşe ayraçları, ayraç dirsekleri, artı merkezi ve hedef
  köşeleri için ikinci renk. Hibrit biçimlerin iki katmanı bununla ayrışıyor;
  `glow` açıkken vurgu rengi de koyulaşır.
- `cyberpunk` teması — yüksek kontrastlı palet, `target` biçimi, beyaz vurgu.
- Yeni tema alanları: `dash_length`, `gap_length`, `arm_length`, `center_size`,
  `edge_thickness`. Geçerli biçimler `cvflair.BOX_STYLES` içinde.
- Playground sekiz biçimi de destekliyor; ilgili ayarlar seçilen biçime göre görünüyor.

### Eklendi (playground)
- [Tema playground](https://kbycode.github.io/cvflair/) (`docs/index.html`): temalar
  tarayıcıda ayarlanıyor, hazır `Theme(...)` kodu kopyalanabiliyor. Sunucu, hesap ve
  kayıt yok; ayarlar query string ile taşınıyor, isteğe bağlı kamera önizlemesi
  görüntüyü sayfadan çıkarmıyor. Bağlantıdan gelen değerler yalnızca hex renk, sayı
  ve sabit sözcük olarak kabul edilip aralığa kırpılıyor.

## 0.2.0

### Eklendi
- `Camera.stream(model=...)`: model verildiğinde akış `(kare, tespitler)` çifti üretir.
- `cvflair.models`: ağırlık yolu, hazır Ultralytics modeli veya tespit döndüren
  herhangi bir çağrılabilir aynı detektör arayüzüne çevriliyor.
- `UltralyticsDetector` — `conf`, `iou`, `device` gibi ayarları her çağrıya taşır,
  `verbose` varsayılan olarak kapalı.
- `yolo` extra'sı (`pip install "cvflair[yolo]"`). Ultralytics AGPL-3.0 olduğu için
  zorunlu bağımlılık değil; eksikse hata mesajı extra'yı ve lisans gerekçesini söylüyor.
- `examples/yolo_quickstart.py`.

### Paketleme
- `MANIFEST.in`: sdist artık `conftest.py`, `CHANGELOG.md`, örnekler ve `tools/`
  içeriyor — testler sdist'ten de çalışıyor.
- README görselleri mutlak `raw.githubusercontent.com` adreslerine taşındı; göreli
  yollar PyPI sayfasında render edilmiyor.
- `release` extra'sı (`build`, `twine`).
- GitHub Actions: `ci.yml` (Python 3.10-3.13 üzerinde lint + test, derleme ve
  `twine check`), `release.yml` (`v*` etiketinde trusted publishing ile PyPI yayını).

## 0.1.0

### Eklendi
- `Camera`: webcam açma, ayrı thread'de okuma, son kareyi tutan tek slotlu kuyruk,
  `q`/ESC ile biten `stream()` akışı.
- `Theme`: çizimi yapılandıran katman; `minimal`, `neon`, `pastel` temaları.
  Annotator'lar tema kurulurken bir kez oluşturuluyor.
- Türkçe README, `docs/demo.gif` üreten `tools/make_demo_gif.py`, kamerasız örnekler.
- Kamera gerektirmeyen pytest paketi (sahte `capture_factory`).
