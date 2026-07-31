# Değişiklik günlüğü

Sürümleme [Semantic Versioning](https://semver.org/lang/tr/) izler.

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
