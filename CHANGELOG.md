# Değişiklik günlüğü

Sürümleme [Semantic Versioning](https://semver.org/lang/tr/) izler.
Paket henüz PyPI'da yayınlanmadı; sürümler depo içi kilometre taşlarıdır.

## 0.3.0

### Eklendi
- Beş yeni çerçeve biçimi (`cvflair.annotators`): `dashed` (kesikli), `dashed_corner`
  (kesikli çerçeve + üstüne dolu köşe ayraçları), `bracket` (yuvarlak dirsekli köşe ayracı),
  `crosshair` (kenar ortası çentikleri + merkez artı), `target` (ince çerçeve + kalın
  köşeler). Hepsi `supervision`'ın `BaseAnnotator`
  arayüzünü ve `resolve_color` renk çözümlemesini kullanıyor, yani palet ve
  `ColorLookup` davranışı yerleşik annotator'larla aynı.
- `Theme.accent_palette`: köşe ayraçları, ayraç dirsekleri, artı merkezi ve hedef
  köşeleri için ikinci renk. Hibrit biçimlerin iki katmanı bununla ayrışıyor;
  `glow` açıkken vurgu rengi de koyulaşır.
- `cyberpunk` teması — yüksek kontrastlı palet, `target` biçimi, beyaz vurgu.
- Yeni tema alanları: `dash_length`, `gap_length`, `arm_length`, `center_size`,
  `edge_thickness`. Geçerli biçimler `cvflair.BOX_STYLES` içinde.
- Playground sekiz biçimi de destekliyor; ilgili ayarlar seçilen biçime göre görünüyor.

### Not
Bu sürüm, "çizim matematiği yeniden yazılmaz" ilkesinden bilinçli bir sapma: bu beş
biçim `supervision`'da yok, dolayısıyla bakım sorumluluğu cvflair'de.
Kutu/yuvarlak/köşe biçimleri hâlâ doğrudan `supervision`'dan geliyor.

### Eklendi (playground)
- [Tema playground](https://kbycode.github.io/cvflair/) (`docs/index.html`): temalar
  tarayıcıda ayarlanıyor, hazır `Theme(...)` kodu kopyalanabiliyor. Sunucu, hesap ve
  kayıt yok; ayarlar query string ile taşınıyor, isteğe bağlı kamera önizlemesi
  görüntüyü sayfadan çıkarmıyor. Bağlantıdan gelen değerler yalnızca hex renk, sayı
  ve sabit sözcük olarak kabul edilip aralığa kırpılıyor.

## 0.2.0

### Eklendi
- `Camera.stream(model=...)`: model verildiğinde akış `(kare, tespitler)` çifti üretir.
- `cvflair.models`: ağırlık yolu, hazır Ultralytics modeli veya `sv.Detections`
  döndüren herhangi bir çağrılabilir aynı detektör arayüzüne çevriliyor.
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
- `release` extra'sı (`build`, `twine`) ve [PyPI yayın rehberi](docs/pypi-yayin-rehberi.md).
- GitHub Actions: `ci.yml` (Python 3.10-3.13 üzerinde lint + test, derleme ve
  `twine check`), `release.yml` (`v*` etiketinde trusted publishing ile PyPI yayını).

## 0.1.0

### Eklendi
- `Camera`: webcam açma, ayrı thread'de okuma, son kareyi tutan tek slotlu kuyruk,
  `q`/ESC ile biten `stream()` akışı.
- `Theme`: `supervision` annotator'larını yapılandıran katman; `minimal`, `neon`,
  `pastel` temaları. Annotator'lar tema kurulurken bir kez oluşturuluyor.
- Türkçe README, `docs/demo.gif` üreten `tools/make_demo_gif.py`, kamerasız örnekler.
- Kamera gerektirmeyen pytest paketi (sahte `capture_factory`).
