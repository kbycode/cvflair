# Temalar ve çerçeve biçimleri

Bir tema, çizimle ilgili bütün ayarların tek bir demeti: palet, çerçeve biçimi,
etiket görünümü, sayaç paneli. `Camera`'ya ad ya da nesne olarak verilir.

Ayarları kurulum yapmadan denemek için:
**[tema playground](https://kbycode.github.io/cvflair/)** — oynat, hazır
`Theme(...)` kodunu kopyala. Sayfa İngilizce açılır, sağ üstteki **TR** düğmesiyle
Türkçeye geçer.

## Hazır temalar

| Tema | Görünüm | |
|---|---|---|
| `minimal` | ince beyaz çerçeve, sade etiket — ekran kaydı ve profesyonel demo için | ![minimal](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-minimal.png) |
| `neon` | sınıf başına canlı renk, yuvarlak köşe, koyu hâle ile parlama hissi | ![neon](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-neon.png) |
| `pastel` | yumuşak tonlar, geniş yuvarlama, koyu etiket yazısı — atölye/projeksiyon | ![pastel](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-pastel.png) |
| `cyberpunk` | ince çerçeve + kalın beyaz köşeler, yüksek kontrast — hedef kilitleme görünümü | ![cyberpunk](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-cyberpunk.png) |
| `hud` | ince köşe çentikleri + köşede sayaç paneli — oyun ve robotik demoları | ![hud](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/theme-hud.png) |

```python
cam = Camera(source=0, theme="pastel")
cam.theme = "neon"                     # çalışırken de değiştirilebilir
```

Adları `available_themes()` listeler.

## Çerçeve biçimleri

![çerçeve biçimleri](https://raw.githubusercontent.com/kbycode/cvflair/main/docs/box-styles.png)

`box_style` sekiz değerden birini alır:

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
paleti ve `ColorLookup` davranışı sekizinde de aynıdır. Geçerli adlar
`cvflair.BOX_STYLES` demetinde.

### İkinci renk

`dashed_corner`, `bracket`, `crosshair` ve `target` bir vurgu rengi kabul eder;
köşe ayraçları, dirsekler ve merkez artısı o renge geçer. Hibrit biçimlerin iki
katmanı böyle ayrışır:

```python
theme = Theme(
    palette=["#00F0FF"],
    accent_palette="#FF206E",
    box_style="target",
    thickness=3,
)
```

## Renk paleti

Renkler sınıf sırasına göre dağıtılır: `class_id=0` olan tespit paletin ilk
rengini alır, `class_id=1` ikincisini, palet biterse başa döner.

```python
Theme(palette=["#39FF14", "#FF00E5", "#00E5FF"])
```

Hex dizgesi, dizge listesi, tek bir `Color` ya da `ColorPalette` — hepsi kabul
edilir. Dağıtım kuralı `color_lookup` ile değişir: `ColorLookup.CLASS`
(varsayılan), `ColorLookup.INDEX` (kutu sırası) veya `ColorLookup.TRACK`
(takip kimliği).

## Sayaç paneli

`hud` teması köşeye küçük bir panel çizer. Sayılar kutulardan değil döngüden
gelir, bu yüzden ayrı bir yoldan veriliyor — `Camera` kare hızını ve tespit
sayısını kendisi dolduruyor:

```python
cam = Camera(source=0, theme="hud")
for frame, detections in cam.stream(model="yolov8n.pt"):
    cam.show(frame, detections)          # FPS ve Objects panelde
```

Kendi satırlarını eklemek için `stats`; aynı anahtar verilirse seninki kazanır:

```python
cam.show(frame, detections, stats={"Skor": score, "Tur": lap})
```

Panel her temaya açılabilir:

```python
Theme(hud=True, hud_position="bottom_right", hud_opacity=0.5)
```

Konumlar `cvflair.HUD_POSITIONS` içinde. Plaka sahneye harmanlanır (arkası
görünmeye devam eder), yazı tam opaklıkta kalır. Tespit olmasa bile panel
çizilir; ama `stats` boşsa panel de yoktur.

## Kendi temanı yazmak

```python
from cvflair import Camera, Theme

my_theme = Theme(
    name="my-theme",
    palette=["#39FF14", "#FF00E5"],
    box_style="corner",
    thickness=2,
    glow=True,
    text_scale=0.6,
)
cam = Camera(source=0, theme=my_theme)
```

Alanların tamamı ve varsayılanları için `help(Theme)`. Birkaç davranış notu:

- Alanlar yalnızca kurulum anında okunur. Sonradan bir alanı değiştirmek çizim
  nesnelerini yeniden kurmaz — yeni bir `Theme` oluşturmak gerekir. Mevcut bir
  temanın türevi için `dataclasses.replace(theme, hud=True)` kullanılabilir.
- `glow` çerçeveyi iki kez çizer: arkada koyulaştırılmış kalın bir geçiş, önde
  ana çizgi. Vurgu rengi varsa o da koyulaşır.
- `get_theme("neon")` her çağrıda yeni bir nesne döndürür, iki kamera aynı
  çizim nesnelerini paylaşmaz.

Yeni bir tema eklemek (depoya katkı olarak):
[CONTRIBUTING.md](https://github.com/kbycode/cvflair/blob/main/CONTRIBUTING.md)
