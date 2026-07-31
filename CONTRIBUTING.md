# Katkı rehberi

Proje boş zamanlarda geliştiriliyor. Issue ve PR'lar okunuyor, ama yanıt bazen
birkaç gün sürebilir — acele bir şey varsa çatallamak (fork) her zaman serbest.

## Kurulum

```bash
git clone https://github.com/kbycode/cvflair.git
cd cvflair
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Üç komut, hepsi hızlı:

```bash
pytest              # kamera gerektirmez
ruff check .        # biçim ve içe aktarma düzeni
mypy                # paket py.typed gönderiyor, tip iddiası denetleniyor
```

## Kapsam

cvflair tespitleri **çizer**; tespit, takip veya bölge mantığı içermez. Bir öneri
"kutuyu şöyle göstersek" ise doğru yerdedir; "kutuyu şöyle bulsak" ise değildir.

Çalışma zamanı bağımlılıkları **numpy ve opencv ile sınırlı**. Yeni bir zorunlu
bağımlılık, kurulumu ve `import` süresini doğrudan etkilediği için kabul edilmiyor;
isteğe bağlı özellikler extra olarak eklenebilir (`cvflair[yolo]` gibi).

## Kod

- Tanımlayıcılar ve docstring'ler İngilizce; kullanıcıya görünen belgeler Türkçe.
- Çizim nesneleri döngü içinde değil, tema kurulurken bir kez oluşturulur.
- Yeni bir çerçeve biçimi `cvflair/annotators.py` içine, `Theme._build_box_annotator`
  içindeki dağıtıma ve `BOX_STYLES` demetine eklenir.
- Testler kamera, model veya ağ gerektirmez. Kamera için `capture_factory` üzerinden
  sahte bir `VideoCapture` verilir; çizim testleri piksel sayar.

## Yeni tema eklemek

1. `cvflair/themes.py` içinde bir fabrika fonksiyonu yaz, `_THEMES` sözlüğüne ekle.
2. `tests/test_themes.py` içindeki tema listesini güncelle.
3. Görselleri üret: `python tools/make_style_sheet.py`.
4. README'deki tema tablosuna bir satır ekle.

Temayı önce tarayıcıda denemek en hızlısı:
[tema playground](https://kbycode.github.io/cvflair/) → "Kodu kopyala".

## Belgeler ve görseller

`docs/` altındaki görseller elle çizilmiyor, üretiliyor:

```bash
python tools/make_demo_gif.py     # docs/demo.gif
python tools/make_style_sheet.py  # docs/box-styles.png + docs/theme-*.png
python tools/benchmark.py         # README'deki performans tablosu
```

Çizim maliyetini değiştiren bir değişiklik yaptıysan `benchmark.py` çıktısını
öncesi/sonrası olarak PR'a ekle — tablo oradan güncelleniyor.

Kaynak görsel `docs/city.png`; kutu oranları iki araçta ortak.

## Sürüm

Sürüm numarası `src/cvflair/__init__.py` içindeki `__version__` — tek kaynak orası.
Yayın adımları depoda tutulmuyor; bakımcıya ait.
