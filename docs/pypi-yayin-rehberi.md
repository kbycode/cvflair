# PyPI yayın rehberi

Bu belge cvflair'in PyPI'a nasıl yayınlandığını adım adım anlatır. Sonunda tek
ekrana sığan bir [hızlı özet](#9-hizli-ozet) var; ilk yayından sonra genelde o yeterli.

> **Geri alınamaz adım tek tane:** `twine upload`. PyPI'da aynı sürüm numarası ikinci
> kez yayınlanamaz; bir sürüm silinse bile numarası yeniden kullanılamaz. Yanlış giden
> bir yayının çaresi yeni bir yama sürümü çıkarmaktır. Bu yüzden yükleme en sonda ve
> öncesinde TestPyPI provası var.

Yayın için iki yol var:

| Yol | Ne zaman |
|---|---|
| **A — Yerel `twine`** | İlk yayın, tek kişilik bakım, hızlı yama |
| **B — GitHub Actions (trusted publishing)** | Etiket atınca otomatik yayın, depoda token tutmadan |

İkisinin de 1-4 arası adımları ortaktır.

## 0. Tek seferlik hazırlık

### Hesap

1. <https://pypi.org/account/register/> üzerinden hesap açılır.
2. İki adımlı doğrulama zorunludur (TOTP uygulaması veya donanım anahtarı).
3. Prova için ayrı bir hesap gerekir: <https://test.pypi.org/account/register/>
   (TestPyPI bağımsız bir sistemdir, PyPI parolası orada geçmez).

### API token (yol A için)

<https://pypi.org/manage/account/token/> → yeni token. İlk yayında proje henüz
olmadığı için token'ın kapsamı zorunlu olarak "tüm hesap" olur. **Yayından sonra o
token silinip yerine yalnızca `cvflair` projesine yetkili yeni bir token oluşturulur.**

Token `pypi-` ile başlar ve yalnızca bir kez gösterilir.

### Kimlik bilgisinin saklanması

Kullanıcı adı her zaman `__token__`, parola token'ın kendisidir. İki seçenek:

Ortam değişkeni (CI ve tek seferlik kullanım için uygun):

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...
```

Ya da ev dizinindeki `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-...

[testpypi]
username = __token__
password = pypi-...
```

`.pypirc` asla depoya girmez — depo kökündeki `.gitignore` bu dosyayı zaten dışlıyor.
Windows'ta dosyanın yeri `C:\Users\<kullanıcı>\.pypirc`, Unix'te `~/.pypirc`
(izinler `chmod 600` yapılmalı).

### İsim müsaitliği

<https://pypi.org/project/cvflair/> adresi **404** vermeli. (30 Temmuz 2026 itibarıyla
boştaydı.) İsim bir kez alındığında geri alınamaz, bu yüzden yayından hemen önce
tekrar bakılır.

## 1. Sürüm ve değişiklik günlüğü

- `src/cvflair/__init__.py` içindeki `__version__` yayınlanacak sürümü göstermeli.
  Paket sürümü buradan okunur (`pyproject.toml` → `dynamic = ["version"]`), tek kaynak
  budur.
- `CHANGELOG.md` bu sürümün başlığını ve değişikliklerini içermeli.
- Sürüm numarası [semantic versioning](https://semver.org/lang/tr/) izler:
  kırıcı değişiklik → major, yeni özellik → minor, düzeltme → patch.

## 2. Doğrulama

```bash
pip install -e ".[dev,release]"
ruff check .
pytest
```

İkisi de temiz geçmeden ilerlenmez.

## 3. Temiz derleme

Eski çıktılar yeni derlemeye karışmasın diye önce silinir:

```bash
rm -rf dist build src/cvflair.egg-info
python -m build
twine check dist/*
```

Beklenen sonuç: `dist/` altında bir `.whl` ve bir `.tar.gz`, `twine check` ikisi için
de `PASSED`. `twine check` README'nin PyPI'da render edilip edilmeyeceğini de denetler.

İçerik doğrulaması:

```bash
# Wheel yalnızca paketi ve tip işaretini taşımalı.
python -c "import zipfile,glob; print(*zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(), sep='\n')"

# sdist testleri de içermeli (conftest.py dahil), yoksa kaynak paketten test çalışmaz.
python -c "import tarfile,glob; print(*tarfile.open(glob.glob('dist/*.tar.gz')[0]).getnames(), sep='\n')"
```

Görseller hakkında: README'deki resimler `raw.githubusercontent.com` mutlak
adreslerini kullanır, çünkü göreli yollar PyPI sayfasında çalışmaz. Bu adreslerin
karşılığı olması için depo public olmalı ve `main` dalına push edilmiş olmalı.

## 4. TestPyPI provası

```bash
twine upload -r testpypi dist/*
```

Ardından **ayrı ve boş** bir sanal ortamda kurulum denenir. Bağımlılıklar TestPyPI'da
bulunmadığı için gerçek PyPI ek indeks olarak verilir:

```bash
python -m venv /tmp/cvflair-prova
/tmp/cvflair-prova/bin/pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple cvflair
/tmp/cvflair-prova/bin/python -c "import cvflair; print(cvflair.__version__, cvflair.available_themes())"
```

TestPyPI'daki proje sayfasında README'nin düzgün göründüğü ve görsellerin yüklendiği
kontrol edilir. TestPyPI'da da aynı sürüm iki kez yüklenemez; prova sürümleri için
`0.2.0rc1` gibi bir aday numarası kullanılabilir.

## 5-A. Yayın — yerel twine

```bash
twine upload dist/*
```

Yükleme bittiğinde <https://pypi.org/project/cvflair/> adresi açılır ve sayfa
kontrol edilir.

## 5-B. Yayın — GitHub Actions ile (trusted publishing)

Bu yolda depoda veya CI ayarlarında token tutulmaz; PyPI, GitHub'ın imzaladığı kısa
ömürlü bir kimlikle (OIDC) doğrular.

Tek seferlik PyPI ayarı: proje sayfası → *Publishing* → *Add a new publisher* →
GitHub. Proje henüz yayınlanmadıysa aynı form <https://pypi.org/manage/account/publishing/>
altında "pending publisher" olarak doldurulur. İstenen alanlar:

| Alan | Değer |
|---|---|
| Owner | `kbycode` |
| Repository name | `cvflair` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Depodaki [`.github/workflows/release.yml`](../.github/workflows/release.yml) bu ayarla
eşleşecek şekilde yazıldı; environment adı iki tarafta da aynı olmalı. Ayrıca GitHub'da
*Settings → Environments* altında `pypi` adlı environment oluşturulur (isteğe bağlı
onay kuralı eklenebilir).

Yayın, sürüm etiketi push edilerek tetiklenir:

```bash
git tag -a v0.2.0 -m "cvflair 0.2.0"
git push origin v0.2.0
```

İş akışı paketi derler, `twine check` çalıştırır ve PyPI'a yükler.

## 6. Yayın sonrası

```bash
git tag -a v0.2.0 -m "cvflair 0.2.0"   # yol A kullanıldıysa etiket burada atılır
git push origin v0.2.0
```

- README'nin rozet satırına PyPI rozetleri eklenir (yayın öncesinde eklenirse
  "not found" görünür, o yüzden bu adımda):

  ```markdown
  [![PyPI](https://img.shields.io/pypi/v/cvflair)](https://pypi.org/project/cvflair/)
  [![İndirme](https://img.shields.io/pypi/dm/cvflair)](https://pypi.org/project/cvflair/)
  ```

- GitHub deposunun *About* alanına `https://pypi.org/project/cvflair/` adresi eklenir.
- GitHub'da release oluşturulur; açıklamaya CHANGELOG'un ilgili bölümü konur.
- Temiz bir ortamda `pip install cvflair` ile son doğrulama yapılır.
- Hesap kapsamlı token silinip yerine proje kapsamlı token oluşturulur (yol A).
- Bir sonraki geliştirme turu için CHANGELOG'a yeni başlık açılır.

## 7. Sık karşılaşılan hatalar

| Hata | Sebep ve çözüm |
|---|---|
| `400 File already exists` | O sürüm zaten yayınlanmış. Sürüm numarası artırılır; üzerine yazma yok. |
| `403 Invalid or non-existent authentication` | Kullanıcı adı `__token__` değil, ya da token eksik/yanlış kopyalanmış (`pypi-` öneki dahil olmalı). |
| `403 The user ... isn't allowed to upload to project` | Token yanlış projeye kapsamlı ya da isim başkasına ait. |
| `InvalidDistribution` / metadata hatası | Eski `setuptools`/`twine`. `pip install -U build twine` sonrası temiz derleme. |
| README PyPI'da düz metin görünüyor | `readme` alanı `pyproject.toml`'da tanımlı olmalı; `twine check` bunu yakalar. |
| Sayfada görseller boş | Göreli yol kullanılmış ya da depo public/push edilmemiş. Mutlak `raw.githubusercontent.com` adresi gerekir. |
| TestPyPI'da bağımlılık bulunamıyor | `--extra-index-url https://pypi.org/simple` eklenmemiş. |
| Yüklenen pakette eski dosyalar var | `dist/` temizlenmeden derlenmiş. `rm -rf dist build src/cvflair.egg-info`. |

## 8. Bu depoda neyin hazır olduğu

- `pyproject.toml` — metadata, bağımlılıklar, `dynamic` sürüm, `yolo`/`dev`/`release` extra'ları.
- `MANIFEST.in` — sdist içeriği (testler, örnekler, `tools/`, CHANGELOG).
- `.github/workflows/ci.yml` — her push/PR'da Python 3.10-3.13 üzerinde lint + test,
  ayrıca derleme ve `twine check`.
- `.github/workflows/release.yml` — `v*` etiketinde derleme + PyPI yayını.
- `.gitignore` — `dist/`, `build/`, `.pypirc` dışarıda.

## 9. Hızlı özet

```bash
# 1. sürüm ve CHANGELOG güncellenir  (src/cvflair/__init__.py, CHANGELOG.md)
ruff check . && pytest                        # 2. doğrulama
rm -rf dist build src/cvflair.egg-info        # 3. temiz derleme
python -m build
twine check dist/*
twine upload -r testpypi dist/*               # 4. prova
twine upload dist/*                           # 5. yayın  (geri alınamaz)
git tag -a v0.2.0 -m "cvflair 0.2.0" && git push origin v0.2.0   # 6. etiket
```
