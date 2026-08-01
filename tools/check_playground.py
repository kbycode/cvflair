"""
Playground sayfasının betiğini denetler.

`docs/index.html` tek dosyalık, derleme adımı olmayan bir sayfa: bozuk bir
sözdizimi hiçbir yerde hata vermeden yayınlanır, sayfa da sessizce boş açılır.
Bu betik script bloğunu çıkarıp `node --check` ile ayrıştırır ve sayfanın
Python tarafıyla uyuşması gereken sabitlerini karşılaştırır.

Node kurulu değilse sözdizimi denetimi atlanır, sabit karşılaştırması yine yapılır.

Çalıştırmak için:  python tools/check_playground.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cvflair import BOX_STYLES, Theme, available_themes
from cvflair.annotators import HUD_POSITIONS

PAGE = Path(__file__).resolve().parents[1] / "docs" / "index.html"


def script_of(html: str) -> str:
    match = re.search(r"<script>(.*)</script>", html, re.S)
    if match is None:
        raise SystemExit("Sayfada script bloğu bulunamadı.")
    return match.group(1)


def js_array(script: str, name: str) -> list[str]:
    """`const NAME = [ ... ];` içindeki dizgeleri okur."""
    match = re.search(rf"const {name} = \[(.*?)\];", script, re.S)
    if match is None:
        raise SystemExit(f"Sayfada {name} bulunamadı.")
    return re.findall(r'"([^"]+)"', match.group(1))


def check_syntax(script: str) -> list[str]:
    node = shutil.which("node")
    if node is None:
        print("node bulunamadı, sözdizimi denetimi atlandı")
        return []

    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "playground.js"
        target.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(target)], capture_output=True, text=True
        )
    if result.returncode:
        return [f"Betik ayrıştırılamıyor:\n{result.stderr.strip()}"]
    print("sözdizimi temiz")
    return []


def check_constants(script: str) -> list[str]:
    """Sayfadaki listeler kütüphanedekilerle aynı mı."""
    problems = []
    for name, expected in (
        ("BOX_STYLES", list(BOX_STYLES)),
        ("HUD_POSITIONS", list(HUD_POSITIONS)),
    ):
        found = js_array(script, name)
        if found != expected:
            problems.append(f"{name} sayfada {found}, kütüphanede {expected}")
        else:
            print(f"{name}: {len(found)} değer, kütüphaneyle aynı")

    presets = re.search(r"const PRESETS = \{(.*?)\n\};", script, re.S)
    if presets is None:
        problems.append("Sayfada PRESETS bulunamadı")
    else:
        found = re.findall(r"^  ([a-z_]+): \{", presets.group(1), re.M)
        expected = list(available_themes())
        if sorted(found) != sorted(expected):
            problems.append(f"PRESETS sayfada {sorted(found)}, kütüphanede {sorted(expected)}")
        else:
            print(f"PRESETS: {len(found)} tema, kütüphaneyle aynı")
    return problems


def block_of(script: str, name: str) -> str:
    """`const NAME = { ... };` bloğunu süslü parantezleri sayarak çıkarır."""
    start = script.index(f"const {name} = {{")
    depth = 0
    for index in range(script.index("{", start), len(script)):
        depth += {"{": 1, "}": -1}.get(script[index], 0)
        if depth == 0:
            return script[start : index + 1] + ";"
    raise SystemExit(f"{name} bloğu kapanmıyor.")


def page_themes(script: str) -> dict[str, dict] | None:
    """Sayfadaki tema tanımlarını node ile okur; node yoksa None."""
    node = shutil.which("node")
    if node is None:
        return None

    source = "\n".join(
        [block_of(script, name) for name in ("DEFAULTS", "PRESETS", "DEFAULT_STATE")]
        + ["console.log(JSON.stringify({...PRESETS, _start: DEFAULT_STATE}));"]
    )
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "themes.js"
        target.write_text(source, encoding="utf-8")
        result = subprocess.run([node, str(target)], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"Tema tanımları okunamadı:\n{result.stderr.strip()}")
    return dict(json.loads(result.stdout))


#: Sayfadaki ad -> Theme alanı; gerisi camelCase'den kendiliğinden çözülüyor.
SPECIAL_KEYS = {"accent": "accent_palette", "preview": None}


def to_theme_kwargs(settings: dict) -> dict:
    kwargs = {}
    for key, value in settings.items():
        field = SPECIAL_KEYS.get(key, re.sub(r"([A-Z])", lambda m: "_" + m.group(1), key).lower())
        if field is None:
            continue
        kwargs[field] = value
    return kwargs


def check_themes_build(script: str) -> list[str]:
    """
    Sayfadaki her hazır tema ve açılış görünümü Python'da gerçekten kurulabiliyor mu.

    Sayfa ayarları kendi adlarıyla tutuyor; bir alan yeniden adlandırılır ya da
    değer aralığı daralırsa sayfa yine çizmeye devam eder, kopyalanan kod ise
    kullanıcının makinesinde patlar.
    """
    themes = page_themes(script)
    if themes is None:
        print("node bulunamadı, tema kurulumu denenmedi")
        return []

    problems = []
    for name, settings in themes.items():
        kwargs = to_theme_kwargs(settings)
        try:
            Theme(name=name, **kwargs)
        except Exception as error:  # noqa: BLE001 - hepsi rapor ediliyor
            problems.append(f"{name} teması kurulamıyor: {error}")
    if not problems:
        print(f"tema kurulumu: {len(themes)} tanım Python'da kuruluyor")
    return problems


def check_theme_fields(script: str) -> list[str]:
    """`themeSettings()` içindeki alan adları Theme'de gerçekten var mı."""
    from cvflair.themes import _FIELD_SET

    body = re.search(r"function themeSettings\(\) \{(.*?)\n\}", script, re.S)
    if body is None:
        return ["Sayfada themeSettings bulunamadı"]

    used = set(re.findall(r'add\("([a-z_]+)"', body.group(1)))
    unknown = sorted(used - _FIELD_SET)
    if unknown:
        return [f"Sayfanın ürettiği tema dosyasında Theme'de olmayan alanlar: {unknown}"]
    print(f"tema alanları: {len(used)} tanesi kullanılıyor, hepsi Theme'de var")
    return []


def main() -> int:
    html = PAGE.read_text(encoding="utf-8")
    script = script_of(html)

    problems = (
        check_syntax(script)
        + check_constants(script)
        + check_theme_fields(script)
        + check_themes_build(script)
    )
    if problems:
        print("\n".join(f"HATA: {problem}" for problem in problems), file=sys.stderr)
        return 1

    print(json.dumps({"sayfa": str(PAGE.name), "durum": "tamam"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
