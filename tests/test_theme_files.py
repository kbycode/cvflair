"""Temanın dosyaya yazılması ve geri okunması."""

from __future__ import annotations

import json

import pytest

from cvflair import Theme, get_theme
from cvflair.colors import ColorLookup


def custom() -> Theme:
    return Theme(
        name="playground",
        palette=["#39FF14", "#FF00E5"],
        accent_palette="#FFFFFF",
        box_style="sketch",
        thickness=3,
        wobble=5,
        pulse=True,
        trace=True,
        glow=True,
    )


# -- serileştirme -----------------------------------------------------------


def test_only_the_changed_settings_are_written():
    assert Theme().to_dict() == {"name": "custom"}


def test_field_order_follows_the_class():
    keys = list(custom().to_dict())

    assert keys[:4] == ["name", "palette", "accent_palette", "box_style"]


def test_colours_become_hex_strings():
    data = custom().to_dict()

    assert data["palette"] == ["#39FF14", "#FF00E5"]
    assert data["accent_palette"] == ["#FFFFFF"]


def test_enum_becomes_its_value():
    data = Theme(color_lookup=ColorLookup.TRACK).to_dict()

    assert data["color_lookup"] == "track"
    assert json.dumps(data), "çıktı JSON'a yazılabilir olmalı"


def test_round_trip_keeps_every_setting():
    original = custom()

    restored = Theme.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()


def test_round_trip_draws_the_same_thing():
    """Asıl ölçüt eşit sözlük değil, aynı görüntü."""
    import numpy as np

    from cvflair import Detections

    boxes = Detections(xyxy=[[20, 30, 120, 140]], class_id=[0], confidence=[0.9], names=["a"])
    first, second = (np.zeros((200, 200, 3), dtype=np.uint8) for _ in range(2))

    custom().annotate(first, boxes, moment=0.4)
    Theme.from_dict(custom().to_dict()).annotate(second, boxes, moment=0.4)

    assert np.array_equal(first, second)


def test_unknown_field_is_named():
    with pytest.raises(ValueError, match="renk_paleti"):
        Theme.from_dict({"renk_paleti": ["#FFFFFF"]})


# -- dosya ------------------------------------------------------------------


def test_save_then_load(tmp_path):
    path = custom().save(tmp_path / "theme.json")

    assert path.exists()
    assert Theme.load(path).to_dict() == custom().to_dict()


def test_saved_file_is_readable_json(tmp_path):
    custom().save(tmp_path / "theme.json")

    data = json.loads((tmp_path / "theme.json").read_text(encoding="utf-8"))
    assert data["box_style"] == "sketch"


def test_missing_folder_is_created(tmp_path):
    custom().save(tmp_path / "yeni" / "theme.json")

    assert (tmp_path / "yeni" / "theme.json").exists()


def test_broken_json_says_which_file(tmp_path):
    path = tmp_path / "bozuk.json"
    path.write_text("{ bozuk", encoding="utf-8")

    with pytest.raises(ValueError, match="bozuk.json"):
        Theme.load(path)


def test_json_holding_a_list_is_refused(tmp_path):
    path = tmp_path / "liste.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(ValueError, match="nesne içermeli"):
        Theme.load(path)


# -- get_theme --------------------------------------------------------------


def test_get_theme_reads_a_json_path(tmp_path):
    custom().save(tmp_path / "theme.json")

    theme = get_theme(str(tmp_path / "theme.json"))

    assert theme.name == "playground" and theme.box_style == "sketch"


def test_get_theme_accepts_a_mapping():
    theme = get_theme({"name": "elle", "box_style": "round"})

    assert theme.name == "elle" and theme.box_style == "round"


def test_get_theme_still_resolves_names():
    assert get_theme("neon").name == "neon"


def test_preset_names_are_not_treated_as_paths():
    """Ayrım uzantıda: 'neon' dosya diye aranmamalı."""
    with pytest.raises(ValueError, match="Unknown theme"):
        get_theme("neon.tema")


def test_each_call_builds_a_fresh_theme(tmp_path):
    custom().save(tmp_path / "theme.json")

    first = get_theme(str(tmp_path / "theme.json"))
    second = get_theme(str(tmp_path / "theme.json"))

    assert first is not second, "iki kamera aynı annotator'ları paylaşmamalı"


def test_presets_survive_the_round_trip():
    """Her hazır tema kendi dosyasından birebir geri kurulabilmeli."""
    from cvflair import available_themes

    for name in available_themes():
        original = get_theme(name)

        assert Theme.from_dict(original.to_dict()).to_dict() == original.to_dict(), name
