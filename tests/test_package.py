"""Paketin dış yüzeyi: `__all__` gerçekten dışa aktarılanla aynı mı."""

from __future__ import annotations

import importlib

import cvflair


def test_every_exported_name_exists():
    """
    `__all__`'a bir ad eklenip import'u unutulduğunda hata ancak kullanıcı o adı
    çağırınca çıkıyor -- ruff bunu `__init__.py` içinde yakalamıyor.
    """
    missing = [name for name in cvflair.__all__ if not hasattr(cvflair, name)]

    assert missing == [], f"__all__ içinde olup import edilmeyen: {missing}"


def test_public_names_are_listed():
    """Dışa aktarılan sınıf ve fonksiyonlar `__all__` dışında kalmamalı."""
    exported = {
        name
        for name, value in vars(cvflair).items()
        if not name.startswith("_")
        and getattr(value, "__module__", "").startswith("cvflair")
    }

    assert exported - set(cvflair.__all__) == set()


def test_import_does_not_pull_in_optional_extras():
    """Ultralytics kurulu olmasa da paket import edilebilmeli."""
    import sys

    before = set(sys.modules)
    importlib.reload(cvflair)

    pulled = {name.split(".")[0] for name in set(sys.modules) - before}
    assert "ultralytics" not in pulled and "torch" not in pulled
