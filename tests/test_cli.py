"""`cvflair` komutu. Hiçbir test kamera açmıyor, pencere açmıyor."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from cvflair import Detections, available_themes
from cvflair.cli import collect_images, image_target, main


def write_image(path, value: int = 40) -> None:
    image = np.full((60, 80, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


@pytest.fixture
def gallery(tmp_path):
    folder = tmp_path / "gorseller"
    folder.mkdir()
    for index in range(3):
        write_image(folder / f"kare{index}.png", 30 + index * 10)
    return folder


def fake_model(_frame) -> Detections:
    """Sabit bir kutu döndürür; ağırlık dosyası ve indirme gerektirmez."""
    return Detections(xyxy=[[10, 10, 50, 40]], class_id=[0], confidence=[0.9], names=["nesne"])


# -- kaynak çözümleme -------------------------------------------------------


def test_image_file_is_recognised(tmp_path):
    path = tmp_path / "tek.png"
    write_image(path)

    assert collect_images(str(path)) == [path]


def test_folder_is_listed_in_order(gallery):
    found = collect_images(str(gallery))

    assert [p.name for p in found] == ["kare0.png", "kare1.png", "kare2.png"]


def test_video_is_not_treated_as_an_image(tmp_path):
    path = tmp_path / "girdi.mp4"
    path.write_bytes(b"")

    assert collect_images(str(path)) is None


def test_camera_index_is_not_a_path():
    assert collect_images("0") is None


def test_empty_folder_is_reported(tmp_path):
    (tmp_path / "bos").mkdir()

    with pytest.raises(OSError, match="görsel bulunamadı"):
        collect_images(str(tmp_path / "bos"))


def test_output_naming(tmp_path):
    source = tmp_path / "kare.png"

    assert image_target(source, None, 1).name == "kare-cvflair.png"
    assert image_target(source, tmp_path / "cikti.png", 1).name == "cikti.png"
    assert image_target(source, tmp_path / "klasor", 3) == tmp_path / "klasor" / "kare.png"


# -- komut ------------------------------------------------------------------


def test_themes_are_listed(capsys):
    assert main(["--themes"]) == 0

    listed = capsys.readouterr().out.split()
    assert listed == list(available_themes())


def test_source_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "kaynak gerekli" in capsys.readouterr().err


def test_unknown_theme_lists_the_real_ones(tmp_path, capsys):
    assert main([str(tmp_path / "x.png"), "--theme", "yok"]) == 2

    assert "neon" in capsys.readouterr().err


def test_missing_source_is_reported(tmp_path, capsys):
    assert main([str(tmp_path / "yok.mp4"), "--no-window"]) == 1

    assert "bulunamadı" in capsys.readouterr().err


def test_folder_is_annotated_into_the_output_folder(gallery, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("cvflair.cli.resolve_detector", lambda model: fake_model)
    target = tmp_path / "isaretli"

    assert main([str(gallery), "-o", str(target), "--model", "sahte.pt"]) == 0

    written = sorted(p.name for p in target.iterdir())
    assert written == ["kare0.png", "kare1.png", "kare2.png"]
    assert "3 görsel işaretlendi" in capsys.readouterr().out


def test_single_image_is_written_next_to_the_source(tmp_path, monkeypatch):
    monkeypatch.setattr("cvflair.cli.resolve_detector", lambda model: fake_model)
    source = tmp_path / "kare.png"
    write_image(source)

    assert main([str(source), "--model", "sahte.pt"]) == 0

    assert (tmp_path / "kare-cvflair.png").exists()


def test_annotation_actually_changes_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr("cvflair.cli.resolve_detector", lambda model: fake_model)
    source = tmp_path / "kare.png"
    write_image(source, value=40)

    main([str(source), "--model", "sahte.pt", "--theme", "neon"])

    before = cv2.imread(str(source))
    after = cv2.imread(str(tmp_path / "kare-cvflair.png"))
    assert not np.array_equal(before, after), "çizim görselde görünmüyor"


def test_without_a_model_nothing_is_drawn(tmp_path, capsys):
    source = tmp_path / "kare.png"
    write_image(source)

    assert main([str(source)]) == 0

    assert "--model verilmedi" in capsys.readouterr().err
    before = cv2.imread(str(source))
    after = cv2.imread(str(tmp_path / "kare-cvflair.png"))
    assert np.array_equal(before, after), "model yokken kare değişmemeli"


def test_unreadable_image_is_skipped_not_fatal(tmp_path, capsys):
    good = tmp_path / "iyi.png"
    write_image(good)
    (tmp_path / "bozuk.png").write_bytes(b"bu bir png degil")

    assert main([str(tmp_path), "-o", str(tmp_path / "cikti")]) == 0

    output = capsys.readouterr()
    assert "atlanıyor" in output.err
    assert "1 görsel işaretlendi" in output.out
