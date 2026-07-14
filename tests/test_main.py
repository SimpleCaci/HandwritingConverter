from pathlib import Path

import cv2
import numpy as np

import main


def make_image(path: Path) -> None:
    image = np.full((180, 500), 255, dtype=np.uint8)
    cv2.putText(image, "Hello notes", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    assert cv2.imwrite(str(path), image)


def test_convert_images_uses_ocr_and_saves_debug_image(tmp_path, monkeypatch):
    source = tmp_path / "note.png"
    make_image(source)
    captured = {}

    def fake_ocr(image, lang, config):
        captured.update(shape=image.shape, lang=lang, config=config)
        return "Hello notes\n"

    monkeypatch.setattr(main.pytesseract, "image_to_string", fake_ocr)
    results = main.convert_images(
        [source], language="eng", page_segmentation=6, debug_dir=tmp_path / "debug"
    )

    assert results[0].text == "Hello notes"
    assert results[0].processed_image.is_file()
    assert captured["lang"] == "eng"
    assert "--psm 6" in captured["config"]


def test_writers_create_text_and_pdf(tmp_path):
    results = [
        main.ConversionResult(Path("page-one.jpg"), "First page"),
        main.ConversionResult(Path("page-two.jpg"), "Second page"),
    ]
    text_path = tmp_path / "notes.txt"
    pdf_path = tmp_path / "notes.pdf"

    main.write_text(results, text_path)
    main.write_pdf(results, pdf_path, "Class notes")

    assert "# page-one.jpg" in text_path.read_text(encoding="utf-8")
    assert text_path.read_text(encoding="utf-8").endswith("\n")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 500


def test_missing_input_is_reported(tmp_path):
    missing = tmp_path / "missing.jpg"
    try:
        main.convert_images([missing])
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected a missing input error")
