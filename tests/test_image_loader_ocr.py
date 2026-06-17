from pathlib import Path

from app.loader.image_loader import load_image


def _make_png(tmp_path) -> str:
    from PIL import Image

    p = Path(tmp_path) / "pic.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(p)
    return str(p)


def test_ocr_text_written_into_entry(tmp_path):
    img = _make_png(tmp_path)
    doc = load_image(img, ocr_fn=lambda path: "Invoice total 99.00")

    entry = doc["entries"][0]
    assert "Invoice total 99.00" in entry["text"]
    assert entry["modality"] == "image"
    assert entry["image_path"] == img


def test_falls_back_to_placeholder_when_ocr_empty(tmp_path):
    img = _make_png(tmp_path)
    doc = load_image(img, ocr_fn=lambda path: "   ")

    entry = doc["entries"][0]
    assert entry["text"] == f"Image source: {Path(img).name}"


def test_falls_back_to_placeholder_when_ocr_raises(tmp_path):
    img = _make_png(tmp_path)

    def boom(path):
        raise RuntimeError("ocr engine down")

    doc = load_image(img, ocr_fn=boom)
    entry = doc["entries"][0]
    assert entry["text"] == f"Image source: {Path(img).name}"


def test_ocr_text_exposed_separately(tmp_path):
    img = _make_png(tmp_path)
    doc = load_image(img, ocr_fn=lambda path: "hello world")
    assert doc["entries"][0]["ocr_text"] == "hello world"
