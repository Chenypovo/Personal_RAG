from pathlib import Path

from app.loader.image_loader import load_image


def test_load_image_uses_injected_ocr(tmp_path):
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(b"not-a-real-png-but-path-exists")

    doc = load_image(str(image_path), ocr_fn=lambda _: "hello from ocr")

    assert doc["entries"][0]["text"] == "hello from ocr"


def test_load_image_falls_back_when_ocr_fails(tmp_path):
    image_path = tmp_path / "cat.png"
    image_path.write_bytes(b"still-fine")

    doc = load_image(str(image_path), ocr_fn=lambda _: (_ for _ in ()).throw(RuntimeError("ocr failed")))

    assert doc["entries"][0]["text"] == f"Image source: {Path(image_path).name}"
