from pathlib import Path
from typing import Any, Callable, Dict, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# path -> recognized text
OcrFn = Callable[[str], str]

_default_engine = None


def _default_ocr(image_path: str) -> str:
    """Run OCR with rapidocr-onnxruntime, lazily loaded and cached."""
    global _default_engine
    if _default_engine is None:
        from rapidocr_onnxruntime import RapidOCR  # lazy: heavy, only when OCR runs

        _default_engine = RapidOCR()

    result, _elapsed = _default_engine(image_path)
    if not result:
        return ""
    # result is a list of [box, text, score]; join the recognized text lines
    lines = [item[1] for item in result if len(item) >= 2 and item[1]]
    return "\n".join(lines).strip()


def load_image(file_path: str, ocr_fn: Optional[OcrFn] = None) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix.lower()}")

    engine = ocr_fn if ocr_fn is not None else _default_ocr
    try:
        ocr_text = (engine(str(path)) or "").strip()
    except Exception:
        ocr_text = ""  # OCR failure must not break ingestion

    # OCR text makes the image searchable via BM25 + embedding; fall back to a
    # placeholder so the image still has a (non-empty) text field for retrieval.
    entry_text = ocr_text if ocr_text else f"Image source: {path.name}"

    return {
        "source": str(path),
        "file_type": "image",
        "modality": "image",
        "entries": [
            {
                "source": str(path),
                "file_type": "image",
                "modality": "image",
                "chunk_id": 0,
                "image_path": str(path),
                "text": entry_text,
                "ocr_text": ocr_text,
            }
        ],
    }
