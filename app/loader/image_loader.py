from pathlib import Path
from typing import Any, Callable, Dict, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


OCRFn = Callable[[str], str]


def build_rapidocr_fn() -> OCRFn:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run_ocr(image_path: str) -> str:
        result, _ = engine(image_path)
        if not result:
            return ""
        lines = [str(item[1]) for item in result if len(item) >= 2 and str(item[1]).strip()]
        return "\n".join(lines).strip()

    return run_ocr


def load_image(file_path: str, ocr_fn: Optional[OCRFn] = None) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix.lower()}")

    extracted_text = ""
    resolved_ocr = ocr_fn
    if resolved_ocr is None:
        try:
            resolved_ocr = build_rapidocr_fn()
        except Exception:
            resolved_ocr = None

    if resolved_ocr is not None:
        try:
            extracted_text = (resolved_ocr(str(path)) or "").strip()
        except Exception:
            extracted_text = ""

    text = extracted_text or f"Image source: {path.name}"

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
                "text": text,
            }
        ],
    }
