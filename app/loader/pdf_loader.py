from pathlib import Path
from typing import List

from pypdf import PdfReader


def _extract_text_with_pypdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    parts: List[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _extract_text_with_ocr(file_path: str, dpi: int = 200) -> str:
    try:
        import fitz  # pymupdf
    except Exception as exc:
        raise RuntimeError("OCR fallback requires 'pymupdf'") from exc

    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError("OCR fallback requires 'rapidocr-onnxruntime'") from exc

    import numpy as np

    ocr = RapidOCR()
    doc = fitz.open(file_path)
    parts: List[str] = []
    try:
        zoom = max(dpi / 72.0, 1.0)
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            result = ocr(img, use_det=True, use_cls=True, use_rec=True)
            lines = result[0] if isinstance(result, tuple) and len(result) > 0 else result
            if not lines:
                continue
            text_lines: List[str] = []
            for item in lines:
                # RapidOCR returns [box, text, score]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    txt = str(item[1]).strip()
                    if txt:
                        text_lines.append(txt)
            if text_lines:
                parts.append("\n".join(text_lines))
    finally:
        doc.close()

    return "\n\n".join(parts).strip()


def load_pdf(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {file_path}")

    full_text = _extract_text_with_pypdf(str(path))
    if not full_text:
        # Fallback for scanned/image-only PDFs
        try:
            full_text = _extract_text_with_ocr(str(path))
        except Exception as exc:
            raise ValueError(
                f"No extractable text in PDF and OCR fallback failed: {file_path}. "
                f"Detail: {exc}"
            ) from exc

    if not full_text:
        raise ValueError(f"No extractable text in PDF after OCR fallback: {file_path}")

    return {
        "source": str(path),
        "file_type": "pdf",
        "text": full_text,
    }
