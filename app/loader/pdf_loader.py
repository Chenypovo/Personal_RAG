from pathlib import Path
from pypdf import PdfReader

def load_pdf(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {file_path}")
    
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(text)

    full_text = "\n\n".join(parts).strip()
    if not full_text:
        raise ValueError(f"No extractable text in PDF: {file_path}")

    return {
        "source": str(path),
        "file_type": "pdf",
        "text": full_text,
    }  