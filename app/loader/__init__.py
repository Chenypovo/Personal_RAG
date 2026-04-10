from pathlib import Path

from app.loader.md_loader import load_md
from app.loader.pdf_loader import load_pdf
from app.loader.txt_loader import load_text


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_document(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return load_text(file_path)
    if ext == ".md":
        return load_md(file_path)
    if ext == ".pdf":
        return load_pdf(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


__all__ = ["SUPPORTED_EXTENSIONS", "load_document"]
