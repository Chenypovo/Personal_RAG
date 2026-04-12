from pathlib import Path

from app.loader.md_loader import load_md
from app.loader.pdf_loader import load_pdf
from app.loader.txt_loader import load_text
from app.loader.image_loader import load_image, IMAGE_EXTENSIONS
from app.loader.video_loader import load_video, VIDEO_EXTENSIONS

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"} | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS



def load_document(file_path: str) -> dict:
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return load_text(file_path)
    if ext == ".md":
        return load_md(file_path)
    if ext == ".pdf":
        return load_pdf(file_path)
    if ext in IMAGE_EXTENSIONS:
        return load_image(file_path)
    if ext in VIDEO_EXTENSIONS:
        return load_video(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


__all__ = ["SUPPORTED_EXTENSIONS", "load_document", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS"]
