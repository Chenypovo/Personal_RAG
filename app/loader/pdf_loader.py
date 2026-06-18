from pathlib import Path

def load_pdf(file_path: str) -> dict:
    from pypdf import PdfReader  # lazy: only needed when loading a PDF

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {file_path}")

    sep = "\n\n"
    reader = PdfReader(str(path))
    parts: list[str] = []
    page_offsets: list[tuple[int, int]] = []  # (char offset in full_text, 1-based page no.)
    cursor = 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        if parts:
            cursor += len(sep)  # the separator inserted before this page
        page_offsets.append((cursor, page_no))
        parts.append(text)
        cursor += len(text)

    full_text = sep.join(parts)
    if not full_text.strip():
        raise ValueError(f"No extractable text in PDF: {file_path}")

    return {
        "source": str(path),
        "file_type": "pdf",
        "text": full_text,
        "page_offsets": page_offsets,
    }
