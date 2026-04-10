from pathlib import Path

def load_md(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    
    if path.suffix.lower() != "md":
        raise ValueError(f"Not a .md file: {file_path}")
    
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Markdown file is empty: {file_path}")
    
    return {
        "source": str(path),
        "file_type": "md",
        "text": text,
    }