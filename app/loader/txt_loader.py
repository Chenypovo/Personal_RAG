from pathlib import Path

def load_text(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Not a .txt file: {file_path}")
    
    text = path.read_text(encoding = "utf-8", errors = "ignore").strip()
    if not text:
        raise ValueError(f"TXT file is empty: {file_path}")
    
    return {
        "source": str(path),
        "file_type": "txt",
        "text": text
    }