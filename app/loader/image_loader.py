from pathlib import Path
from typing import Any, Dict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix.lower()}")

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
                "text": f"Image source: {path.name}",
            }
        ],
    }
