from pathlib import Path
from typing import Any, Dict, List

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_video(
    file_path: str,
    frame_interval_sec: float = 2.0,
    max_frames: int = 120,
    frames_root: str = "data/frames",
) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {path.suffix}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    frame_step = max(int(fps * frame_interval_sec), 1)

    save_dir = Path(frames_root) / path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    frame_idx = 0
    sample_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            sec = frame_idx / fps
            img_name = f"frame_{sample_idx:05d}.jpg"
            img_path = save_dir / img_name
            cv2.imwrite(str(img_path), frame)

            entries.append(
                {
                    "source": str(path),
                    "file_type": "video",
                    "modality": "image",
                    "chunk_id": sample_idx,
                    "image_path": str(img_path),
                    "time_sec": float(sec),
                    "text": f"Video frame at {sec:.2f}s from {path.name}",
                }
            )
            sample_idx += 1

            if sample_idx >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if not entries:
        raise ValueError(f"No frames extracted from video: {file_path}")

    return {
        "source": str(path),
        "file_type": "video",
        "entries": entries,
    }
