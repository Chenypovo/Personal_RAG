import re
from typing import Dict, List


def _find_split_pos(text: str, start: int, hard_end: int, min_ratio: float = 0.6) -> int:
    min_end = start + int((hard_end - start) * min_ratio)
    if min_end >= hard_end:
        return hard_end

    window = text[start:hard_end]
    relative_min = min_end - start

    # 1) Prefer paragraph break
    para_break = window.rfind("\n\n", relative_min)
    if para_break != -1:
        return start + para_break + 2

    # 2) Then sentence boundary
    sentence_matches = list(re.finditer(r"[。！？!?\.](?:\s|\n|$)", window))
    for m in reversed(sentence_matches):
        if m.end() >= relative_min:
            return start + m.end()

    # 3) Then newline
    line_break = window.rfind("\n", relative_min)
    if line_break != -1:
        return start + line_break + 1

    # 4) Fallback to hard cut
    return hard_end


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 700,
    overlap: int = 120,
) -> List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    clean = (text or "").strip()
    if not clean:
        return []

    chunks: List[Dict] = []
    start = 0
    idx = 0
    n = len(clean)

    while start < n:
        hard_end = min(start + chunk_size, n)
        end = hard_end if hard_end == n else _find_split_pos(clean, start, hard_end)

        if end <= start:
            end = hard_end

        piece = clean[start:end].strip()
        if piece:
            chunks.append(
                {
                    "chunk_id": idx,
                    "source": source,
                    "text": piece,
                    "start": start,
                    "end": end,
                }
            )
            idx += 1

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks
