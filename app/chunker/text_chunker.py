import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import tiktoken


SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?]|[\u3002\uFF01\uFF1F])(?:\s|\n|$)")
PARAGRAPH_BOUNDARY_RE = re.compile(r"\n{2,}")

EmbedFn = Callable[[Sequence[str]], List[List[float]]]


def _find_paragraph_break(window: str, relative_min: int) -> Optional[int]:
    # 1) Prefer paragraph boundary.
    pos = window.rfind("\n\n", relative_min)
    return None if pos == -1 else pos + 2


def _find_sentence_break(window: str, relative_min: int) -> Optional[int]:
    # 2) Then sentence boundary.
    matches = list(SENTENCE_BOUNDARY_RE.finditer(window))
    for m in reversed(matches):
        if m.end() >= relative_min:
            return m.end()
    return None


def _find_line_break(window: str, relative_min: int) -> Optional[int]:
    # 3) Then line break.
    pos = window.rfind("\n", relative_min)
    return None if pos == -1 else pos + 1


def _find_split_pos(text: str, start: int, hard_end: int, min_ratio: float = 0.6) -> Tuple[int, str]:
    min_end = start + int((hard_end - start) * min_ratio)
    if min_end >= hard_end:
        return hard_end, "hard"

    window = text[start:hard_end]
    relative_min = min_end - start

    para_break = _find_paragraph_break(window, relative_min)
    if para_break is not None:
        return start + para_break, "paragraph"

    sentence_break = _find_sentence_break(window, relative_min)
    if sentence_break is not None:
        return start + sentence_break, "sentence"

    line_break = _find_line_break(window, relative_min)
    if line_break is not None:
        return start + line_break, "line"

    # Fall back to hard cut only when no soft boundary is found.
    return hard_end, "hard"


def _token_char_spans(text: str, token_ids: List[int], encoding: tiktoken.Encoding) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        piece = encoding.decode([tid])
        start = cursor
        cursor += len(piece)
        spans.append((start, cursor))
    return spans


def _char_end_to_token_end(
    spans: List[Tuple[int, int]],
    token_start: int,
    char_end: int,
    hard_token_end: int,
) -> int:
    i = token_start
    while i < hard_token_end:
        if spans[i][1] >= char_end:
            return i + 1
        i += 1
    return hard_token_end


def _char_start_to_token_start(
    spans: List[Tuple[int, int]],
    char_start: int,
    token_hint: int = 0,
) -> int:
    i = max(token_hint, 0)
    n = len(spans)
    while i < n:
        # First token that overlaps char_start.
        if spans[i][1] > char_start:
            return i
        i += 1
    return n


def _trim_with_span(raw: str, start: int) -> Tuple[str, int, int]:
    left = 0
    right = len(raw)
    while left < right and raw[left].isspace():
        left += 1
    while right > left and raw[right - 1].isspace():
        right -= 1
    return raw[left:right], start + left, start + right


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _l2_norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in a))


def _safe_cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return _dot(a, b) / (na * nb)


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    last = 0
    for m in PARAGRAPH_BOUNDARY_RE.finditer(text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))
    return spans


def _sentence_spans(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    window = text[start:end]
    last = 0
    for m in SENTENCE_BOUNDARY_RE.finditer(window):
        s = last
        e = m.end()
        if e > s:
            out.append((start + s, start + e))
        last = e
    if last < len(window):
        out.append((start + last, start + len(window)))
    return out


def _split_semantic_units(text: str) -> List[Dict]:
    units: List[Dict] = []
    for p_start, p_end in _paragraph_spans(text):
        if p_end <= p_start:
            continue
        s_spans = _sentence_spans(text, p_start, p_end)
        if not s_spans:
            s_spans = [(p_start, p_end)]

        for s_start, s_end in s_spans:
            raw_piece = text[s_start:s_end]
            piece, piece_start, piece_end = _trim_with_span(raw_piece, s_start)
            if not piece:
                continue
            units.append(
                {
                    "text": piece,
                    "start": piece_start,
                    "end": piece_end,
                }
            )
    return units


def _chunk_text_token(
    *,
    clean: str,
    source: str,
    chunk_size: int,
    overlap: int,
    spans: List[Tuple[int, int]],
    total_tokens: int,
) -> List[Dict]:
    chunks: List[Dict] = []
    token_start = 0
    idx = 0

    while token_start < total_tokens:
        hard_token_end = min(token_start + chunk_size, total_tokens)
        boundary_type = "hard"

        if hard_token_end == total_tokens:
            token_end = hard_token_end
        else:
            char_start = spans[token_start][0]
            hard_char_end = spans[hard_token_end - 1][1]
            split_char_end, boundary_type = _find_split_pos(clean, char_start, hard_char_end)
            token_end = _char_end_to_token_end(spans, token_start, split_char_end, hard_token_end)

        if token_end <= token_start:
            token_end = hard_token_end
        if token_end <= token_start:
            token_end = min(token_start + 1, total_tokens)

        char_start = spans[token_start][0]
        char_end = spans[token_end - 1][1]
        raw_piece = clean[char_start:char_end]
        piece, piece_start, piece_end = _trim_with_span(raw_piece, char_start)

        if piece:
            chunks.append(
                {
                    "chunk_id": idx,
                    "source": source,
                    "text": piece,
                    "start": piece_start,
                    "end": piece_end,
                    "token_start": token_start,
                    "token_end": token_end,
                    "boundary": boundary_type,
                    "chunk_strategy": "token",
                }
            )
            idx += 1

        if token_end >= total_tokens:
            break

        token_start = max(token_end - overlap, token_start + 1)

    return chunks


def _chunk_text_semantic(
    *,
    clean: str,
    source: str,
    chunk_size: int,
    overlap: int,
    spans: List[Tuple[int, int]],
    total_tokens: int,
    semantic_threshold: float,
    semantic_min_sentences: int,
    semantic_embed_fn: Optional[EmbedFn],
) -> List[Dict]:
    units = _split_semantic_units(clean)
    if len(units) <= 1:
        return _chunk_text_token(
            clean=clean,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
            spans=spans,
            total_tokens=total_tokens,
        )

    token_hint = 0
    for u in units:
        t_start = _char_start_to_token_start(spans, int(u["start"]), token_hint)
        t_end = _char_end_to_token_end(spans, t_start, int(u["end"]), total_tokens)
        if t_end <= t_start:
            t_end = min(t_start + 1, total_tokens)
        u["token_start"] = t_start
        u["token_end"] = t_end
        u["token_len"] = max(1, t_end - t_start)
        token_hint = t_start

    if semantic_embed_fn is None:
        # Semantic strategy requested but no embedding function provided.
        return _chunk_text_token(
            clean=clean,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
            spans=spans,
            total_tokens=total_tokens,
        )

    vectors = semantic_embed_fn([str(u["text"]) for u in units])
    if len(vectors) != len(units):
        return _chunk_text_token(
            clean=clean,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
            spans=spans,
            total_tokens=total_tokens,
        )

    groups: List[Tuple[int, int, str]] = []
    g_start = 0
    g_tokens = int(units[0]["token_len"])
    g_sent_count = 1
    break_reason = "semantic"

    for i in range(1, len(units)):
        sim = _safe_cosine(vectors[i - 1], vectors[i])
        next_tokens = int(units[i]["token_len"])

        should_break = False
        if g_tokens + next_tokens > chunk_size:
            should_break = True
            break_reason = "token_limit"
        elif g_sent_count >= semantic_min_sentences and sim < semantic_threshold:
            should_break = True
            break_reason = "semantic"

        if should_break:
            groups.append((g_start, i - 1, break_reason))
            g_start = i
            g_tokens = next_tokens
            g_sent_count = 1
        else:
            g_tokens += next_tokens
            g_sent_count += 1

    groups.append((g_start, len(units) - 1, "semantic"))

    chunks: List[Dict] = []
    for idx, (s_i, e_i, reason) in enumerate(groups):
        u0 = units[s_i]
        u1 = units[e_i]
        c_start = int(u0["start"])
        c_end = int(u1["end"])
        piece = clean[c_start:c_end].strip()
        if not piece:
            continue

        t_start = int(u0["token_start"])
        t_end = int(u1["token_end"])
        chunks.append(
            {
                "chunk_id": idx,
                "source": source,
                "text": piece,
                "start": c_start,
                "end": c_end,
                "token_start": t_start,
                "token_end": t_end,
                "boundary": reason,
                "chunk_strategy": "semantic",
            }
        )

    return chunks


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 700,
    overlap: int = 120,
    encoding_name: str = "cl100k_base",
    chunk_strategy: str = "token",
    semantic_threshold: float = 0.78,
    semantic_min_sentences: int = 2,
    semantic_embed_fn: Optional[EmbedFn] = None,
) -> List[Dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not (0.0 <= semantic_threshold <= 1.0):
        raise ValueError("semantic_threshold must be in [0, 1]")
    if semantic_min_sentences <= 0:
        raise ValueError("semantic_min_sentences must be > 0")

    clean = (text or "").strip()
    if not clean:
        return []

    # Token budget is more stable than char budget across languages/types.
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = encoding.encode(clean)
    if not token_ids:
        return []

    spans = _token_char_spans(clean, token_ids, encoding)
    total_tokens = len(token_ids)

    strategy = (chunk_strategy or "token").strip().lower()
    if strategy == "semantic":
        return _chunk_text_semantic(
            clean=clean,
            source=source,
            chunk_size=chunk_size,
            overlap=overlap,
            spans=spans,
            total_tokens=total_tokens,
            semantic_threshold=semantic_threshold,
            semantic_min_sentences=semantic_min_sentences,
            semantic_embed_fn=semantic_embed_fn,
        )

    return _chunk_text_token(
        clean=clean,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
        spans=spans,
        total_tokens=total_tokens,
    )
