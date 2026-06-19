from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List

_HEADING = re.compile(r"^#{1,6}\s")
_LIST = re.compile(r"^(?:[-*+]|\d+\.)\s")


def classify_block_type(text: str) -> str:
    """Classify a chunk's dominant markdown block type.

    Returns one of: heading / code / list / table / paragraph. Structural blocks
    (code/list/table) win when they make up at least half the content lines; a
    chunk that is only heading line(s) is "heading"; everything else is paragraph.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "paragraph"

    counts: Counter = Counter()
    in_code = False
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
            counts["code"] += 1
            continue
        if in_code:
            counts["code"] += 1
        elif _HEADING.match(ln):
            counts["heading"] += 1
        elif _LIST.match(ln):
            counts["list"] += 1
        elif ln.startswith("|") and ln.endswith("|"):
            counts["table"] += 1
        else:
            counts["text"] += 1

    text_lines = counts.get("text", 0)
    structural = {k: counts[k] for k in ("code", "list", "table") if counts[k]}
    if structural:
        dominant = max(structural, key=structural.get)
        if structural[dominant] >= text_lines:
            return dominant

    if text_lines == 0 and counts.get("heading", 0) > 0:
        return "heading"
    return "paragraph"


def assign_block_types(chunks: List[Dict[str, Any]]) -> None:
    """Tag each chunk with its content ``block_type``."""
    for c in chunks:
        c["block_type"] = classify_block_type(str(c.get("text", "")))
