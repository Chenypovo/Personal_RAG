from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

# (char offset of the heading line, level 1-6, title text)
Heading = Tuple[int, int, str]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)


def extract_headings(text: str) -> List[Heading]:
    """Markdown ATX headings (``#`` .. ``######``) with their char offset and level."""
    return [(m.start(), len(m.group(1)), m.group(2).strip()) for m in _HEADING_RE.finditer(text or "")]


def heading_path_for_offset(offset: int, headings: Sequence[Heading]) -> List[str]:
    """The active heading stack at a char offset (top-level first).

    Walks headings up to ``offset`` keeping one entry per level: a heading of
    level L drops any deeper-or-equal headings before becoming the current one,
    so the result is the section breadcrumb a chunk at that offset lives under.
    """
    stack: List[Tuple[int, str]] = []
    for off, level, title in headings:
        if off > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return [title for _, title in stack]


def assign_headings(chunks: List[Dict[str, Any]], text: str) -> None:
    """Annotate each chunk with its heading breadcrumb (heading_path + heading).

    No-op when the document has no headings (e.g. plain text).
    """
    headings = extract_headings(text)
    if not headings:
        return
    for c in chunks:
        path = heading_path_for_offset(int(c.get("start", 0)), headings)
        if path:
            c["heading_path"] = path
            c["heading"] = " > ".join(path)
