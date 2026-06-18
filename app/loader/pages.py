from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# (char offset where the page starts in the joined text, 1-based page number)
PageOffsets = Sequence[Tuple[int, int]]


def page_for_offset(offset: int, page_offsets: PageOffsets) -> Optional[int]:
    """Page number containing a character offset.

    ``page_offsets`` is ascending by start offset. Returns the page whose start
    is the greatest one not after ``offset`` (offsets before the first page clamp
    to the first page). Returns None when there is no page information.
    """
    if not page_offsets:
        return None
    page = page_offsets[0][1]
    for start, num in page_offsets:
        if offset >= start:
            page = num
        else:
            break
    return page


def assign_pages(chunks: List[Dict[str, Any]], page_offsets: PageOffsets) -> None:
    """Annotate each chunk with the source page(s) it spans (page_start/page_end).

    No-op when no page information is available (e.g. non-PDF sources).
    """
    if not page_offsets:
        return
    for c in chunks:
        start = int(c.get("start", 0))
        end = int(c.get("end", start))
        ps = page_for_offset(start, page_offsets)
        pe = page_for_offset(end, page_offsets)
        if ps is not None:
            c["page_start"] = ps
        if pe is not None:
            c["page_end"] = pe
