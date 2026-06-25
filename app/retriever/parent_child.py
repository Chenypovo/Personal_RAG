from __future__ import annotations

from typing import Any, Dict, List, Optional


def assign_parent_ids(chunks: List[Dict[str, Any]], window: int = 3) -> None:
    """Assign each chunk a ``parent_id`` for parent-child (small-to-big) retrieval.

    Section-aware: chunks under the same markdown heading share a parent (the
    section), so a precise child hit expands back to its whole section. Documents
    without headings (plain text / heading-less PDFs) fall back to a fixed window
    of consecutive chunks — i.e. "回填父级章节 或 相邻上下文".
    """
    w = max(int(window), 1)
    for c in chunks:
        source = str(c.get("source", "unknown"))
        heading = c.get("heading")
        if isinstance(heading, str) and heading.strip():
            c["parent_id"] = f"{source}::{heading.strip()}"
        else:
            cid = int(c.get("chunk_id", 0))
            c["parent_id"] = f"{source}#p{cid // w}"


def _parent_key(meta: Dict[str, Any]) -> str:
    """Parent identifier for a chunk; falls back to the chunk being its own parent."""
    pid = meta.get("parent_id")
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    return f"{meta.get('source', 'unknown')}#c{meta.get('chunk_id')}"


def _cap_around_hit(
    children: List[Dict[str, Any]], hit_meta: Dict[str, Any], max_parent_chunks: int
) -> List[Dict[str, Any]]:
    """Keep at most ``max_parent_chunks`` sibling chunks, centered on the hit child.

    Without this a large section (heading-based parent) has no size bound and can
    blow up the LLM context; the cap keeps the precise hit plus its closest
    neighbours up to the budget.
    """
    if max_parent_chunks <= 0 or len(children) <= max_parent_chunks:
        return children
    ids = [c.get("chunk_id") for c in children]
    try:
        idx = ids.index(hit_meta.get("chunk_id"))
    except ValueError:
        idx = len(children) // 2
    half = max_parent_chunks // 2
    start = max(0, idx - half)
    end = min(len(children), start + max_parent_chunks)
    start = max(0, end - max_parent_chunks)
    return children[start:end]


def expand_to_parents(
    hits: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    text_joiner: str = "\n\n",
    max_parent_chunks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Parent-child (small-to-big) retrieval: expand each retrieved child chunk to
    its parent block (sibling chunks sharing ``parent_id``), concatenated in chunk
    order. ``max_parent_chunks`` caps the block centered on the hit (so large
    heading sections don't blow up the context). Parents are deduplicated and the
    original hit ranking order and score/distance fields are preserved.
    """
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for c in all_chunks:
        if isinstance(c, dict):
            by_parent.setdefault(_parent_key(c), []).append(c)
    for key in by_parent:
        by_parent[key].sort(key=lambda c: c.get("chunk_id", 0))

    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for h in hits:
        meta = h.get("metadata") if isinstance(h.get("metadata"), dict) else {}
        pkey = _parent_key(meta)
        if pkey in seen:
            continue  # dedup: this parent was already emitted by a higher-ranked child
        seen.add(pkey)

        children = by_parent.get(pkey, [meta])
        if max_parent_chunks is not None:
            children = _cap_around_hit(children, meta, max_parent_chunks)
        text = text_joiner.join(
            str(c.get("text", "")).strip() for c in children if str(c.get("text", "")).strip()
        )

        parent_meta = dict(meta)  # keep the hit child's metadata (source, citation, etc.)
        parent_meta["text"] = text
        parent_meta["parent_id"] = pkey
        parent_meta["child_chunk_ids"] = [c.get("chunk_id") for c in children]

        new_item = dict(h)
        new_item["metadata"] = parent_meta
        out.append(new_item)

    return out
