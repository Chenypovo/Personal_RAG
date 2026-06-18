from __future__ import annotations

from typing import Any, Dict, List


def _parent_key(meta: Dict[str, Any]) -> str:
    """Parent identifier for a chunk; falls back to the chunk being its own parent."""
    pid = meta.get("parent_id")
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    return f"{meta.get('source', 'unknown')}#c{meta.get('chunk_id')}"


def expand_to_parents(
    hits: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
    text_joiner: str = "\n\n",
) -> List[Dict[str, Any]]:
    """Parent-child (small-to-big) retrieval: expand each retrieved child chunk to
    its full parent block (all sibling chunks sharing ``parent_id``), concatenated
    in chunk order. Parents are deduplicated, so a parent appears once even when
    several of its children are retrieved, and the original hit ranking order is
    preserved. The hit's score/distance fields are carried through unchanged.
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
