from __future__ import annotations

from typing import Any, Dict, List, Optional


def _doc_key(meta: Dict[str, Any], fallback_idx: int) -> str:
    doc_id = meta.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()
    source = str(meta.get("source", "unknown"))
    chunk_id = meta.get("chunk_id", fallback_idx)
    return f"{source}#{chunk_id}"


def rrf_fuse(
    vec_items: List[Dict[str, Any]],
    bm25_items: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of two ranked result lists.

    Each item contributes ``1 / (rrf_k + rank)`` to its document's fused score;
    a document present in both lists sums both contributions and ranks higher.
    Results are deduplicated by ``source#chunk_id`` and tagged with hybrid_score.
    """
    fused: Dict[str, float] = {}
    best: Dict[str, Dict[str, Any]] = {}

    for items in (vec_items, bm25_items):
        for rank, it in enumerate(items, start=1):
            meta = it.get("metadata") if isinstance(it.get("metadata"), dict) else {}
            key = _doc_key(meta, rank)
            fused[key] = fused.get(key, 0.0) + 1.0 / float(rrf_k + rank)
            best.setdefault(key, it)

    ranked = sorted(fused.keys(), key=lambda k: fused[k], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]

    out: List[Dict[str, Any]] = []
    for key in ranked:
        merged = dict(best[key])
        merged["hybrid_score"] = float(fused[key])
        out.append(merged)
    return out
