from __future__ import annotations

from typing import List

from app.memory.models import MemoryFact
from app.memory.store import MemoryStore


def recall_memories(
    store: MemoryStore, query: str, top_k: int = 5, min_score: float = 0.0
) -> List[MemoryFact]:
    """Return active memory facts relevant to the query, most relevant first."""
    return [fact for fact, _score in store.search(query, top_k=top_k, min_score=min_score)]


def format_memory_block(facts: List[MemoryFact]) -> str:
    """Render recalled memories as a context block for the generator.

    Returns an empty string when there is nothing to inject so callers can
    cheaply decide whether to include the block at all.
    """
    if not facts:
        return ""
    lines = []
    for f in facts:
        prefix = f"({f.fact_object}) " if f.fact_object else ""
        lines.append(f"- {prefix}{f.fact_content}")
    return "\n".join(lines)
