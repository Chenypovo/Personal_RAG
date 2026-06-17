from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from app.agent.router import RouteDecision, Router
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger, MergeOp
from app.memory.models import MemoryFact
from app.memory.recall import format_memory_block, recall_memories
from app.memory.store import MemoryStore

# query -> list of retrieved chunk dicts
RetrieveDocsFn = Callable[[str], List[Dict[str, Any]]]
# (query, chunks, user_memory_block) -> {"answer": str, "sources": list}
GenerateFn = Callable[[str, List[Dict[str, Any]], str], Dict[str, Any]]


@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    recalled_memories: List[MemoryFact] = field(default_factory=list)
    memory_ops: List[MergeOp] = field(default_factory=list)
    route: Optional[RouteDecision] = None


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class MemoryAgent:
    """Orchestrates one assistant turn: route -> recall + retrieve -> generate -> learn.

    All external capabilities (router LLM, doc retrieval, generation, extraction,
    merging) are injected, so the control flow is testable without network access.
    """

    def __init__(
        self,
        router: Router,
        store: MemoryStore,
        extractor: MemoryExtractor,
        merger: MemoryMerger,
        retrieve_docs_fn: RetrieveDocsFn,
        generate_fn: GenerateFn,
        recall_k: int = 5,
        message_time_fn: Callable[[], str] = _today,
    ) -> None:
        self.router = router
        self.store = store
        self.extractor = extractor
        self.merger = merger
        self.retrieve_docs_fn = retrieve_docs_fn
        self.generate_fn = generate_fn
        self.recall_k = recall_k
        self.message_time_fn = message_time_fn

    def chat(self, user_msg: str) -> AgentResult:
        decision = self.router.route(user_msg)

        recalled: List[MemoryFact] = []
        if decision.use_memory:
            recalled = recall_memories(self.store, decision.rewritten_query, top_k=self.recall_k)

        chunks: List[Dict[str, Any]] = []
        if decision.use_docs:
            chunks = self.retrieve_docs_fn(decision.rewritten_query)

        memory_block = format_memory_block(recalled)
        gen = self.generate_fn(decision.rewritten_query, chunks, memory_block)

        ops: List[MergeOp] = []
        if decision.write_memory:
            facts = self.extractor.extract(user_msg, source="chat", message_time=self.message_time_fn())
            ops = self.merger.merge(facts, source="chat")

        return AgentResult(
            answer=gen.get("answer", ""),
            sources=gen.get("sources", []),
            recalled_memories=recalled,
            memory_ops=ops,
            route=decision,
        )
