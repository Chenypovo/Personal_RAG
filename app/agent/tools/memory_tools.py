from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from app.agent.tools.base import ToolResult
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
from app.memory.recall import format_memory_block, recall_memories
from app.memory.store import MemoryStore


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


class ReadMemoryTool:
    name = "read_memory"
    description = "召回与查询相关的、已知的关于用户的事实（长期记忆）。"
    args_schema = {
        "query": {"type": "str", "required": True, "desc": "要召回什么方面的用户事实"},
        "k": {"type": "int", "required": False, "desc": "最多召回条数（默认 5）"},
    }

    def __init__(self, store: MemoryStore, default_k: int = 5) -> None:
        self.store = store
        self.default_k = default_k

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, content="", error="empty query")
        k = args.get("k")
        top_k = k if isinstance(k, int) and k > 0 else self.default_k
        try:
            facts = recall_memories(self.store, query, top_k=top_k)
        except Exception as e:
            return ToolResult(ok=False, content="", error=f"memory recall failed: {e}")
        if not facts:
            return ToolResult(ok=True, content="no relevant memories found", data={"memories": []})
        return ToolResult(ok=True, content="recalled memories:\n" + format_memory_block(facts),
                          data={"memories": facts})


class WriteMemoryTool:
    name = "write_memory"
    description = "从文本中提取用户的持久事实并归并进长期记忆（非破坏归并）。text 缺省为本轮用户消息。"
    args_schema = {
        "text": {"type": "str", "required": False, "desc": "要学习的文本（缺省: 本轮用户消息）"},
    }

    def __init__(
        self,
        extractor: MemoryExtractor,
        merger: MemoryMerger,
        message_time_fn: Callable[[], str] = _today,
    ) -> None:
        self.extractor = extractor
        self.merger = merger
        self.message_time_fn = message_time_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "").strip()
        if not text:
            return ToolResult(ok=False, content="", error="empty text: nothing to learn")
        try:
            facts = self.extractor.extract(text, source="chat", message_time=self.message_time_fn())
            if not facts:
                return ToolResult(ok=True, content="no durable facts found in the text", data={"ops": []})
            ops = self.merger.merge(facts, source="chat")
        except Exception as e:
            return ToolResult(ok=False, content="", error=f"memory write failed: {e}")
        counts = Counter(op.type for op in ops if op.applied)
        content = (
            f"memory merged: {counts.get('add', 0)} add, "
            f"{counts.get('update', 0)} update, {counts.get('delete', 0)} delete"
        )
        return ToolResult(ok=True, content=content, data={"ops": ops})
