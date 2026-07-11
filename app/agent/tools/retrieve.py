from __future__ import annotations

from typing import Any, Dict

from app.agent.agent import RetrieveDocsFn
from app.agent.tools.base import ToolResult


class RetrieveDocsTool:
    name = "retrieve_docs"
    description = (
        "在文档知识库中检索。多跳/复合问题请用不同的子查询多次调用本工具，每次一个独立子问题。"
    )
    args_schema = {
        "query": {"type": "str", "required": True, "desc": "独立成句的检索查询"},
        "k": {"type": "int", "required": False, "desc": "最多保留的 chunk 数（默认用检索器自身 top-k）"},
    }

    def __init__(self, retrieve_docs_fn: RetrieveDocsFn) -> None:
        self.retrieve_docs_fn = retrieve_docs_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult(ok=False, content="", error="empty query")
        try:
            chunks = self.retrieve_docs_fn(query)
        except Exception as e:
            return ToolResult(ok=False, content="", error=f"retrieval failed: {e}")
        k = args.get("k")
        if isinstance(k, int) and k > 0:
            chunks = chunks[:k]
        if not chunks:
            return ToolResult(ok=True, content="no documents matched this query; try a different query",
                              data={"chunks": []})
        lines = []
        for i, c in enumerate(chunks):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            text = str(meta.get("text", "")).replace("\n", " ")[:200]
            lines.append(f"[{meta.get('source', 'unknown')}#{meta.get('chunk_id', i)}] {text}")
        return ToolResult(ok=True, content="retrieved chunks:\n" + "\n".join(lines), data={"chunks": chunks})
