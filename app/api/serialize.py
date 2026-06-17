from __future__ import annotations

from typing import Any, Dict, Optional

from app.agent.agent import AgentResult
from app.memory.models import MemoryFact


def memory_to_dict(fact: MemoryFact) -> Dict[str, Any]:
    return {
        "id": fact.id,
        "fact_object": fact.fact_object,
        "fact_content": fact.fact_content,
    }


def agent_result_to_dict(result: AgentResult) -> Dict[str, Any]:
    route: Optional[Dict[str, Any]] = None
    if result.route is not None:
        route = {
            "use_docs": result.route.use_docs,
            "use_memory": result.route.use_memory,
            "write_memory": result.route.write_memory,
            "rewritten_query": result.route.rewritten_query,
        }

    return {
        "answer": result.answer,
        "sources": list(result.sources or []),
        "recalled_memories": [memory_to_dict(m) for m in result.recalled_memories],
        "memory_ops": [
            {
                "type": op.type,
                "fact_content": op.fact_content,
                "fact_object": op.fact_object,
                "id": op.id,
                "applied": op.applied,
            }
            for op in result.memory_ops
        ],
        "route": route,
    }
