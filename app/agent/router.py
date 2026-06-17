from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from app.memory.extractor import CompleteFn


@dataclass
class RouteDecision:
    use_docs: bool
    use_memory: bool
    write_memory: bool
    rewritten_query: str


_SYSTEM_PROMPT = (
    "You are the router of a personal assistant with two resources: a DOCUMENT knowledge base "
    "and a personal MEMORY of facts about the user. For each user turn decide which to use.\n"
    "- use_docs: the turn needs information from uploaded documents.\n"
    "- use_memory: the answer should be personalized with what we know about the user.\n"
    "- write_memory: the turn reveals a durable new fact about the user worth remembering.\n"
    "Also provide rewritten_query: a standalone search query for retrieval.\n"
    'Return ONLY JSON: {"use_docs": bool, "use_memory": bool, "write_memory": bool, "rewritten_query": str}.'
)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class Router:
    def __init__(self, complete_fn: CompleteFn) -> None:
        self.complete_fn = complete_fn

    def route(self, user_msg: str, history: Optional[str] = None) -> RouteDecision:
        msg = (user_msg or "").strip()
        if not msg:
            # nothing to do: don't retrieve or write on an empty turn
            return RouteDecision(use_docs=False, use_memory=False, write_memory=False, rewritten_query="")

        user_prompt = msg if not history else f"Conversation so far:\n{history}\n\nCurrent turn:\n{msg}"

        try:
            raw = self.complete_fn(_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(_strip_code_fences(raw))
            if not isinstance(parsed, dict):
                raise ValueError("not an object")
            return RouteDecision(
                use_docs=bool(parsed.get("use_docs", True)),
                use_memory=bool(parsed.get("use_memory", True)),
                write_memory=bool(parsed.get("write_memory", True)),
                rewritten_query=str(parsed.get("rewritten_query") or msg).strip() or msg,
            )
        except Exception:
            # conservative fallback: use everything, keep original query
            return RouteDecision(use_docs=True, use_memory=True, write_memory=True, rewritten_query=msg)
