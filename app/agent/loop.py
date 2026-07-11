from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


@dataclass
class Decision:
    thought: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[List[str]] = None
    final_answer: bool = False


def parse_decision(raw: str) -> Optional[Decision]:
    """防御式解析每步决策；解析不出可执行动作一律返回 None（观察 format error）。

    字段优先级 final_answer > plan > tool；plan 可与 tool 同时出现。
    """
    try:
        parsed = json.loads(_strip_code_fences(raw or ""))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    thought = str(parsed.get("thought", "")).strip()
    raw_plan = parsed.get("plan")
    plan = [str(p).strip() for p in raw_plan if str(p).strip()] if isinstance(raw_plan, list) else None

    if parsed.get("final_answer"):
        return Decision(thought=thought, plan=plan, final_answer=True)

    tool = parsed.get("tool")
    if isinstance(tool, str) and tool.strip():
        args = parsed.get("args")
        return Decision(thought=thought, tool=tool.strip(),
                        args=args if isinstance(args, dict) else {}, plan=plan)

    if plan is not None:
        return Decision(thought=thought, plan=plan)
    return None
