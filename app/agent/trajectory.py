from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryStep:
    step_index: int
    thought: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    ok: bool = True
    error: Optional[str] = None
    latency_ms: Optional[float] = None


def format_trajectory(steps: List[TrajectoryStep]) -> str:
    """人读轨迹，命令行/日志打印用。"""
    lines: List[str] = []
    for s in steps:
        status = "ok" if s.ok else f"ERROR: {s.error}"
        if s.tool:
            head = f"[{s.step_index}] {s.tool}({json.dumps(s.args, ensure_ascii=False)}) -> {status}"
        else:
            head = f"[{s.step_index}] {status}"
        lines.append(head)
        if s.thought:
            lines.append(f"    thought: {s.thought}")
        if s.observation:
            lines.append(f"    obs: {s.observation[:300]}")
    return "\n".join(lines)
