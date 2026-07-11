from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ToolResult:
    ok: bool
    content: str                                        # 喂回模型的自然语言观察
    data: Dict[str, Any] = field(default_factory=dict)  # 编排层用的结构化载荷
    error: Optional[str] = None                         # ok=False 时的错误说明


class Tool(Protocol):
    name: str
    description: str
    # arg name -> {"type": "str"|"int"|"number", "required": bool, "desc": str}
    args_schema: Dict[str, Dict[str, Any]]

    def run(self, args: Dict[str, Any]) -> ToolResult: ...
