from __future__ import annotations

from typing import Any, Dict, List

from app.agent.tools.base import Tool, ToolResult

_TYPE_MAP = {"str": str, "int": int, "number": (int, float)}


class ToolRegistry:
    """持有 name -> Tool；prompt 渲染工具清单；分发前做参数校验（不信任模型输出）。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> List[str]:
        return list(self._tools)

    def render_tools(self) -> str:
        blocks: List[str] = []
        for tool in self._tools.values():
            arg_lines: List[str] = []
            for arg, spec in tool.args_schema.items():
                req = "required" if spec.get("required") else "optional"
                arg_lines.append(f"    - {arg} ({spec.get('type', 'str')}, {req}): {spec.get('desc', '')}")
            args_text = "\n".join(arg_lines) if arg_lines else "    (no args)"
            blocks.append(f"- {tool.name}: {tool.description}\n  args:\n{args_text}")
        return "\n".join(blocks)

    def dispatch(self, name: str, args: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, content="", error=f"unknown tool: {name}; available: {', '.join(self.names())}")
        if not isinstance(args, dict):
            return ToolResult(ok=False, content="", error="bad args: args must be a JSON object")
        problem = self._validate(tool, args)
        if problem:
            return ToolResult(ok=False, content="", error=f"bad args: {problem}")
        try:
            return tool.run(args)
        except Exception as e:  # 工具自身兜底失守时，注册表兜住，绝不中断循环
            return ToolResult(ok=False, content="", error=f"tool crashed: {e}")

    @staticmethod
    def _validate(tool: Tool, args: Dict[str, Any]) -> str:
        for arg, spec in tool.args_schema.items():
            if args.get(arg) is None:
                if spec.get("required"):
                    return f"missing required arg '{arg}'"
                continue
            expected = _TYPE_MAP.get(spec.get("type", "str"))
            if expected and not isinstance(args[arg], expected):
                return f"arg '{arg}' must be {spec.get('type')}"
        return ""
