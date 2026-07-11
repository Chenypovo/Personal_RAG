from __future__ import annotations

import ast
from typing import Any, Dict

from app.agent.tools.base import ToolResult

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("unsupported expression")


class CalculatorTool:
    name = "calculator"
    description = "计算一个纯算术表达式（仅 + - * / // % ** 括号与数字），用于对检索到的数值做计算。"
    args_schema = {
        "expression": {"type": "str", "required": True, "desc": "算术表达式，如 '(120-80)/80*100'"},
    }

    def run(self, args: Dict[str, Any]) -> ToolResult:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return ToolResult(ok=False, content="", error="empty expression")
        try:
            value = _eval_node(ast.parse(expr, mode="eval"))
        except ZeroDivisionError:
            return ToolResult(ok=False, content="", error="division by zero")
        except Exception:
            return ToolResult(ok=False, content="", error="unsupported expression")
        return ToolResult(ok=True, content=f"{expr} = {value}", data={"result": value})
