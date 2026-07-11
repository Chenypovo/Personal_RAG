from app.agent.registry import ToolRegistry
from app.agent.tools.base import ToolResult
from app.agent.tools.calculator import CalculatorTool


class EchoTool:
    name = "echo"
    description = "回显 query"
    args_schema = {
        "query": {"type": "str", "required": True, "desc": "要回显的内容"},
        "k": {"type": "int", "required": False, "desc": "可选整数"},
    }

    def run(self, args):
        return ToolResult(ok=True, content=f"echo: {args['query']}", data={"query": args["query"]})


class CrashTool:
    name = "crash"
    description = "总是抛异常"
    args_schema = {}

    def run(self, args):
        raise RuntimeError("boom")


def make_registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(CalculatorTool())
    return reg


def test_dispatch_ok():
    r = make_registry().dispatch("echo", {"query": "hi"})
    assert r.ok and r.content == "echo: hi"


def test_unknown_tool():
    r = make_registry().dispatch("nope", {})
    assert not r.ok and "unknown tool: nope" in r.error
    assert "echo" in r.error  # 报错里带可用工具清单，帮模型纠正


def test_missing_required_arg():
    r = make_registry().dispatch("echo", {})
    assert not r.ok and "bad args" in r.error and "query" in r.error


def test_wrong_arg_type():
    r = make_registry().dispatch("echo", {"query": "hi", "k": "three"})
    assert not r.ok and "bad args" in r.error and "k" in r.error


def test_args_not_a_dict():
    r = make_registry().dispatch("echo", "hi")
    assert not r.ok and "bad args" in r.error


def test_tool_exception_becomes_result():
    reg = ToolRegistry()
    reg.register(CrashTool())
    r = reg.dispatch("crash", {})
    assert not r.ok and "boom" in r.error


def test_render_tools_lists_names_and_args():
    text = make_registry().render_tools()
    assert "echo" in text and "calculator" in text
    assert "query" in text and "expression" in text
    assert "required" in text and "optional" in text


def test_names():
    assert set(make_registry().names()) == {"echo", "calculator"}
