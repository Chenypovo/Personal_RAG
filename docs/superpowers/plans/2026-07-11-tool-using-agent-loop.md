# 工具调用 Agent 循环（Tool-Using Agent Loop）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把固定三开关路由升级为真正的工具调用 agent：自研控制循环 + 工具注册表 + 失败重试/预算/死循环护栏 + 轨迹日志，全程 prompt-based JSON、依赖注入、可离线单测，并用离线评测对比单次基线。

**Architecture:** 新增 `app/agent/tools/`（4 个真工具）、`app/agent/registry.py`（校验+分发）、`app/agent/trajectory.py`、`app/agent/loop.py`（ReAct 式单步决策 + 轻量 plan + 反思重试；`final_answer` 由循环特判，终局合成复用现成 `generate_fn`）。`Router`/`MemoryAgent` 原样保留作单次基线。评测逻辑放 `app/eval/agent_eval.py`，CLI 在 `scripts/eval_agent.py`。

**Tech Stack:** Python 3.9+，`@dataclass` + `typing.Protocol`，`ast` 安全求值，pytest（无网络、假 LLM/假工具），复用 `app/eval/metrics.py` 的 recall 逻辑。

**Spec:** `docs/superpowers/specs/2026-07-11-tool-using-agent-loop-design.md`

**关键既有约定（实现时必须对齐）：**
- `CompleteFn = Callable[[str, str], str]`（`app/memory/extractor.py:8`），签名 `(system_prompt, user_prompt) -> raw`。
- `GenerateFn = Callable[[str, List[Dict], str], Dict]`（`app/agent/agent.py:17`），`(query, chunks, memory_block) -> {"answer", "sources"}`。
- chunk 形如 `{"metadata": {"source", "chunk_id", "text"}, ...}`。
- 防御式解析仿 `app/agent/router.py:29` 的 `_strip_code_fences` + try/except + 缺字段默认。
- 记忆：`recall_memories` / `format_memory_block`（`app/memory/recall.py`）、`MemoryExtractor.extract`、`MemoryMerger.merge`。

---

### Task 0: 建分支

- [ ] **Step 1: 从 master 建特性分支**

```bash
cd /Users/starrystark/项目/Personal_RAG && git checkout -b feature/tool-agent-loop
```

- [ ] **Step 2: 基线全绿确认**

Run: `python -m pytest -q`
Expected: 全部通过（记录当前通过数，后续不许变红）。

---

### Task 1: `Tool`/`ToolResult` 基座 + `calculator`（AST 安全求值）

**Files:**
- Create: `app/agent/tools/__init__.py`（空文件）
- Create: `app/agent/tools/base.py`
- Create: `app/agent/tools/calculator.py`
- Test: `tests/test_calculator.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_calculator.py
from app.agent.tools.calculator import CalculatorTool


def run(expr):
    return CalculatorTool().run({"expression": expr})


def test_basic_arithmetic():
    assert run("1 + 2 * 3").data["result"] == 7
    assert run("(120 - 80) / 80 * 100").data["result"] == 50.0
    assert run("2 ** 10").data["result"] == 1024
    assert run("7 // 2").data["result"] == 3
    assert run("7 % 3").data["result"] == 1
    assert run("-5 + +2").data["result"] == -3


def test_result_content_readable():
    r = run("1+1")
    assert r.ok and "2" in r.content


def test_division_by_zero():
    r = run("1 / 0")
    assert not r.ok and r.error == "division by zero"


def test_rejects_non_arithmetic_nodes():
    for bad in ["__import__('os')", "abs(1)", "a + 1", "[1,2][0]", "(1).real", "'x' * 3"]:
        r = run(bad)
        assert not r.ok
        assert r.error == "unsupported expression"


def test_empty_expression():
    r = run("")
    assert not r.ok and r.error == "empty expression"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_calculator.py -q`
Expected: FAIL（ModuleNotFoundError: app.agent.tools）

- [ ] **Step 3: 实现 base + calculator**

```python
# app/agent/tools/base.py
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
```

```python
# app/agent/tools/calculator.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_calculator.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/tools/ tests/test_calculator.py
git commit -m "feat(agent): Tool/ToolResult base + AST-safe calculator tool"
```

---

### Task 2: `ToolRegistry`（校验 + 分发 + prompt 渲染）

**Files:**
- Create: `app/agent/registry.py`
- Test: `tests/test_tool_registry.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_registry.py
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_tool_registry.py -q`
Expected: FAIL（ModuleNotFoundError: app.agent.registry）

- [ ] **Step 3: 实现注册表**

```python
# app/agent/registry.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_tool_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/registry.py tests/test_tool_registry.py
git commit -m "feat(agent): ToolRegistry with arg validation, safe dispatch, prompt rendering"
```

---

### Task 3: 包装现成能力的三个工具（retrieve_docs / read_memory / write_memory）

**Files:**
- Create: `app/agent/tools/retrieve.py`
- Create: `app/agent/tools/memory_tools.py`
- Test: `tests/test_agent_tools.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_tools.py
from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from app.agent.tools.retrieve import RetrieveDocsTool
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
from app.memory.models import MemoryFact
from app.memory.store import MemoryStore
from app.memory.vector_index import NumpyVectorIndex

VOCAB = ["guitar", "gym", "python", "faiss", "project"]


def fake_embed(text: str):
    t = (text or "").lower()
    return [1.0 if w in t else 0.0 for w in VOCAB]


def const(canned: str):
    def _complete(system_prompt: str, user_prompt: str) -> str:
        return canned
    return _complete


def make_store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "mem.db"),
        vector_index=NumpyVectorIndex(dim=len(VOCAB)),
        embed_fn=fake_embed,
    )


# ---- retrieve_docs ----

CHUNKS = [
    {"metadata": {"source": "a.md", "chunk_id": 1, "text": "FAISS supports IVF indexes"}},
    {"metadata": {"source": "b.md", "chunk_id": 5, "text": "BM25 is sparse retrieval"}},
]


def test_retrieve_docs_returns_chunks_and_readable_content():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "faiss"})
    assert r.ok
    assert r.data["chunks"] == CHUNKS
    assert "a.md#1" in r.content and "IVF" in r.content


def test_retrieve_docs_k_caps_results():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "faiss", "k": 1})
    assert len(r.data["chunks"]) == 1


def test_retrieve_docs_empty_query_rejected():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "  "})
    assert not r.ok and "empty query" in r.error


def test_retrieve_docs_no_hits_is_ok_with_hint():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: [])
    r = tool.run({"query": "zzz"})
    assert r.ok and r.data["chunks"] == [] and "no documents" in r.content


def test_retrieve_docs_fn_exception_wrapped():
    def boom(q):
        raise RuntimeError("index missing")
    r = RetrieveDocsTool(retrieve_docs_fn=boom).run({"query": "x"})
    assert not r.ok and "index missing" in r.error


# ---- read_memory ----

def test_read_memory_recalls_relevant_facts(tmp_path):
    store = make_store(tmp_path)
    store.add(MemoryFact(fact_content="has been learning guitar since 2026"))
    r = ReadMemoryTool(store=store).run({"query": "guitar"})
    assert r.ok
    assert any("guitar" in f.fact_content for f in r.data["memories"])
    assert "guitar" in r.content


def test_read_memory_empty_store(tmp_path):
    r = ReadMemoryTool(store=make_store(tmp_path)).run({"query": "guitar"})
    assert r.ok and r.data["memories"] == [] and "no relevant memories" in r.content


# ---- write_memory ----

def test_write_memory_extracts_and_merges(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const(
            '{"facts": [{"fact_object": "project", "fact_content": "is building a faiss project"}]}'
        )),
        merger=MemoryMerger(store=store, complete_fn=const(
            '{"operations": [{"type": "add", "fact_object": "project", "fact_content": "is building a faiss project"}]}'
        )),
    )
    r = tool.run({"text": "我在做一个 faiss 项目"})
    assert r.ok
    assert [op.type for op in r.data["ops"]] == ["add"]
    assert "1 add" in r.content
    assert any("faiss project" in f.fact_content for f in store.list_active())


def test_write_memory_nothing_durable(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const('{"facts": []}')),
        merger=MemoryMerger(store=store, complete_fn=const('{"operations": []}')),
    )
    r = tool.run({"text": "哈哈好的"})
    assert r.ok and r.data["ops"] == [] and store.list_active() == []


def test_write_memory_empty_text_rejected(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const("[]")),
        merger=MemoryMerger(store=store, complete_fn=const('{"operations": []}')),
    )
    r = tool.run({})
    assert not r.ok and "empty text" in r.error
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_agent_tools.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现两个模块**

```python
# app/agent/tools/retrieve.py
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
```

```python
# app/agent/tools/memory_tools.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_agent_tools.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/tools/retrieve.py app/agent/tools/memory_tools.py tests/test_agent_tools.py
git commit -m "feat(agent): retrieve_docs/read_memory/write_memory tools wrapping existing capabilities"
```

---

### Task 4: 轨迹 `TrajectoryStep` + 决策解析 `parse_decision`

**Files:**
- Create: `app/agent/trajectory.py`
- Create: `app/agent/loop.py`（本任务只放 `Decision` / `parse_decision`，循环下个任务加）
- Test: `tests/test_tool_agent.py`（先只放解析与轨迹的测试）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_agent.py
from app.agent.loop import parse_decision
from app.agent.trajectory import TrajectoryStep, format_trajectory


# ---- parse_decision ----

def test_parse_tool_call():
    d = parse_decision('{"thought": "先查", "tool": "retrieve_docs", "args": {"query": "IVF"}}')
    assert d.tool == "retrieve_docs" and d.args == {"query": "IVF"}
    assert d.thought == "先查" and not d.final_answer and d.plan is None


def test_parse_with_code_fences():
    d = parse_decision('```json\n{"thought": "t", "final_answer": true}\n```')
    assert d.final_answer


def test_parse_plan_only():
    d = parse_decision('{"thought": "拆解", "plan": ["查A", "查B"]}')
    assert d.plan == ["查A", "查B"] and d.tool is None and not d.final_answer


def test_parse_plan_and_tool_together():
    d = parse_decision('{"plan": ["查A", "查B"], "tool": "retrieve_docs", "args": {"query": "A"}}')
    assert d.plan == ["查A", "查B"] and d.tool == "retrieve_docs"


def test_final_answer_beats_tool():
    d = parse_decision('{"final_answer": true, "tool": "retrieve_docs", "args": {}}')
    assert d.final_answer and d.tool is None


def test_parse_invalid_json_returns_none():
    assert parse_decision("我觉得应该先检索") is None


def test_parse_non_dict_returns_none():
    assert parse_decision('["not", "a", "dict"]') is None


def test_parse_dict_without_action_returns_none():
    assert parse_decision('{"thought": "只想不做"}') is None


def test_parse_missing_args_defaults_empty():
    d = parse_decision('{"tool": "calculator"}')
    assert d.tool == "calculator" and d.args == {}


# ---- trajectory ----

def test_format_trajectory_prints_steps():
    steps = [
        TrajectoryStep(step_index=0, thought="查一下", tool="retrieve_docs",
                       args={"query": "IVF"}, observation="retrieved chunks:...", ok=True),
        TrajectoryStep(step_index=1, observation="format error", ok=False, error="invalid decision JSON"),
    ]
    text = format_trajectory(steps)
    assert "retrieve_docs" in text and "IVF" in text
    assert "invalid decision JSON" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_tool_agent.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现 trajectory + 解析**

```python
# app/agent/trajectory.py
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
```

```python
# app/agent/loop.py
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_tool_agent.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/trajectory.py app/agent/loop.py tests/test_tool_agent.py
git commit -m "feat(agent): trajectory records + defensive per-step decision parsing"
```

---

### Task 5: `ToolAgent` 控制循环（核心：护栏 + 反思 + 终局合成）

**Files:**
- Modify: `app/agent/loop.py`（追加系统 prompt、`AgentResult`、`ToolAgent`）
- Test: `tests/test_tool_agent.py`（追加循环测试）

- [ ] **Step 1: 追加失败测试（追加到 `tests/test_tool_agent.py`）**

```python
# ==== 追加到 tests/test_tool_agent.py ====
from app.agent.loop import ToolAgent
from app.agent.registry import ToolRegistry
from app.agent.tools.base import ToolResult
from app.memory.models import MemoryFact


def scripted(responses):
    """按序吐出脚本化响应的假 complete_fn，并记录收到的 user_prompt。"""
    it = iter(responses)
    prompts = []

    def _complete(system_prompt, user_prompt):
        prompts.append(user_prompt)
        return next(it)

    _complete.prompts = prompts
    return _complete


class FakeSearch:
    """内存态小语料检索工具，签名与 RetrieveDocsTool 一致。"""
    name = "retrieve_docs"
    description = "search tiny corpus"
    args_schema = {"query": {"type": "str", "required": True, "desc": "q"}}

    CORPUS = [
        {"metadata": {"source": "a.md", "chunk_id": 1, "text": "IVF is an inverted-file index in faiss"}},
        {"metadata": {"source": "b.md", "chunk_id": 5, "text": "BM25 is sparse lexical retrieval"}},
    ]

    def run(self, args):
        q = str(args.get("query", "")).lower()
        hits = [c for c in self.CORPUS if any(w in c["metadata"]["text"].lower() for w in q.split())]
        return ToolResult(ok=True, content=f"{len(hits)} hits", data={"chunks": hits})


class FakeMemoryRead:
    name = "read_memory"
    description = "fake recall"
    args_schema = {"query": {"type": "str", "required": True, "desc": "q"}}

    def run(self, args):
        facts = [MemoryFact(fact_content="prefers concise answers")]
        return ToolResult(ok=True, content="recalled 1", data={"memories": facts})


class CaptureArgsTool:
    name = "write_memory"
    description = "capture args"
    args_schema = {"text": {"type": "str", "required": False, "desc": "t"}}

    def __init__(self):
        self.seen = []

    def run(self, args):
        self.seen.append(dict(args))
        return ToolResult(ok=True, content="memory merged: 1 add, 0 update, 0 delete", data={"ops": []})


def build_agent(responses, tools=(), captured=None, **kwargs):
    captured = captured if captured is not None else {}
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)

    def generate(query, chunks, memory_block):
        captured["query"] = query
        captured["chunks"] = chunks
        captured["memory_block"] = memory_block
        return {"answer": "final synthesized answer", "sources": [{"citation": "c"}]}

    complete = scripted(responses)
    agent = ToolAgent(complete_fn=complete, registry=reg, generate_fn=generate, **kwargs)
    return agent, complete, captured


def chunk_keys(chunks):
    return {f"{c['metadata']['source']}#{c['metadata']['chunk_id']}" for c in chunks}


def test_multihop_two_retrievals_then_final(tmp_path=None):
    responses = [
        '{"thought": "拆两步", "plan": ["查IVF", "查BM25", "回答"], "tool": "retrieve_docs", "args": {"query": "IVF"}}',
        '{"thought": "再查第二跳", "tool": "retrieve_docs", "args": {"query": "BM25"}}',
        '{"thought": "证据够了", "final_answer": true}',
    ]
    agent, complete, captured = build_agent(responses, tools=[FakeSearch()])
    result = agent.chat("对比 IVF 和 BM25")

    assert result.stop_reason == "final_answer"
    assert result.answer == "final synthesized answer"
    assert chunk_keys(captured["chunks"]) == {"a.md#1", "b.md#5"}   # 两跳证据都进终局合成
    assert result.plan == ["查IVF", "查BM25", "回答"]
    tool_steps = [s for s in result.trajectory if s.tool]
    assert [s.args["query"] for s in tool_steps] == ["IVF", "BM25"]
    # 第二步的 prompt 能看到第一步的观察（观察进上下文）
    assert "hits" in complete.prompts[1]


def test_read_memory_feeds_memory_block():
    responses = [
        '{"tool": "read_memory", "args": {"query": "偏好"}}',
        '{"final_answer": true}',
    ]
    agent, _, captured = build_agent(responses, tools=[FakeMemoryRead()])
    result = agent.chat("按我的偏好回答")
    assert "concise" in captured["memory_block"]
    assert any("concise" in f.fact_content for f in result.recalled_memories)


def test_write_memory_defaults_text_to_user_msg():
    tool = CaptureArgsTool()
    responses = [
        '{"tool": "write_memory", "args": {}}',
        '{"final_answer": true}',
    ]
    agent, _, _ = build_agent(responses, tools=[tool])
    agent.chat("我在学吉他")
    assert tool.seen[0]["text"] == "我在学吉他"


def test_unknown_tool_error_fed_back_and_recovered():
    responses = [
        '{"tool": "web_search", "args": {"query": "x"}}',
        '{"final_answer": true}',
    ]
    agent, complete, _ = build_agent(responses, tools=[FakeSearch()])
    result = agent.chat("查点东西")
    assert result.stop_reason == "final_answer"
    bad = [s for s in result.trajectory if not s.ok]
    assert bad and "unknown tool" in bad[0].error
    assert "unknown tool" in complete.prompts[1]      # 错误作为观察喂回下一轮


def test_parse_abort_after_retries():
    responses = ["not json", "still not json", "nope"]
    agent, _, captured = build_agent(responses, tools=[FakeSearch()], max_parse_retries=2)
    result = agent.chat("你好")
    assert result.stop_reason == "parse_abort"
    assert result.answer == "final synthesized answer"   # 兜底仍走合成，不崩
    assert captured["chunks"] == []


def test_step_budget_fallback():
    responses = [
        '{"tool": "retrieve_docs", "args": {"query": "IVF"}}',
        '{"tool": "retrieve_docs", "args": {"query": "BM25"}}',
    ]
    agent, _, captured = build_agent(responses, tools=[FakeSearch()], max_steps=2)
    result = agent.chat("查资料")
    assert result.stop_reason == "budget"
    assert chunk_keys(captured["chunks"]) == {"a.md#1", "b.md#5"}   # 已收集证据仍用于合成


def test_tool_call_budget_independent_of_steps():
    responses = ['{"tool": "retrieve_docs", "args": {"query": "IVF"}}']
    agent, _, _ = build_agent(responses, tools=[FakeSearch()], max_steps=10, max_tool_calls=1)
    result = agent.chat("查资料")
    assert result.stop_reason == "budget"


def test_duplicate_call_hint_then_loop_abort():
    same = '{"tool": "retrieve_docs", "args": {"query": "IVF"}}'
    agent, complete, _ = build_agent([same, same, same, same], tools=[FakeSearch()],
                                     max_steps=10, max_duplicate_calls=2)
    result = agent.chat("查资料")
    assert result.stop_reason == "loop_abort"
    dups = [s for s in result.trajectory if s.error == "duplicate call"]
    assert len(dups) == 3
    assert "duplicate call" in complete.prompts[2]     # 提示注入了上下文


def test_plan_only_step_updates_plan():
    responses = [
        '{"thought": "先规划", "plan": ["查A", "回答"]}',
        '{"final_answer": true}',
    ]
    agent, _, _ = build_agent(responses, tools=[FakeSearch()])
    result = agent.chat("复杂问题")
    assert result.plan == ["查A", "回答"]
    assert result.stop_reason == "final_answer"


def test_empty_message_short_circuits():
    agent, complete, _ = build_agent([], tools=[FakeSearch()])
    result = agent.chat("   ")
    assert result.answer == "" and complete.prompts == []


def test_history_carried_and_bounded():
    def one_turn():
        return ['{"final_answer": true}']
    agent, complete, _ = build_agent(one_turn() * 5, tools=[FakeSearch()], history_max_turns=2)
    for i in range(5):
        agent.chat(f"msg{i}")
    assert len(agent.history) <= 2
    assert "msg3" in complete.prompts[-1]              # 后一轮能看到前一轮
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_tool_agent.py -q`
Expected: FAIL（ImportError: ToolAgent）

- [ ] **Step 3: 实现 `ToolAgent`（追加到 `app/agent/loop.py`）**

```python
# ==== 追加到 app/agent/loop.py（imports 合并到文件头）====
import time

from app.agent.agent import GenerateFn
from app.agent.registry import ToolRegistry
from app.agent.trajectory import TrajectoryStep
from app.memory.extractor import CompleteFn
from app.memory.merger import MergeOp
from app.memory.models import MemoryFact
from app.memory.recall import format_memory_block


@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    recalled_memories: List[MemoryFact] = field(default_factory=list)
    memory_ops: List[MergeOp] = field(default_factory=list)
    stop_reason: str = "final_answer"   # "final_answer" | "budget" | "parse_abort" | "loop_abort"
    plan: List[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEMPLATE = """You are the control loop of a personal assistant agent. Solve the user's \
request step by step by calling tools and reading their observations.

Available tools:
{tools}

Terminating action (not a tool):
- final_answer: signal that the collected evidence is sufficient. The system will then \
compose the final grounded answer from everything you retrieved so far.

Rules:
- Take exactly ONE action per step.
- Return ONLY a JSON object, no prose, in one of these shapes:
  {{"thought": "...", "tool": "<tool name>", "args": {{...}}}}
  {{"thought": "...", "plan": ["todo 1", "todo 2"]}}
  {{"thought": "...", "final_answer": true}}
- You may include "plan" alongside "tool" to update your todo list while acting.
- For multi-part questions, retrieve evidence per sub-question with separate retrieve_docs \
calls, each with a different standalone query.
- If an observation reports an error, fix your next call (different tool, corrected args, \
or a reformulated query) instead of repeating the same call.
- Use final_answer only when the evidence is sufficient; if it is not, retrieve more first.
"""


class ToolAgent:
    """工具调用 agent 循环：每步一个动作，观察结果，失败反思重试，直到收尾或预算耗尽。

    全依赖注入（complete_fn / registry / generate_fn），可离线单测；
    工具调用全程 prompt-based JSON，不依赖 provider 原生 function calling。
    """

    def __init__(
        self,
        complete_fn: CompleteFn,
        registry: ToolRegistry,
        generate_fn: GenerateFn,
        max_steps: int = 6,
        max_tool_calls: int = 8,
        max_parse_retries: int = 2,
        max_duplicate_calls: int = 2,
        history_max_turns: int = 6,
    ) -> None:
        self.complete_fn = complete_fn
        self.registry = registry
        self.generate_fn = generate_fn
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.max_duplicate_calls = max_duplicate_calls
        self.history_max_turns = max(int(history_max_turns), 0)
        self.history: List[tuple[str, str]] = []
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools=registry.render_tools())

    def _format_history(self) -> str:
        lines: List[str] = []
        for user, assistant in self.history[-self.history_max_turns:]:
            lines.append(f"用户: {user}")
            lines.append(f"助手: {assistant}")
        return "\n".join(lines)

    def _render_user_prompt(self, user_msg: str, plan: List[str], steps: List[TrajectoryStep]) -> str:
        parts: List[str] = []
        history = self._format_history()
        if history:
            parts.append("Conversation so far:\n" + history)
        parts.append("User request:\n" + user_msg)
        if plan:
            parts.append("Current plan:\n" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(plan)))
        if steps:
            rendered = []
            for s in steps:
                if s.tool:
                    rendered.append(
                        f"step {s.step_index}: called {s.tool} args={json.dumps(s.args, ensure_ascii=False)}\n"
                        f"  observation: {s.observation}"
                    )
                else:
                    rendered.append(f"step {s.step_index}: {s.observation}")
            parts.append("Executed steps and observations:\n" + "\n\n".join(rendered))
        parts.append("Decide the next single action. Return ONLY JSON.")
        return "\n\n".join(parts)

    def chat(self, user_msg: str) -> AgentResult:
        msg = (user_msg or "").strip()
        if not msg:
            return AgentResult(answer="", stop_reason="final_answer")

        plan: List[str] = []
        steps: List[TrajectoryStep] = []
        acc_chunks: List[Dict[str, Any]] = []
        seen_chunk_keys: set = set()
        acc_memories: List[MemoryFact] = []
        seen_memory_ids: set = set()
        acc_ops: List[MergeOp] = []
        seen_calls: set = set()
        parse_fails = 0
        tool_calls = 0
        consecutive_dups = 0

        def finish(reason: str) -> AgentResult:
            # 任何终止路径都走同一条合成：受控/带引用/证据不足拒答由 generate_fn 保证
            try:
                gen = self.generate_fn(msg, acc_chunks, format_memory_block(acc_memories))
            except Exception:
                gen = {"answer": "抱歉，这一轮处理出错，暂时无法回答。", "sources": []}
            answer = str(gen.get("answer", ""))
            if self.history_max_turns > 0:
                self.history.append((msg, answer))
                self.history = self.history[-self.history_max_turns:]
            return AgentResult(
                answer=answer,
                sources=gen.get("sources", []) or [],
                trajectory=steps,
                recalled_memories=acc_memories,
                memory_ops=acc_ops,
                stop_reason=reason,
                plan=plan,
            )

        for step_index in range(self.max_steps):
            if tool_calls >= self.max_tool_calls:
                return finish("budget")

            raw = self.complete_fn(self.system_prompt, self._render_user_prompt(msg, plan, steps))
            decision = parse_decision(raw)

            if decision is None:
                parse_fails += 1
                steps.append(TrajectoryStep(
                    step_index=step_index,
                    observation="format error: return ONLY one valid JSON action object",
                    ok=False, error="invalid decision JSON",
                ))
                if parse_fails > self.max_parse_retries:
                    return finish("parse_abort")
                continue
            parse_fails = 0

            if decision.plan is not None:
                plan = decision.plan
            if decision.final_answer:
                steps.append(TrajectoryStep(step_index=step_index, thought=decision.thought,
                                            observation="final_answer"))
                return finish("final_answer")
            if decision.tool is None:      # plan-only 更新
                steps.append(TrajectoryStep(step_index=step_index, thought=decision.thought,
                                            observation="plan updated"))
                continue

            args = dict(decision.args)
            if decision.tool == "write_memory":
                args.setdefault("text", msg)

            call_key = (decision.tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if call_key in seen_calls:
                consecutive_dups += 1
                steps.append(TrajectoryStep(
                    step_index=step_index, thought=decision.thought, tool=decision.tool, args=args,
                    observation="duplicate call: this exact call was already made; "
                                "take a different action or final_answer",
                    ok=False, error="duplicate call",
                ))
                if consecutive_dups > self.max_duplicate_calls:
                    return finish("loop_abort")
                continue
            consecutive_dups = 0
            seen_calls.add(call_key)

            start = time.time()
            result = self.registry.dispatch(decision.tool, args)
            tool_calls += 1
            _accumulate(decision.tool, result, acc_chunks, seen_chunk_keys,
                        acc_memories, seen_memory_ids, acc_ops)
            steps.append(TrajectoryStep(
                step_index=step_index, thought=decision.thought, tool=decision.tool, args=args,
                observation=result.content if result.ok else f"error: {result.error}",
                ok=result.ok, error=result.error,
                latency_ms=(time.time() - start) * 1000.0,
            ))
            # result.ok=False 不特殊处理：错误已进观察，下一轮模型自行纠正（反思）

        return finish("budget")


def _accumulate(
    tool: str,
    result: "ToolResult",
    acc_chunks: List[Dict[str, Any]],
    seen_chunk_keys: set,
    acc_memories: List[MemoryFact],
    seen_memory_ids: set,
    acc_ops: List[MergeOp],
) -> None:
    """按工具类型把结构化载荷归集到终局合成用的证据池（去重）。"""
    if not result.ok:
        return
    if tool == "retrieve_docs":
        for c in result.data.get("chunks", []):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            key = (str(meta.get("source", "")), str(meta.get("chunk_id", id(c))))
            if key not in seen_chunk_keys:
                seen_chunk_keys.add(key)
                acc_chunks.append(c)
    elif tool == "read_memory":
        for f in result.data.get("memories", []):
            fid = getattr(f, "id", None) or id(f)
            if fid not in seen_memory_ids:
                seen_memory_ids.add(fid)
                acc_memories.append(f)
    elif tool == "write_memory":
        acc_ops.extend(result.data.get("ops", []))
```

注意：`from app.agent.tools.base import ToolResult` 需加进 imports（`_accumulate` 类型标注用）。

- [ ] **Step 4: 跑本文件 + 全量测试**

Run: `python -m pytest tests/test_tool_agent.py -q && python -m pytest -q`
Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/loop.py tests/test_tool_agent.py
git commit -m "feat(agent): ToolAgent control loop with plan, reflection retry, budgets, loop detection"
```

---

### Task 6: factory 装配 `build_tool_agent`（共享运行时抽取，避免复制粘贴）

**Files:**
- Modify: `app/agent/factory.py`

- [ ] **Step 1: 重构 + 新增**

把 `build_memory_agent` 里「embed/complete/memory store/generator/retriever」的装配抽成私有 `_build_runtime(...)`（参数与现在一致），`build_memory_agent` 改为调用它——**行为不变**。然后新增：

```python
# 追加 imports
from app.agent.loop import ToolAgent
from app.agent.registry import ToolRegistry
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from app.agent.tools.retrieve import RetrieveDocsTool


@dataclass
class ToolAgentBundle:
    agent: ToolAgent
    store: MemoryStore
    vector_index: NumpyVectorIndex
    index_path: str

    def save(self) -> None:
        self.vector_index.save(self.index_path)


def build_tool_agent(
    # 与 build_memory_agent 完全相同的参数列表，外加：
    max_steps: int = 6,
    max_tool_calls: int = 8,
) -> ToolAgentBundle:
    """Wire a real ToolAgent: same runtime as build_memory_agent, tools instead of a router."""
    rt = _build_runtime(...)   # 返回 complete_fn, store, vector_index, index_json, generate_fn, retrieve

    registry = ToolRegistry()
    registry.register(RetrieveDocsTool(retrieve_docs_fn=rt.retrieve))
    registry.register(ReadMemoryTool(store=rt.store, default_k=recall_k))
    registry.register(WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=rt.complete_fn),
        merger=MemoryMerger(store=rt.store, complete_fn=rt.complete_fn),
    ))
    registry.register(CalculatorTool())

    agent = ToolAgent(
        complete_fn=rt.complete_fn, registry=registry, generate_fn=rt.generate_fn,
        max_steps=max_steps, max_tool_calls=max_tool_calls,
    )
    return ToolAgentBundle(agent=agent, store=rt.store, vector_index=rt.vector_index, index_path=rt.index_json)
```

`_build_runtime` 返回一个小 `@dataclass _Runtime`（字段：`complete_fn, store, vector_index, index_json, generate_fn, retrieve`），两个 build 函数共用。

- [ ] **Step 2: 全量测试 + import 冒烟**

Run: `python -m pytest -q && python -c "from app.agent.factory import build_tool_agent, build_memory_agent; print('ok')"`
Expected: 全绿 + 打印 ok

- [ ] **Step 3: Commit**

```bash
git add app/agent/factory.py
git commit -m "feat(agent): build_tool_agent factory wiring, shared runtime with build_memory_agent"
```

---

### Task 7: 评测辅助 `app/eval/agent_eval.py`（程序化判定，可单测）

**Files:**
- Create: `app/eval/agent_eval.py`
- Test: `tests/test_agent_eval.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_agent_eval.py
from app.eval.agent_eval import (
    chunk_key, evidence_ids, extract_numbers, judge_success, numeric_hit, tool_precision_recall,
)

C1 = {"metadata": {"source": "a.md", "chunk_id": 1, "text": "x"}}
C2 = {"metadata": {"source": "b.md", "chunk_id": 5, "text": "y"}}


def test_chunk_key():
    assert chunk_key(C1) == "a.md#1"


def test_evidence_ids_dedup_preserves_order():
    assert evidence_ids([C1, C2, C1]) == ["a.md#1", "b.md#5"]


def test_extract_numbers():
    assert extract_numbers("增长了 50.5%，从 -3 到 47") == [50.5, -3.0, 47.0]


def test_numeric_hit_with_tolerance():
    assert numeric_hit("答案约为 50.0001%", 50.0)
    assert not numeric_hit("答案是 12", 50.0)


def test_tool_precision_recall():
    p, r = tool_precision_recall({"retrieve_docs", "calculator"}, {"retrieve_docs"})
    assert p == 0.5 and r == 1.0
    p, r = tool_precision_recall(set(), {"retrieve_docs"})
    assert p == 0.0 and r == 0.0


def test_judge_multihop_success_requires_full_gold_coverage():
    task = {"type": "multihop", "gold_chunk_ids": ["a.md#1", "b.md#5"]}
    assert judge_success(task, answer="...", evidence=[C1, C2], k=8)
    assert not judge_success(task, answer="...", evidence=[C1], k=8)


def test_judge_calc_success_by_numeric_hit():
    task = {"type": "calc", "expected_value": 50.0}
    assert judge_success(task, answer="涨幅是 50%", evidence=[], k=8)
    assert not judge_success(task, answer="不知道", evidence=[], k=8)


def test_judge_memory_success_by_answer_contains():
    task = {"type": "memory", "answer_contains": ["吉他"]}
    assert judge_success(task, answer="你在学吉他", evidence=[], k=8)
    assert not judge_success(task, answer="你在学钢琴", evidence=[], k=8)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_agent_eval.py -q`
Expected: FAIL（ModuleNotFoundError）

- [ ] **Step 3: 实现**

```python
# app/eval/agent_eval.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from app.eval.metrics import recall_at_k

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def chunk_key(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
    return f"{meta.get('source', 'unknown')}#{meta.get('chunk_id', '?')}"


def evidence_ids(chunks: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for c in chunks:
        key = chunk_key(c)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def coverage_at_k(chunks: List[Dict[str, Any]], gold_ids: Set[str], k: int) -> float:
    """多跳 Coverage@k = 累积证据 top-k 覆盖的 gold chunk 比例（即 recall@k）。"""
    return recall_at_k(evidence_ids(chunks), gold_ids, k)


def extract_numbers(text: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(text or "")]


def numeric_hit(answer: str, expected: float, rel_tol: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    for x in extract_numbers(answer):
        if abs(x - expected) <= max(abs_tol, rel_tol * abs(expected)):
            return True
    return False


def tool_precision_recall(called: Set[str], expected: Set[str]) -> Tuple[float, float]:
    if not called and not expected:
        return 1.0, 1.0
    tp = len(called & expected)
    precision = tp / len(called) if called else 0.0
    recall = tp / len(expected) if expected else 0.0
    return precision, recall


def judge_success(task: Dict[str, Any], answer: str, evidence: List[Dict[str, Any]], k: int) -> bool:
    """程序化任务成功判定（离线主指标；语义判定留给线上 LLM-as-judge）。"""
    task_type = task.get("type", "")
    if task_type == "multihop":
        gold = set(task.get("gold_chunk_ids", []))
        return bool(gold) and coverage_at_k(evidence, gold, k) >= 1.0
    if task_type == "calc":
        return numeric_hit(answer or "", float(task["expected_value"]))
    if task_type == "memory":
        needles = task.get("answer_contains", [])
        return bool(needles) and all(n in (answer or "") for n in needles)
    return False
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_agent_eval.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/eval/agent_eval.py tests/test_agent_eval.py
git commit -m "feat(eval): programmatic agent-task judging (coverage, numeric hit, tool P/R)"
```

---

### Task 8: 评测集 + `scripts/eval_agent.py`（基线 vs 循环）

**Files:**
- Create: `scripts/eval_agent.py`
- Create: `scripts/gen_agent_tasks.py`（LLM 生成任务的辅助脚本，需 API 才能跑）
- Create: `data/eval/agent_tasks_example.jsonl`（手写 3 条，schema 示范 + 冒烟）
- Create: `data/eval/agent_tasks_synth.jsonl`（用 gen 脚本生成 + 人工抽查，~20–30 条；若当场无 API/索引则先入库 example，README 注明）

**任务 schema（jsonl，每行一条）：**

```json
{"task_id": "mh_001", "type": "multihop", "question": "...", "gold_chunk_ids": ["a.md#3", "b.md#7"], "expected_tools": ["retrieve_docs"]}
{"task_id": "mem_001", "type": "memory", "question": "...", "setup_memories": ["在学吉他"], "expected_tools": ["read_memory", "retrieve_docs"], "answer_contains": ["吉他"]}
{"task_id": "calc_001", "type": "calc", "question": "...", "gold_chunk_ids": ["c.md#2"], "expected_tools": ["retrieve_docs", "calculator"], "expected_value": 50.0}
```

- [ ] **Step 1: 写 `scripts/eval_agent.py`**

结构（与 `scripts/eval_retrieval.py` 同风格：argparse + sys.path 注入 + `_load_jsonl`）：

```python
# scripts/eval_agent.py 核心骨架（完整实现按此展开）
import argparse, json, os, sys, tempfile
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.agent import MemoryAgent
from app.agent.factory import _build_doc_retriever          # 复用现成检索装配
from app.agent.llm import make_complete_fn, make_embed_fn
from app.agent.loop import ToolAgent
from app.agent.registry import ToolRegistry
from app.agent.router import Router
from app.agent.tools.calculator import CalculatorTool
from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from app.agent.tools.retrieve import RetrieveDocsTool
from app.eval.agent_eval import coverage_at_k, judge_success, tool_precision_recall
from app.eval.metrics import mean
from app.generator.generator import OpenAICompatibleGenerator
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
from app.memory.models import MemoryFact
from app.memory.store import MemoryStore
from app.memory.vector_index import NumpyVectorIndex


class CountingComplete:
    def __init__(self, fn): self.fn, self.count = fn, 0
    def __call__(self, s, u): self.count += 1; return self.fn(s, u)


class RecordingRetrieve:
    def __init__(self, fn): self.fn, self.chunks = fn, []
    def __call__(self, q):
        out = self.fn(q); self.chunks.extend(out); return out


# 每个任务独立评：
#   - 新建临时 MemoryStore（tmp dir），seed setup_memories（store.add(MemoryFact(...))）
#   - 包一层 CountingComplete / RecordingRetrieve
#   - baseline = MemoryAgent(Router, ...)；loop = ToolAgent(registry, ...)
#   - 跑 agent.chat(question)，收集：
#       coverage = coverage_at_k(recorder.chunks, gold, k)
#       success  = judge_success(task, result.answer, recorder.chunks, k)
#       tools_called: loop 从 result.trajectory 取 ok 工具集合；
#                     baseline 从 result.route 映射 {use_docs->retrieve_docs, use_memory->read_memory,
#                     write_memory->write_memory}
#       cost: llm_calls / tool_calls / steps / parse_fail_rate（trajectory 中 error=="invalid decision JSON" 占比）
# 汇总打印对照表 + 写 data/eval/agent_eval_report.json
```

CLI 参数：`--tasks data/eval/agent_tasks_synth.jsonl`、`--coverage-k 8`、`--limit N`、`--agent both|baseline|loop`、`--out data/eval/agent_eval_report.json`、以及与 `build_memory_agent` 一致的索引路径/rerank 开关参数（默认值照抄 factory）。

打印格式（示例）：

```
== agent eval (N tasks) ==
metric                baseline    loop
task_success          0.42        0.71
coverage@8 (multihop) 0.55        0.86
tool_precision        0.78        0.83
tool_recall           0.61        0.90
llm_calls/task        3.0         5.2
tool_calls/task       1.0         2.8
parse_fail_rate       0.00        0.03
```

- [ ] **Step 2: 写 `scripts/gen_agent_tasks.py`**

读 `data/index/metadatas.json`，随机采样 2–3 个不同来源/章节的 chunk 组，喂给 `make_complete_fn()`，prompt 要求生成「必须 ≥2 次工具调用」的复合任务 JSON（含 gold_chunk_ids/expected_tools/expected_value），写 `data/eval/agent_tasks_synth.jsonl`。生成后**人工抽查**再入库。

- [ ] **Step 3: 手写 `data/eval/agent_tasks_example.jsonl`**

先 `python -c "import json; ms=json.load(open('data/index/metadatas.json')); print(len(ms)); [print(m.get('source'), m.get('chunk_id'), str(m.get('text',''))[:80]) for m in ms[:20]]"` 查看真实语料，然后按上面 schema 写 3 条（multihop / memory / calc 各一），gold_chunk_ids 用真实存在的 `source#chunk_id`。

- [ ] **Step 4: 冒烟**

Run: `python scripts/eval_agent.py --tasks data/eval/agent_tasks_example.jsonl --limit 1 --agent baseline`（有 API/索引时）；无 API 时至少 `python -c "import scripts.eval_agent"` 不报错 + `python scripts/eval_agent.py --help` 正常。
Expected: 脚本可运行、报表可打印/落盘。

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_agent.py scripts/gen_agent_tasks.py data/eval/agent_tasks_example.jsonl
git commit -m "feat(eval): offline agent eval script + task schema (baseline vs tool loop)"
```

---

### Task 9: 生成评测集 + 跑对照 + README + 全量验收

**Files:**
- Create: `data/eval/agent_tasks_synth.jsonl`
- Create: `data/eval/agent_eval_report.json`
- Modify: `README.md`

- [ ] **Step 1: 生成并抽查评测集（需 API + 文档索引）**

Run: `python scripts/gen_agent_tasks.py --n 24 --out data/eval/agent_tasks_synth.jsonl`
然后人工抽查 gold_chunk_ids 是否真实、任务是否确实需要 ≥2 次工具调用；修掉坏样本。

- [ ] **Step 2: 跑基线 vs 循环对照**

Run: `python scripts/eval_agent.py --tasks data/eval/agent_tasks_synth.jsonl --out data/eval/agent_eval_report.json`
Expected: 打印对照表；把数字记录下来填 README。

- [ ] **Step 3: README 增补一节**

在 README.md 增加「工具调用 Agent 循环」小节，内容必须含：
1. 链路图/文字版（循环 → 注册表 → 4 工具 + final_answer 特判 → 终局合成复用受控生成）。
2. 评测结果表（基线 vs 循环：任务成功率、多跳 Coverage@k、工具选择 P/R、每任务平均 LLM/工具调用数、解析失败率）——用 Step 2 的真实数字。
3. 诚实边界（照 spec 第 12 节五条写：prompt-based 不如原生 FC 稳、评测集小仅信相对结论、多步更慢更贵 +N 次调用、calculator 是演示工具、记忆侧无新消融）。
4. 运行方式：`python scripts/eval_agent.py ...`。

- [ ] **Step 4: 全量验收**

Run: `python -m pytest -q`
Expected: 全绿（新老全部）。对照 spec 第 11 节逐条勾验收。

- [ ] **Step 5: Commit**

```bash
git add README.md data/eval/agent_tasks_synth.jsonl data/eval/agent_eval_report.json
git commit -m "docs+eval: agent loop section, baseline-vs-loop numbers, honest limits"
```

---

## 验收对照（spec §11）

- [ ] 循环跑通、5 动作可用、任何路径返回合法 `AgentResult`（Task 5 测试覆盖）
- [ ] 全程 prompt-based JSON、`complete_fn` 注入（Task 5）
- [ ] 六类失败场景各有测试（Task 2/5：非法 JSON、未知工具、坏参数、工具异常、预算、死循环；Task 1：calculator 除零/非算术）
- [ ] `scripts/eval_agent.py` 离线可跑、打印基线 vs 循环（Task 8/9）
- [ ] 轨迹随结果返回 + `format_trajectory` 可打印（Task 4/5）
- [ ] README 增补 + 全量 pytest 全绿（Task 9）
