from app.agent.loop import ToolAgent, parse_decision
from app.agent.registry import ToolRegistry
from app.agent.tools.base import ToolResult
from app.agent.trajectory import TrajectoryStep, format_trajectory
from app.memory.models import MemoryFact


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


# ---- ToolAgent loop ----

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
    """内存态小语料检索工具，接口与 RetrieveDocsTool 一致。"""
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


def test_multihop_two_retrievals_then_final():
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
    responses = ['{"final_answer": true}'] * 5
    agent, complete, _ = build_agent(responses, tools=[FakeSearch()], history_max_turns=2)
    for i in range(5):
        agent.chat(f"msg{i}")
    assert len(agent.history) <= 2
    assert "msg3" in complete.prompts[-1]              # 后一轮能看到前一轮
