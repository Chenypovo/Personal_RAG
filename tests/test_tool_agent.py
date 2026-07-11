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
