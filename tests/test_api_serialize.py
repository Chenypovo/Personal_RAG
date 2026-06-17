from app.agent.agent import AgentResult
from app.agent.router import RouteDecision
from app.api.serialize import agent_result_to_dict, memory_to_dict
from app.memory.merger import MergeOp
from app.memory.models import MemoryFact


def test_memory_to_dict():
    f = MemoryFact(fact_content="likes green tea", fact_object="drink", id="x1")
    d = memory_to_dict(f)
    assert d == {"id": "x1", "fact_object": "drink", "fact_content": "likes green tea"}


def test_agent_result_to_dict():
    res = AgentResult(
        answer="hi",
        sources=[{"citation": "c1"}],
        recalled_memories=[MemoryFact(fact_content="m", id="m1")],
        memory_ops=[MergeOp(type="add", fact_content="m", applied=True)],
        route=RouteDecision(use_docs=True, use_memory=False, write_memory=True, rewritten_query="q"),
    )
    d = agent_result_to_dict(res)

    assert d["answer"] == "hi"
    assert d["sources"] == [{"citation": "c1"}]
    assert d["recalled_memories"][0]["id"] == "m1"
    assert d["memory_ops"][0]["type"] == "add"
    assert d["memory_ops"][0]["applied"] is True
    assert d["route"]["write_memory"] is True
    assert d["route"]["use_memory"] is False


def test_agent_result_to_dict_handles_none_route():
    res = AgentResult(answer="x")
    d = agent_result_to_dict(res)
    assert d["answer"] == "x"
    assert d["recalled_memories"] == []
    assert d["memory_ops"] == []
    assert d["route"] is None
