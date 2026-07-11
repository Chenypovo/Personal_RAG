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
