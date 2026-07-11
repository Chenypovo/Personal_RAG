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
