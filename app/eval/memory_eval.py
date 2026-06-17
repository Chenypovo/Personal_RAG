from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Set, Tuple

from app.eval.metrics import mean, mrr_at_k, recall_at_k

# (query, top_k) -> ranked list of memory ids
SearchFn = Callable[[str, int], List[str]]


@dataclass
class MemQuery:
    query_id: str
    query: str


def evaluate_recall(
    search_fn: SearchFn,
    queries: List[MemQuery],
    qrels: Dict[str, Set[str]],
    ks: List[int],
) -> Dict[str, Any]:
    """Evaluate memory recall with Recall@K / MRR@K (same metrics as retrieval eval)."""
    ks = sorted(set(ks))
    top_k_max = max(ks)
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: Dict[int, List[float]] = {k: [] for k in ks}
    per_query: List[Dict[str, Any]] = []

    for q in queries:
        gold = qrels.get(q.query_id, set())
        if not gold:
            continue
        pred = search_fn(q.query, top_k_max)
        row: Dict[str, Any] = {"query_id": q.query_id, "pred": pred}
        for k in ks:
            r = recall_at_k(pred, gold, k)
            m = mrr_at_k(pred, gold, k)
            recalls[k].append(r)
            mrrs[k].append(m)
            row[f"recall@{k}"] = r
            row[f"mrr@{k}"] = m
        per_query.append(row)

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = mean(recalls[k])
        metrics[f"mrr@{k}"] = mean(mrrs[k])

    return {"metrics": metrics, "per_query": per_query, "n_eval": len(per_query)}


def merge_accuracy(cases: List[Tuple[str, str]]) -> float:
    """Fraction of (predicted_op_type, expected_op_type) pairs that match."""
    if not cases:
        return 0.0
    correct = sum(1 for pred, gold in cases if pred == gold)
    return correct / float(len(cases))
