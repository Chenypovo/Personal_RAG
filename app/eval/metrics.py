from __future__ import annotations

from typing import Dict, Iterable, List, Set


def recall_at_k(predicted_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not gold_ids:
        return 0.0
    hit = len(set(predicted_ids[:k]).intersection(gold_ids))
    return hit / float(len(gold_ids))


def mrr_at_k(predicted_ids: List[str], gold_ids: Set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    for rank, item_id in enumerate(predicted_ids[:k], start=1):
        if item_id in gold_ids:
            return 1.0 / float(rank)
    return 0.0


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / float(len(items)) if items else 0.0


def summarize_rank_metrics(
    predicted_ids: List[str],
    gold_ids: Set[str],
    ks: List[int],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(predicted_ids, gold_ids, k)
        metrics[f"mrr@{k}"] = mrr_at_k(predicted_ids, gold_ids, k)
    return metrics

