from __future__ import annotations

from typing import Iterable, List, Set


def recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k predictions."""
    if not gold:
        return 0.0
    hit = len(set(pred[:k]).intersection(gold))
    return hit / float(len(gold))


def mrr_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    """Reciprocal rank of the first relevant item within the top-k."""
    for rank, item_id in enumerate(pred[:k], start=1):
        if item_id in gold:
            return 1.0 / float(rank)
    return 0.0


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / float(len(vals)) if vals else 0.0
