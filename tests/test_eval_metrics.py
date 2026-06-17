from app.eval.metrics import mean, mrr_at_k, recall_at_k, summarize_rank_metrics


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "x"}, 2) == 0.5


def test_mrr_at_k():
    assert mrr_at_k(["a", "b", "c"], {"c"}, 3) == 1.0 / 3.0


def test_summarize_rank_metrics():
    summary = summarize_rank_metrics(["a", "b"], {"b"}, [1, 2])
    assert summary["recall@1"] == 0.0
    assert summary["recall@2"] == 1.0
    assert summary["mrr@2"] == 0.5


def test_mean_empty_is_zero():
    assert mean([]) == 0.0
