from app.eval.memory_eval import MemQuery, evaluate_recall, merge_accuracy


def test_evaluate_recall_computes_metrics():
    results = {"qa": ["m1", "x"], "qb": ["x", "m2", "m3"]}
    queries = [MemQuery("q1", "qa"), MemQuery("q2", "qb")]
    qrels = {"q1": {"m1"}, "q2": {"m2", "m3"}}

    def search_fn(query, top_k):
        return results[query][:top_k]

    out = evaluate_recall(search_fn, queries, qrels, ks=[1, 3])

    assert out["metrics"]["recall@1"] == 0.5   # (1.0 + 0.0)/2
    assert out["metrics"]["recall@3"] == 1.0   # (1.0 + 1.0)/2
    assert out["metrics"]["mrr@1"] == 0.5      # (1.0 + 0.0)/2
    assert out["metrics"]["mrr@3"] == 0.75     # (1.0 + 0.5)/2
    assert out["n_eval"] == 2


def test_skips_queries_without_qrels():
    queries = [MemQuery("q1", "qa"), MemQuery("q9", "qz")]
    qrels = {"q1": {"m1"}}

    def search_fn(query, top_k):
        return ["m1"]

    out = evaluate_recall(search_fn, queries, qrels, ks=[1])
    assert out["n_eval"] == 1


def test_merge_accuracy():
    assert merge_accuracy([("add", "add"), ("add", "update"), ("delete", "delete")]) == 2 / 3
    assert merge_accuracy([]) == 0.0
