from app.eval.metrics import mean, mrr_at_k, recall_at_k


def test_recall_at_k_fraction_of_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "x"}, k=3) == 0.5  # 1 of 2 gold


def test_recall_at_k_respects_cutoff():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0  # c is rank 3, beyond k=2


def test_recall_at_k_empty_gold_is_zero():
    assert recall_at_k(["a"], set(), k=1) == 0.0


def test_mrr_at_k_reciprocal_of_first_hit_rank():
    assert mrr_at_k(["a", "b", "c"], {"b"}, k=3) == 0.5     # first hit rank 2
    assert mrr_at_k(["a", "b"], {"a"}, k=2) == 1.0          # rank 1


def test_mrr_at_k_no_hit_is_zero():
    assert mrr_at_k(["a", "b"], {"z"}, k=2) == 0.0


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0
