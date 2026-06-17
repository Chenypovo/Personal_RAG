from app.reranker.bge_reranker import BGEReranker


def test_reranker_uses_injected_scores():
    reranker = BGEReranker(score_fn=lambda pairs: [0.1, 0.9])
    retrieved = [
        {"metadata": {"text": "low score doc"}},
        {"metadata": {"text": "high score doc"}},
    ]

    ranked = reranker.rerank("query", retrieved, top_k=2)

    assert ranked[0]["metadata"]["text"] == "high score doc"
    assert ranked[0]["rerank_score"] == 0.9


def test_reranker_skips_empty_text_docs():
    reranker = BGEReranker(score_fn=lambda pairs: [0.5])
    retrieved = [
        {"metadata": {"text": ""}},
        {"metadata": {"text": "usable"}},
    ]

    ranked = reranker.rerank("query", retrieved, top_k=2)

    assert ranked[0]["metadata"]["text"] == "usable"
    assert ranked[1]["rerank_score"] == float("-inf")
