from app.reranker.bge_reranker import BGEReranker


def item(text, **extra):
    d = {"metadata": {"text": text}}
    d.update(extra)
    return d


def test_orders_by_score_desc_and_truncates_top_k():
    scores = {"low": 0.1, "mid": 0.5, "high": 0.9}

    def score_fn(pairs):
        return [scores[doc] for (_q, doc) in pairs]

    r = BGEReranker(score_fn=score_fn)
    out = r.rerank("q", [item("low"), item("high"), item("mid")], top_k=2)

    assert [i["metadata"]["text"] for i in out] == ["high", "mid"]
    assert out[0]["rerank_score"] == 0.9


def test_empty_query_returns_prefix_without_scoring():
    calls = {"n": 0}

    def score_fn(pairs):
        calls["n"] += 1
        return [0.0] * len(pairs)

    r = BGEReranker(score_fn=score_fn)
    items = [item("a"), item("b")]
    out = r.rerank("   ", items, top_k=1)

    assert out == items[:1]
    assert calls["n"] == 0


def test_items_without_text_sink_to_bottom():
    def score_fn(pairs):
        return [0.5 for _ in pairs]

    r = BGEReranker(score_fn=score_fn)
    out = r.rerank("q", [{"metadata": {"text": ""}}, item("good")], top_k=2)

    assert out[0]["metadata"]["text"] == "good"


def test_score_fn_receives_query_doc_pairs():
    seen = {}

    def score_fn(pairs):
        seen["pairs"] = list(pairs)
        return [0.0 for _ in pairs]

    r = BGEReranker(score_fn=score_fn)
    r.rerank("myquery", [item("doc1"), item("doc2")], top_k=2)

    assert seen["pairs"] == [("myquery", "doc1"), ("myquery", "doc2")]
