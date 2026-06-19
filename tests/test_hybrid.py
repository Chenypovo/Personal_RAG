from app.retriever.hybrid import rrf_fuse


def item(source, cid, **extra):
    return {"metadata": {"source": source, "chunk_id": cid}, **extra}


def _keys(out):
    return [(o["metadata"]["source"], o["metadata"]["chunk_id"]) for o in out]


def test_rrf_dedups_and_ranks_shared_doc_first():
    vec = [item("a", 0), item("b", 0)]
    bm25 = [item("b", 0), item("c", 0)]
    out = rrf_fuse(vec, bm25, rrf_k=60)
    keys = _keys(out)
    assert keys[0] == ("b", 0)          # appears in both lists -> highest fused score
    assert len(keys) == len(set(keys)) == 3  # deduplicated union


def test_rrf_respects_top_k():
    vec = [item("a", 0), item("b", 0), item("c", 0)]
    out = rrf_fuse(vec, [], rrf_k=60, top_k=2)
    assert len(out) == 2


def test_rrf_sets_hybrid_score_and_keeps_metadata():
    out = rrf_fuse([item("a", 0, distance=0.3)], [], rrf_k=60)
    assert "hybrid_score" in out[0]
    assert out[0]["metadata"]["source"] == "a"


def test_rrf_empty_inputs():
    assert rrf_fuse([], [], rrf_k=60) == []
