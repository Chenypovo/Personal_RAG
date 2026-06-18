from app.retriever.parent_child import expand_to_parents


def chunk(source, cid, pid, text):
    return {"source": source, "chunk_id": cid, "parent_id": pid, "text": text}


ALL = [
    chunk("a.md", 0, "a.md#p0", "intro one"),
    chunk("a.md", 1, "a.md#p0", "intro two"),
    chunk("a.md", 2, "a.md#p0", "intro three"),
    chunk("a.md", 3, "a.md#p1", "body one"),
    chunk("b.md", 0, "b.md#p0", "other doc"),
]


def hit(meta, distance=0.1):
    return {"metadata": meta, "distance": distance}


def test_expands_child_to_full_parent():
    out = expand_to_parents([hit(ALL[1])], ALL)
    assert len(out) == 1
    txt = out[0]["metadata"]["text"]
    assert "intro one" in txt and "intro two" in txt and "intro three" in txt


def test_dedups_parent_when_multiple_children_hit():
    out = expand_to_parents([hit(ALL[0]), hit(ALL[2])], ALL)
    assert len(out) == 1  # both children share one parent -> single block


def test_preserves_rank_order_across_parents():
    out = expand_to_parents([hit(ALL[3], 0.1), hit(ALL[0], 0.2)], ALL)
    assert [o["metadata"]["parent_id"] for o in out] == ["a.md#p1", "a.md#p0"]


def test_concatenates_children_in_chunk_order():
    out = expand_to_parents([hit(ALL[2])], ALL)  # hit the last child
    txt = out[0]["metadata"]["text"]
    assert txt.index("intro one") < txt.index("intro two") < txt.index("intro three")


def test_chunk_without_parent_id_returned_as_is():
    lone = {"source": "c.md", "chunk_id": 0, "text": "lonely"}
    out = expand_to_parents([hit(lone)], [lone])
    assert out[0]["metadata"]["text"] == "lonely"


def test_carries_score_and_child_ids():
    out = expand_to_parents([hit(ALL[1], distance=0.42)], ALL)
    assert out[0]["distance"] == 0.42
    assert set(out[0]["metadata"]["child_chunk_ids"]) == {0, 1, 2}
