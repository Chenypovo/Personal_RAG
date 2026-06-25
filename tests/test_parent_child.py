from app.retriever.parent_child import assign_parent_ids, expand_to_parents


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


def test_parent_id_groups_by_heading_section():
    chunks = [
        {"source": "a.md", "chunk_id": 0, "heading": "Doc > Sec A"},
        {"source": "a.md", "chunk_id": 1, "heading": "Doc > Sec A"},
        {"source": "a.md", "chunk_id": 2, "heading": "Doc > Sec B"},
    ]
    assign_parent_ids(chunks, window=3)
    assert chunks[0]["parent_id"] == chunks[1]["parent_id"] == "a.md::Doc > Sec A"
    assert chunks[2]["parent_id"] == "a.md::Doc > Sec B"


def test_parent_id_falls_back_to_window_without_heading():
    chunks = [{"source": "t.txt", "chunk_id": i} for i in range(4)]
    assign_parent_ids(chunks, window=2)
    assert chunks[0]["parent_id"] == chunks[1]["parent_id"] == "t.txt#p0"
    assert chunks[2]["parent_id"] == chunks[3]["parent_id"] == "t.txt#p1"


SECTION = [chunk("a.md", i, "a.md::S", f"t{i}") for i in range(6)]  # one big section, 6 chunks


def test_section_parent_capped_and_centered_on_hit():
    out = expand_to_parents([hit(SECTION[3])], SECTION, max_parent_chunks=3)
    assert out[0]["metadata"]["child_chunk_ids"] == [2, 3, 4]  # centered on hit cid=3
    txt = out[0]["metadata"]["text"]
    assert "t2" in txt and "t3" in txt and "t4" in txt
    assert "t0" not in txt and "t5" not in txt


def test_section_parent_cap_clamps_at_edge():
    out = expand_to_parents([hit(SECTION[0])], SECTION, max_parent_chunks=3)
    assert out[0]["metadata"]["child_chunk_ids"] == [0, 1, 2]


def test_no_cap_returns_whole_parent():
    out = expand_to_parents([hit(SECTION[3])], SECTION)
    assert out[0]["metadata"]["child_chunk_ids"] == [0, 1, 2, 3, 4, 5]
