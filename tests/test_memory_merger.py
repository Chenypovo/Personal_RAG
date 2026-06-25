from app.memory.extractor import ExtractedFact
from app.memory.merger import MemoryMerger
from app.memory.models import MemoryFact
from app.memory.store import MemoryStore
from app.memory.vector_index import NumpyVectorIndex


VOCAB = ["guitar", "gym", "python"]


def fake_embed(text: str):
    t = (text or "").lower()
    return [1.0 if w in t else 0.0 for w in VOCAB]


def make_store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "memory.db"),
        vector_index=NumpyVectorIndex(dim=len(VOCAB)),
        embed_fn=fake_embed,
    )


def const_complete(canned: str):
    def _complete(system_prompt: str, user_prompt: str) -> str:
        return canned
    return _complete


def test_update_supersedes_old_fact_non_destructively(tmp_path):
    store = make_store(tmp_path)
    old = store.add(MemoryFact(fact_content="learning guitar python", fact_object="skill"))
    # LLM updates with a clearly-similar refinement (same vocab -> cosine 1.0)
    merger = MemoryMerger(
        store=store,
        complete_fn=const_complete(
            '{"operations":[{"type":"update","id":"%s","fact_content":"now advanced at guitar python"}]}' % old.id
        ),
    )
    merger.merge([ExtractedFact(fact_content="now advanced at guitar python", fact_object="skill")])

    assert store.get(old.id).state == "SUPERSEDED"           # old kept, not overwritten
    active = store.list_active()
    assert len(active) == 1 and "advanced" in active[0].fact_content


def test_dissimilar_update_downgrades_to_add_no_data_loss(tmp_path):
    # The "我喜欢 Claude" vs "我试了 Cursor" case: recalled as loosely related, but the
    # LLM over-eagerly says UPDATE. The similarity gate must refuse and ADD instead.
    store = make_store(tmp_path)
    old = store.add(MemoryFact(fact_content="learning guitar python"))  # [guitar,python]
    merger = MemoryMerger(
        store=store,
        complete_fn=const_complete(
            '{"operations":[{"type":"update","id":"%s","fact_content":"started guitar gym"}]}' % old.id
        ),
    )
    # "guitar gym" vs "guitar python": cosine 0.5 -> recalled but below the 0.8 gate
    ops = merger.merge([ExtractedFact(fact_content="started guitar gym")])

    assert store.get(old.id).state == "ACTIVE"               # old fact untouched -> no data loss
    assert store.get(old.id).fact_content == "learning guitar python"
    contents = {f.fact_content for f in store.list_active()}
    assert contents == {"learning guitar python", "started guitar gym"}
    assert any(o.type == "add" for o in ops)                 # update downgraded to add


def test_empty_new_facts_no_llm_call(tmp_path):
    called = {"n": 0}

    def fake(s, u):
        called["n"] += 1
        return "{}"

    merger = MemoryMerger(store=make_store(tmp_path), complete_fn=fake)
    assert merger.merge([]) == []
    assert called["n"] == 0


def test_add_operation_creates_fact(tmp_path):
    store = make_store(tmp_path)
    merger = MemoryMerger(
        store=store,
        complete_fn=const_complete('{"operations": [{"type": "add", "fact_object": "guitar", "fact_content": "started learning guitar"}]}'),
    )

    ops = merger.merge([ExtractedFact(fact_content="started learning guitar", fact_object="guitar")])

    active = store.list_active()
    assert len(active) == 1
    assert active[0].fact_content == "started learning guitar"
    assert ops[0].type == "add" and ops[0].applied


def test_update_operation_modifies_existing(tmp_path):
    store = make_store(tmp_path)
    existing = store.add(MemoryFact(fact_content="learning guitar", fact_object="guitar"))

    merger = MemoryMerger(
        store=store,
        complete_fn=const_complete(
            '{"operations": [{"type": "update", "id": "%s", "fact_content": "plays guitar in a band"}]}' % existing.id
        ),
    )
    ops = merger.merge([ExtractedFact(fact_content="now plays guitar in a band", fact_object="guitar")])

    # update is non-destructive: old superseded, new content is the active memory
    assert store.get(existing.id).state == "SUPERSEDED"
    active = store.list_active()
    assert len(active) == 1 and active[0].fact_content == "plays guitar in a band"
    assert ops[0].type == "update" and ops[0].applied


def test_delete_operation_soft_deletes(tmp_path):
    store = make_store(tmp_path)
    existing = store.add(MemoryFact(fact_content="learning guitar", fact_object="guitar"))

    merger = MemoryMerger(
        store=store,
        complete_fn=const_complete('{"operations": [{"type": "delete", "id": "%s"}]}' % existing.id),
    )
    ops = merger.merge([ExtractedFact(fact_content="quit guitar", fact_object="guitar")])

    assert store.get(existing.id).state == "DELETED"
    assert ops[0].type == "delete" and ops[0].applied


def test_parse_failure_falls_back_to_add_all(tmp_path):
    store = make_store(tmp_path)
    merger = MemoryMerger(store=store, complete_fn=const_complete("garbage not json"))

    new = [
        ExtractedFact(fact_content="learning guitar"),
        ExtractedFact(fact_content="going to the gym"),
    ]
    ops = merger.merge(new)

    contents = {f.fact_content for f in store.list_active()}
    assert contents == {"learning guitar", "going to the gym"}
    assert all(o.type == "add" and o.applied for o in ops)


def test_existing_memories_offered_to_llm(tmp_path):
    store = make_store(tmp_path)
    existing = store.add(MemoryFact(fact_content="learning guitar", fact_object="guitar"))

    seen = {}

    def capture(system_prompt: str, user_prompt: str) -> str:
        seen["user"] = user_prompt
        return '{"operations": []}'

    merger = MemoryMerger(store=store, complete_fn=capture)
    merger.merge([ExtractedFact(fact_content="practiced guitar today", fact_object="guitar")])

    # the relevant existing memory (and its id) should be offered for merge decisions
    assert existing.id in seen["user"]
    assert "learning guitar" in seen["user"]
    assert "practiced guitar today" in seen["user"]
