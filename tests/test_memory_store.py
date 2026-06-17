import threading

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


def test_add_and_get(tmp_path):
    store = make_store(tmp_path)
    fact = MemoryFact(fact_content="started learning guitar", fact_object="guitar", source="chat")
    store.add(fact)

    got = store.get(fact.id)
    assert got is not None
    assert got.fact_content == "started learning guitar"
    assert got.fact_object == "guitar"
    assert got.state == "ACTIVE"


def test_list_active_excludes_deleted(tmp_path):
    store = make_store(tmp_path)
    a = store.add(MemoryFact(fact_content="learning guitar"))
    b = store.add(MemoryFact(fact_content="going to the gym"))

    store.soft_delete(a.id)

    active_ids = {f.id for f in store.list_active()}
    assert active_ids == {b.id}


def test_soft_delete_removes_from_search(tmp_path):
    store = make_store(tmp_path)
    a = store.add(MemoryFact(fact_content="learning guitar"))

    store.soft_delete(a.id)

    assert store.search("guitar", top_k=5) == []
    assert store.get(a.id).state == "DELETED"


def test_update_reembeds_for_search(tmp_path):
    store = make_store(tmp_path)
    fact = store.add(MemoryFact(fact_content="learning guitar"))

    store.update(fact.id, fact_content="now coding in python")

    # old content no longer matches; new content does
    guitar_hits = [f.id for f, _ in store.search("guitar", top_k=5)]
    python_hits = [f.id for f, _ in store.search("python", top_k=5)]
    assert fact.id not in guitar_hits
    assert fact.id in python_hits
    assert store.get(fact.id).updated_at >= fact.updated_at


def test_search_orders_by_similarity_and_only_active(tmp_path):
    store = make_store(tmp_path)
    g = store.add(MemoryFact(fact_content="learning guitar"))
    store.add(MemoryFact(fact_content="going to the gym"))

    results = store.search("guitar practice", top_k=5)
    assert results[0][0].id == g.id
    assert results[0][1] >= results[-1][1]


def test_persists_across_instances(tmp_path):
    store = make_store(tmp_path)
    fact = store.add(MemoryFact(fact_content="learning guitar"))

    reopened = make_store(tmp_path)  # same db file, fresh vector index
    got = reopened.get(fact.id)
    assert got is not None
    assert got.fact_content == "learning guitar"


def test_rebuild_index_from_truth_store(tmp_path):
    store = make_store(tmp_path)
    fact = store.add(MemoryFact(fact_content="learning guitar"))

    reopened = make_store(tmp_path)  # vector index empty until rebuilt
    assert reopened.search("guitar", top_k=5) == []

    count = reopened.rebuild_index()
    assert count == 1
    assert [f.id for f, _ in reopened.search("guitar", top_k=5)] == [fact.id]


def test_usable_from_another_thread(tmp_path):
    # The FastAPI server creates the store in one thread but handles requests in
    # a worker thread; the store must not raise sqlite "same thread" errors.
    store = make_store(tmp_path)
    fact = MemoryFact(fact_content="learning guitar")
    err = {}

    def worker():
        try:
            store.add(fact)
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "e" not in err, f"store raised across threads: {err.get('e')}"
    assert store.get(fact.id) is not None
