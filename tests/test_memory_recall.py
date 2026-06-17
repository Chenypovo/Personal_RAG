from app.memory.models import MemoryFact
from app.memory.recall import format_memory_block, recall_memories
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


def test_recall_returns_relevant_active_facts(tmp_path):
    store = make_store(tmp_path)
    g = store.add(MemoryFact(fact_content="learning guitar"))
    store.add(MemoryFact(fact_content="going to the gym"))

    facts = recall_memories(store, "guitar lessons", top_k=5)
    assert [f.id for f in facts] == [g.id]


def test_recall_empty_when_nothing_relevant(tmp_path):
    store = make_store(tmp_path)
    store.add(MemoryFact(fact_content="learning guitar"))
    assert recall_memories(store, "python programming", top_k=5) == []


def test_format_memory_block_lists_contents():
    facts = [
        MemoryFact(fact_content="learning guitar", fact_object="guitar"),
        MemoryFact(fact_content="lives in Singapore"),
    ]
    block = format_memory_block(facts)
    assert "learning guitar" in block
    assert "lives in Singapore" in block


def test_format_memory_block_empty_is_blank():
    assert format_memory_block([]) == ""
