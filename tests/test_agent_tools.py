from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool
from app.agent.tools.retrieve import RetrieveDocsTool
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
from app.memory.models import MemoryFact
from app.memory.store import MemoryStore
from app.memory.vector_index import NumpyVectorIndex

VOCAB = ["guitar", "gym", "python", "faiss", "project"]


def fake_embed(text: str):
    t = (text or "").lower()
    return [1.0 if w in t else 0.0 for w in VOCAB]


def const(canned: str):
    def _complete(system_prompt: str, user_prompt: str) -> str:
        return canned
    return _complete


def make_store(tmp_path):
    return MemoryStore(
        db_path=str(tmp_path / "mem.db"),
        vector_index=NumpyVectorIndex(dim=len(VOCAB)),
        embed_fn=fake_embed,
    )


# ---- retrieve_docs ----

CHUNKS = [
    {"metadata": {"source": "a.md", "chunk_id": 1, "text": "FAISS supports IVF indexes"}},
    {"metadata": {"source": "b.md", "chunk_id": 5, "text": "BM25 is sparse retrieval"}},
]


def test_retrieve_docs_returns_chunks_and_readable_content():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "faiss"})
    assert r.ok
    assert r.data["chunks"] == CHUNKS
    assert "a.md#1" in r.content and "IVF" in r.content


def test_retrieve_docs_k_caps_results():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "faiss", "k": 1})
    assert len(r.data["chunks"]) == 1


def test_retrieve_docs_empty_query_rejected():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: list(CHUNKS))
    r = tool.run({"query": "  "})
    assert not r.ok and "empty query" in r.error


def test_retrieve_docs_no_hits_is_ok_with_hint():
    tool = RetrieveDocsTool(retrieve_docs_fn=lambda q: [])
    r = tool.run({"query": "zzz"})
    assert r.ok and r.data["chunks"] == [] and "no documents" in r.content


def test_retrieve_docs_fn_exception_wrapped():
    def boom(q):
        raise RuntimeError("index missing")
    r = RetrieveDocsTool(retrieve_docs_fn=boom).run({"query": "x"})
    assert not r.ok and "index missing" in r.error


# ---- read_memory ----

def test_read_memory_recalls_relevant_facts(tmp_path):
    store = make_store(tmp_path)
    store.add(MemoryFact(fact_content="has been learning guitar since 2026"))
    r = ReadMemoryTool(store=store).run({"query": "guitar"})
    assert r.ok
    assert any("guitar" in f.fact_content for f in r.data["memories"])
    assert "guitar" in r.content


def test_read_memory_empty_store(tmp_path):
    r = ReadMemoryTool(store=make_store(tmp_path)).run({"query": "guitar"})
    assert r.ok and r.data["memories"] == [] and "no relevant memories" in r.content


# ---- write_memory ----

def test_write_memory_extracts_and_merges(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const(
            '{"facts": [{"fact_object": "project", "fact_content": "is building a faiss project"}]}'
        )),
        merger=MemoryMerger(store=store, complete_fn=const(
            '{"operations": [{"type": "add", "fact_object": "project", "fact_content": "is building a faiss project"}]}'
        )),
    )
    r = tool.run({"text": "我在做一个 faiss 项目"})
    assert r.ok
    assert [op.type for op in r.data["ops"]] == ["add"]
    assert "1 add" in r.content
    assert any("faiss project" in f.fact_content for f in store.list_active())


def test_write_memory_nothing_durable(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const('{"facts": []}')),
        merger=MemoryMerger(store=store, complete_fn=const('{"operations": []}')),
    )
    r = tool.run({"text": "哈哈好的"})
    assert r.ok and r.data["ops"] == [] and store.list_active() == []


def test_write_memory_empty_text_rejected(tmp_path):
    store = make_store(tmp_path)
    tool = WriteMemoryTool(
        extractor=MemoryExtractor(complete_fn=const("[]")),
        merger=MemoryMerger(store=store, complete_fn=const('{"operations": []}')),
    )
    r = tool.run({})
    assert not r.ok and "empty text" in r.error
