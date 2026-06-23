from app.agent.agent import MemoryAgent
from app.agent.router import Router
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
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


def build_agent(tmp_path, *, router_json, extractor_json, merger_json, captured):
    store = MemoryStore(
        db_path=str(tmp_path / "mem.db"),
        vector_index=NumpyVectorIndex(dim=len(VOCAB)),
        embed_fn=fake_embed,
    )

    def retrieve_docs(query):
        captured["doc_query"] = query
        return [{"metadata": {"source": "doc.txt", "chunk_id": 0, "text": "FAISS is a vector library"}}]

    def generate(query, chunks, user_memory):
        captured["gen_query"] = query
        captured["gen_chunks"] = chunks
        captured["gen_memory"] = user_memory
        return {"answer": "an answer", "sources": [{"citation": "Chunk 0, doc.txt"}]}

    agent = MemoryAgent(
        router=Router(complete_fn=const(router_json)),
        store=store,
        extractor=MemoryExtractor(complete_fn=const(extractor_json)),
        merger=MemoryMerger(store=store, complete_fn=const(merger_json)),
        retrieve_docs_fn=retrieve_docs,
        generate_fn=generate,
    )
    return agent, store


def test_full_turn_recalls_retrieves_and_writes(tmp_path):
    captured = {}
    agent, store = build_agent(
        tmp_path,
        router_json='{"use_docs": true, "use_memory": true, "write_memory": true, "rewritten_query": "my faiss project"}',
        extractor_json='{"facts": [{"fact_object": "project", "fact_content": "is building a faiss project"}]}',
        merger_json='{"operations": [{"type": "add", "fact_object": "project", "fact_content": "is building a faiss project"}]}',
        captured=captured,
    )

    result = agent.chat("tell me about my faiss project")

    # retrieved docs with rewritten query
    assert captured["doc_query"] == "my faiss project"
    assert captured["gen_chunks"]
    # memory written to store
    assert any("faiss project" in f.fact_content for f in store.list_active())
    assert result.answer == "an answer"
    assert result.memory_ops and result.memory_ops[0].type == "add"


def test_recalled_memory_injected_into_generation(tmp_path):
    captured = {}
    agent, store = build_agent(
        tmp_path,
        router_json='{"use_docs": false, "use_memory": true, "write_memory": false, "rewritten_query": "guitar"}',
        extractor_json="[]",
        merger_json='{"operations": []}',
        captured=captured,
    )
    # seed an existing memory
    from app.memory.models import MemoryFact
    store.add(MemoryFact(fact_content="has been learning guitar since 2026"))

    result = agent.chat("what instrument am I learning?")

    assert "learning guitar" in captured["gen_memory"]
    assert any("learning guitar" in f.fact_content for f in result.recalled_memories)


def test_route_off_skips_docs_memory_and_write(tmp_path):
    captured = {}
    agent, store = build_agent(
        tmp_path,
        router_json='{"use_docs": false, "use_memory": false, "write_memory": false, "rewritten_query": "hi"}',
        extractor_json='{"facts": [{"fact_content": "should not be written"}]}',
        merger_json='{"operations": [{"type": "add", "fact_content": "should not be written"}]}',
        captured=captured,
    )

    result = agent.chat("hello there")

    assert captured.get("doc_query") is None       # docs not retrieved
    assert captured["gen_chunks"] == []            # generation got no docs
    assert captured["gen_memory"] == ""            # no memory injected
    assert store.list_active() == []               # nothing written
    assert result.memory_ops == []


def _history_agent(tmp_path, router_prompts, history_max_turns=6):
    store = MemoryStore(
        db_path=str(tmp_path / "mem.db"),
        vector_index=NumpyVectorIndex(dim=len(VOCAB)),
        embed_fn=fake_embed,
    )

    def router_complete(system_prompt, user_prompt):
        router_prompts.append(user_prompt)
        return '{"use_docs": false, "use_memory": false, "write_memory": false, "rewritten_query": "q"}'

    return MemoryAgent(
        router=Router(complete_fn=router_complete),
        store=store,
        extractor=MemoryExtractor(complete_fn=const("[]")),
        merger=MemoryMerger(store=store, complete_fn=const('{"operations": []}')),
        retrieve_docs_fn=lambda q: [],
        generate_fn=lambda q, c, m: {"answer": "ANSWER-ABOUT-GUITAR", "sources": []},
        history_max_turns=history_max_turns,
    )


def test_router_sees_prior_turn_on_later_turns(tmp_path):
    prompts = []
    agent = _history_agent(tmp_path, prompts)

    agent.chat("我在学吉他")     # turn 1: no history
    agent.chat("那要练多久")     # turn 2: should carry turn-1 context

    assert prompts[0] == "我在学吉他"                  # first turn routes on the msg alone
    assert "我在学吉他" in prompts[1]                  # second turn sees prior user message
    assert "ANSWER-ABOUT-GUITAR" in prompts[1]         # ... and prior assistant answer


def test_history_is_bounded(tmp_path):
    prompts = []
    agent = _history_agent(tmp_path, prompts, history_max_turns=2)
    for i in range(5):
        agent.chat(f"msg{i}")
    assert len(agent.history) <= 2
