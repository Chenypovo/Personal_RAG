"""Interactive multi-turn chat that demonstrates the memory-augmented agent.

Each turn the agent:
  1. routes (decide use_docs / use_memory / write_memory)
  2. recalls relevant long-term memories
  3. retrieves document evidence (FAISS vector search, if an index exists)
  4. generates an evidence-constrained answer personalized with memory
  5. extracts new facts from your message and merges them into long-term memory

Run:
    python scripts/chat_demo.py
Type 'exit' / 'quit' to leave, ':mem' to list stored memories.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.agent import MemoryAgent  # noqa: E402
from app.agent.llm import make_complete_fn, make_embed_fn  # noqa: E402
from app.agent.router import Router  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.memory.extractor import MemoryExtractor  # noqa: E402
from app.memory.merger import MemoryMerger  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.memory.vector_index import NumpyVectorIndex  # noqa: E402


def build_doc_retriever(index_path: str, meta_path: str, embed_fn, top_k: int, max_distance: float):
    """Vector document retrieval with graceful fallback to 'no documents'."""
    if not (Path(index_path).exists() and Path(meta_path).exists()):
        print(f"[docs] no index at {index_path}; running memory-only (no document evidence).")
        return lambda query: []

    from app.vectordb.faiss_store import FaissStore

    store = FaissStore.load(index_path=index_path, meta_path=meta_path)

    def retrieve(query: str) -> List[Dict[str, Any]]:
        vec = embed_fn(query)
        candidates = store.search(query_vector=vec, top_k=max(top_k, 20))
        kept = [c for c in candidates if isinstance(c.get("distance"), (int, float)) and c["distance"] <= max_distance]
        return (kept or candidates)[:top_k]

    return retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-augmented RAG agent (interactive)")
    parser.add_argument("--memory-dir", default="data/memory")
    parser.add_argument("--index-path", default="data/index/faiss.index")
    parser.add_argument("--meta-path", default="data/index/metadatas.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-distance", type=float, default=0.8)
    parser.add_argument("--recall-k", type=int, default=5)
    args = parser.parse_args()

    embed_fn = make_embed_fn()
    complete_fn = make_complete_fn()

    # determine embedding dim from a probe
    dim = len(embed_fn("dimension probe"))

    mem_dir = Path(args.memory_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(mem_dir / "memory.db")
    index_json = mem_dir / "mem_index.json"

    if index_json.exists():
        vector_index = NumpyVectorIndex.load(str(index_json))
    else:
        vector_index = NumpyVectorIndex(dim=dim)

    store = MemoryStore(db_path=db_path, vector_index=vector_index, embed_fn=embed_fn)
    if len(vector_index) == 0 and store.list_active():
        rebuilt = store.rebuild_index()
        print(f"[memory] rebuilt vector index from {rebuilt} stored facts.")

    generator = OpenAICompatibleGenerator()

    def generate_fn(query: str, chunks: List[Dict[str, Any]], user_memory: str) -> Dict[str, Any]:
        if not chunks and not user_memory.strip():
            return {"answer": "No documents indexed and nothing remembered yet. Tell me about yourself or build an index.", "sources": []}
        return generator.generate(query=query, retrieved_chunks=chunks, user_memory=user_memory)

    agent = MemoryAgent(
        router=Router(complete_fn=complete_fn),
        store=store,
        extractor=MemoryExtractor(complete_fn=complete_fn),
        merger=MemoryMerger(store=store, complete_fn=complete_fn),
        retrieve_docs_fn=build_doc_retriever(args.index_path, args.meta_path, embed_fn, args.top_k, args.max_distance),
        generate_fn=generate_fn,
        recall_k=args.recall_k,
    )

    print("=" * 70)
    print("Memory-augmented RAG agent. Type 'exit' to quit, ':mem' to list memories.")
    print("=" * 70)

    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break
        if user_msg == ":mem":
            facts = store.list_active()
            print(f"\n[stored memories: {len(facts)}]")
            for f in facts:
                obj = f"({f.fact_object}) " if f.fact_object else ""
                print(f"  - {obj}{f.fact_content}  [{f.visibility}, {f.source}]")
            continue

        result = agent.chat(user_msg)

        print(f"\nAssistant: {result.answer}")

        if result.recalled_memories:
            print("\n  ↳ recalled about you:")
            for m in result.recalled_memories:
                print(f"      • {m.fact_content}")
        if result.sources:
            cites = ", ".join(str(s.get("citation")) for s in result.sources)
            print(f"  ↳ sources: {cites}")
        if result.memory_ops:
            applied = [f"{o.type}:{o.fact_content or o.id}" for o in result.memory_ops if o.applied]
            if applied:
                print(f"  ↳ memory updated: {', '.join(applied)}")

        # persist memory vector index after each turn (SQLite persists itself)
        vector_index.save(str(index_json))

    vector_index.save(str(index_json))
    print("Memory saved. Bye.")


if __name__ == "__main__":
    main()
