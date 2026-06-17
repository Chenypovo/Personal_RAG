from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.agent.agent import MemoryAgent
from app.agent.llm import make_complete_fn, make_embed_fn
from app.agent.router import Router
from app.generator.generator import OpenAICompatibleGenerator
from app.memory.extractor import MemoryExtractor
from app.memory.merger import MemoryMerger
from app.memory.store import MemoryStore
from app.memory.vector_index import NumpyVectorIndex


@dataclass
class AgentBundle:
    agent: MemoryAgent
    store: MemoryStore
    vector_index: NumpyVectorIndex
    index_path: str

    def save(self) -> None:
        self.vector_index.save(self.index_path)


def _build_doc_retriever(
    index_path: str,
    meta_path: str,
    embed_fn,
    top_k: int,
    max_distance: float,
    use_rerank: bool,
    rerank_model: str,
    rerank_candidates: int,
):
    """Vector document retrieval (optionally reranked) with graceful no-docs fallback."""
    if not (Path(index_path).exists() and Path(meta_path).exists()):
        return lambda query: []

    from app.vectordb.faiss_store import FaissStore

    store = FaissStore.load(index_path=index_path, meta_path=meta_path)

    reranker = None
    if use_rerank:
        from app.reranker.bge_reranker import BGEReranker

        reranker = BGEReranker(model_name=rerank_model)

    def retrieve(query: str) -> List[Dict[str, Any]]:
        vec = embed_fn(query)
        pool = max(top_k, rerank_candidates) if use_rerank else max(top_k, 20)
        candidates = store.search(query_vector=vec, top_k=pool)
        kept = [c for c in candidates if isinstance(c.get("distance"), (int, float)) and c["distance"] <= max_distance]
        items = kept or candidates
        if reranker is not None:
            return reranker.rerank(query=query, retrieved=items, top_k=top_k)
        return items[:top_k]

    return retrieve


def build_memory_agent(
    memory_dir: str = "data/memory",
    index_path: str = "data/index/faiss.index",
    meta_path: str = "data/index/metadatas.json",
    top_k: int = 4,
    max_distance: float = 0.8,
    recall_k: int = 5,
    use_rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-base",
    rerank_candidates: int = 20,
) -> AgentBundle:
    """Wire a real MemoryAgent from configured models, with persistent memory."""
    embed_fn = make_embed_fn()
    complete_fn = make_complete_fn()
    dim = len(embed_fn("dimension probe"))

    mem_dir = Path(memory_dir)
    mem_dir.mkdir(parents=True, exist_ok=True)
    db_path = str(mem_dir / "memory.db")
    index_json = str(mem_dir / "mem_index.json")

    vector_index = NumpyVectorIndex.load(index_json) if Path(index_json).exists() else NumpyVectorIndex(dim=dim)
    store = MemoryStore(db_path=db_path, vector_index=vector_index, embed_fn=embed_fn)
    if len(vector_index) == 0 and store.list_active():
        store.rebuild_index()

    generator = OpenAICompatibleGenerator()

    def generate_fn(query: str, chunks: List[Dict[str, Any]], user_memory: str) -> Dict[str, Any]:
        if not chunks and not user_memory.strip():
            return {
                "answer": "还没有可用的文档证据或记忆。先告诉我一些关于你的事，或先建立文档索引。",
                "sources": [],
            }
        return generator.generate(query=query, retrieved_chunks=chunks, user_memory=user_memory)

    retrieve = _build_doc_retriever(
        index_path, meta_path, embed_fn, top_k, max_distance, use_rerank, rerank_model, rerank_candidates
    )

    agent = MemoryAgent(
        router=Router(complete_fn=complete_fn),
        store=store,
        extractor=MemoryExtractor(complete_fn=complete_fn),
        merger=MemoryMerger(store=store, complete_fn=complete_fn),
        retrieve_docs_fn=retrieve,
        generate_fn=generate_fn,
        recall_k=recall_k,
    )
    return AgentBundle(agent=agent, store=store, vector_index=vector_index, index_path=index_json)
