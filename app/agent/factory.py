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
    vector_store: str,
    index_path: str,
    meta_path: str,
    lancedb_uri: str,
    lancedb_table: str,
    embed_fn,
    top_k: int,
    max_distance: float,
    use_rerank: bool,
    rerank_model: str,
    rerank_candidates: int,
    use_parent_child: bool = True,
):
    """Vector document retrieval (optionally reranked) with graceful no-docs fallback."""
    from app.vectordb import doc_store_exists, load_doc_store

    if not doc_store_exists(
        vector_store, index_path=index_path, meta_path=meta_path, lancedb_uri=lancedb_uri
    ):
        return lambda query: []

    store = load_doc_store(
        vector_store,
        index_path=index_path,
        meta_path=meta_path,
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
    )
    all_chunks = store.all_metadatas() if use_parent_child else []

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
            items = reranker.rerank(query=query, retrieved=items, top_k=top_k)
        else:
            items = items[:top_k]
        if use_parent_child:
            # small-to-big: expand the precise child hits to their parent section
            from app.retriever.parent_child import expand_to_parents

            items = expand_to_parents(items, all_chunks)
        return items

    return retrieve


def build_memory_agent(
    memory_dir: str = "data/memory",
    vector_store: str = "lancedb",
    index_path: str = "data/index/faiss.index",
    meta_path: str = "data/index/metadatas.json",
    lancedb_uri: str = "data/index/lancedb",
    lancedb_table: str = "chunks",
    top_k: int = 4,
    max_distance: float = 0.8,
    recall_k: int = 5,
    use_rerank: bool = False,
    rerank_model: str = "BAAI/bge-reranker-base",
    rerank_candidates: int = 20,
    use_parent_child: bool = True,
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
        vector_store, index_path, meta_path, lancedb_uri, lancedb_table, embed_fn,
        top_k, max_distance, use_rerank, rerank_model, rerank_candidates, use_parent_child,
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
