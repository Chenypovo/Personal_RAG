from typing import Any, Dict, List

from app.embedder.embedder import OpenAICompatibleEmbedder
from app.vectordb.faiss_store import FaissStore


class FaissRetriever:
    def __init__(self, store: FaissStore, embedder: OpenAICompatibleEmbedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        
        q_vec = self.embedder.embed_text(q)
        results = self.store.search(query_vector = q_vec, top_k = top_k)
        return results
