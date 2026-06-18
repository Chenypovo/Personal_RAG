from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def doc_store_exists(
    vector_store: str = "faiss",
    *,
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    lancedb_uri: Optional[str] = None,
) -> bool:
    """Whether a built document vector index exists for the given backend."""
    if (vector_store or "faiss").lower() == "lancedb":
        return bool(lancedb_uri) and Path(lancedb_uri).exists()
    return bool(index_path and meta_path) and Path(index_path).exists() and Path(meta_path).exists()


def load_doc_store(
    vector_store: str = "faiss",
    *,
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    lancedb_uri: Optional[str] = None,
    lancedb_table: str = "chunks",
) -> Any:
    """Load the document vector store for the chosen backend.

    Both backends expose the same ``search()`` / ``all_metadatas()`` interface,
    so callers (retrieval, eval, agent) are backend-agnostic.
    """
    if (vector_store or "faiss").lower() == "lancedb":
        from app.vectordb.lancedb_store import LanceDBStore

        return LanceDBStore.load(uri=lancedb_uri, table_name=lancedb_table)

    from app.vectordb.faiss_store import FaissStore

    return FaissStore.load(index_path=index_path, meta_path=meta_path)
