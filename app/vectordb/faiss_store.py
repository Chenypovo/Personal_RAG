import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[Dict[str, Any]] = []

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        if not vectors:
            return

        arr = np.array(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch, expected {self.dim}, got {arr.shape}")

        self.index.add(arr)
        self.metadatas.extend(metadatas)

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q = np.array([query_vector], dtype="float32")
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dimension mismatch, expected {self.dim}, got {q.shape[1]}")

        distances, indices = self.index.search(q, top_k)
        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadatas[idx]
            results.append(
                {
                    "distance": float(dist),
                    "metadata": meta,
                }
            )
        return results

    def save(self, index_path: str, meta_path: str) -> None:
        index_file = Path(index_path)
        meta_file = Path(meta_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        meta_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        meta_file.write_text(
            json.dumps(self.metadatas, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FaissStore":
        index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadatas = json.load(f)

        store = cls(dim=index.d)
        store.index = index
        store.metadatas = metadatas
        return store
