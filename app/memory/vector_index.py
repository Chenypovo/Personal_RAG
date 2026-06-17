from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class NumpyVectorIndex:
    """Mutable brute-force cosine index for personal-scale memory.

    Unlike the append-only FAISS document index, memories need update/delete,
    so this keeps id -> vector in memory and scores by cosine similarity.
    Scale is small (hundreds of facts), so brute force is more than enough.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim
        self._vectors: Dict[str, np.ndarray] = {}

    def add(self, item_id: str, vector: List[float]) -> None:
        v = np.asarray(vector, dtype="float32")
        if v.ndim != 1 or v.shape[0] != self.dim:
            raise ValueError(f"vector dim mismatch, expected {self.dim}, got {v.shape}")
        self._vectors[item_id] = v  # upsert

    def remove(self, item_id: str) -> None:
        self._vectors.pop(item_id, None)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not self._vectors:
            return []

        q = np.asarray(query_vector, dtype="float32")
        if q.ndim != 1 or q.shape[0] != self.dim:
            raise ValueError(f"query dim mismatch, expected {self.dim}, got {q.shape}")

        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        scored: List[Tuple[str, float]] = []
        for item_id, vec in self._vectors.items():
            v_norm = float(np.linalg.norm(vec))
            if v_norm == 0.0:
                continue
            sim = float(np.dot(q, vec) / (q_norm * v_norm))
            scored.append((item_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self._vectors.clear()

    def __len__(self) -> int:
        return len(self._vectors)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "vectors": {k: v.tolist() for k, v in self._vectors.items()},
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "NumpyVectorIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(dim=int(data["dim"]))
        for k, v in data.get("vectors", {}).items():
            obj._vectors[k] = np.asarray(v, dtype="float32")
        return obj
