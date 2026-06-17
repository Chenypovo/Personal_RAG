from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Tuple

from app.memory.models import MemoryFact, now_iso


EmbedFn = Callable[[str], List[float]]


class VectorIndex(Protocol):
    def add(self, item_id: str, vector: List[float]) -> None: ...
    def remove(self, item_id: str) -> None: ...
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]: ...
    def clear(self) -> None: ...


_COLUMNS = (
    "id",
    "fact_object",
    "fact_content",
    "visibility",
    "state",
    "source",
    "created_at",
    "updated_at",
)


class MemoryStore:
    """SQLite truth store for memory facts, with a pluggable vector index.

    The SQLite table is the single source of truth; the vector index is a
    rebuildable semantic search structure kept in sync on every write.
    The embedding function and vector index are injected for testability.
    """

    def __init__(self, db_path: str, vector_index: VectorIndex, embed_fn: EmbedFn) -> None:
        self.db_path = db_path
        self.vector_index = vector_index
        self.embed_fn = embed_fn

        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id TEXT PRIMARY KEY,
                fact_object TEXT NOT NULL DEFAULT '',
                fact_content TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'PUBLIC',
                state TEXT NOT NULL DEFAULT 'ACTIVE',
                source TEXT NOT NULL DEFAULT 'chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> MemoryFact:
        return MemoryFact(
            id=row["id"],
            fact_object=row["fact_object"],
            fact_content=row["fact_content"],
            visibility=row["visibility"],
            state=row["state"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def add(self, fact: MemoryFact) -> MemoryFact:
        self._conn.execute(
            f"INSERT INTO memory_facts ({', '.join(_COLUMNS)}) VALUES ({', '.join(['?'] * len(_COLUMNS))})",
            (
                fact.id,
                fact.fact_object,
                fact.fact_content,
                fact.visibility,
                fact.state,
                fact.source,
                fact.created_at,
                fact.updated_at,
            ),
        )
        self._conn.commit()
        if fact.state == "ACTIVE":
            self.vector_index.add(fact.id, self.embed_fn(fact.fact_content))
        return fact

    def get(self, fact_id: str) -> Optional[MemoryFact]:
        cur = self._conn.execute("SELECT * FROM memory_facts WHERE id = ?", (fact_id,))
        row = cur.fetchone()
        return self._row_to_fact(row) if row else None

    def update(
        self,
        fact_id: str,
        *,
        fact_content: Optional[str] = None,
        fact_object: Optional[str] = None,
        visibility: Optional[str] = None,
    ) -> Optional[MemoryFact]:
        current = self.get(fact_id)
        if current is None:
            return None

        new_content = fact_content if fact_content is not None else current.fact_content
        new_object = fact_object if fact_object is not None else current.fact_object
        new_visibility = visibility if visibility is not None else current.visibility
        updated_at = now_iso()

        self._conn.execute(
            "UPDATE memory_facts SET fact_content = ?, fact_object = ?, visibility = ?, updated_at = ? WHERE id = ?",
            (new_content, new_object, new_visibility, updated_at, fact_id),
        )
        self._conn.commit()

        if current.state == "ACTIVE" and fact_content is not None:
            self.vector_index.add(fact_id, self.embed_fn(new_content))

        return self.get(fact_id)

    def soft_delete(self, fact_id: str) -> bool:
        current = self.get(fact_id)
        if current is None:
            return False
        self._conn.execute(
            "UPDATE memory_facts SET state = 'DELETED', updated_at = ? WHERE id = ?",
            (now_iso(), fact_id),
        )
        self._conn.commit()
        self.vector_index.remove(fact_id)
        return True

    def list_active(self) -> List[MemoryFact]:
        cur = self._conn.execute(
            "SELECT * FROM memory_facts WHERE state = 'ACTIVE' ORDER BY created_at"
        )
        return [self._row_to_fact(r) for r in cur.fetchall()]

    def list_all(self) -> List[MemoryFact]:
        cur = self._conn.execute("SELECT * FROM memory_facts ORDER BY created_at")
        return [self._row_to_fact(r) for r in cur.fetchall()]

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Tuple[MemoryFact, float]]:
        hits = self.vector_index.search(self.embed_fn(query), top_k=top_k)
        results: List[Tuple[MemoryFact, float]] = []
        for fact_id, score in hits:
            if score <= min_score:
                continue  # non-positive similarity = unrelated
            fact = self.get(fact_id)
            if fact is not None and fact.state == "ACTIVE":
                results.append((fact, score))
        return results

    def rebuild_index(self) -> int:
        self.vector_index.clear()
        count = 0
        for fact in self.list_active():
            self.vector_index.add(fact.id, self.embed_fn(fact.fact_content))
            count += 1
        return count

    def close(self) -> None:
        self._conn.close()
