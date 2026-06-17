import json
from pathlib import Path
from typing import Any, Dict, List


class LanceDBStore:
    """Local LanceDB-backed vector store with the same search shape as FaissStore."""

    def __init__(self, uri: str, table_name: str = "chunks") -> None:
        self.uri = uri
        self.table_name = table_name
        self._db = None
        self._table = None

    @staticmethod
    def _connect(uri: str):
        try:
            import lancedb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "LanceDB is not installed. Install requirements.txt or choose --vector-store faiss."
            ) from e

        Path(uri).mkdir(parents=True, exist_ok=True)
        return lancedb.connect(uri)

    @staticmethod
    def _row_id(meta: Dict[str, Any], idx: int) -> str:
        doc_id = meta.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id.strip()
        chunk_uid = meta.get("chunk_uid")
        if isinstance(chunk_uid, str) and chunk_uid.strip():
            return chunk_uid.strip()
        source = str(meta.get("source", "unknown"))
        chunk_id = meta.get("chunk_id", idx)
        return f"{source}#{chunk_id}"

    @staticmethod
    def _to_row(vector: List[float], metadata: Dict[str, Any], idx: int) -> Dict[str, Any]:
        return {
            "id": LanceDBStore._row_id(metadata, idx),
            "vector": [float(x) for x in vector],
            "text": str(metadata.get("text", "")),
            "source": str(metadata.get("source", "")),
            "metadata_json": json.dumps(metadata, ensure_ascii=False),
        }

    @staticmethod
    def _metadata_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
        raw = row.get("metadata_json")
        if isinstance(raw, str) and raw.strip():
            try:
                meta = json.loads(raw)
                if isinstance(meta, dict):
                    return meta
            except json.JSONDecodeError:
                pass
        return {
            "source": row.get("source", ""),
            "text": row.get("text", ""),
        }

    def add(self, vectors: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        if not vectors:
            return

        rows = [self._to_row(v, m, i) for i, (v, m) in enumerate(zip(vectors, metadatas))]
        db = self._connect(self.uri)
        self._db = db
        self._table = db.create_table(self.table_name, data=rows, mode="overwrite")

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if self._table is None:
            db = self._connect(self.uri)
            self._db = db
            self._table = db.open_table(self.table_name)

        rows = self._table.search([float(x) for x in query_vector]).limit(top_k).to_list()
        results: List[Dict[str, Any]] = []
        for row in rows:
            distance = row.get("_distance", row.get("_score", 0.0))
            results.append(
                {
                    "distance": float(distance),
                    "metadata": self._metadata_from_row(row),
                }
            )
        return results

    @classmethod
    def load(cls, uri: str, table_name: str = "chunks") -> "LanceDBStore":
        store = cls(uri=uri, table_name=table_name)
        db = store._connect(uri)
        store._db = db
        store._table = db.open_table(table_name)
        return store
