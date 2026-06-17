from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

try:
    from elasticsearch import Elasticsearch, helpers  # type: ignore
except Exception:  # pragma: no cover
    Elasticsearch = None  # type: ignore[assignment]
    helpers = None  # type: ignore[assignment]


def _safe_int(x: Any, fallback: int) -> int:
    try:
        return int(x)
    except Exception:
        return fallback


class ElasticsearchStore:
    def __init__(
        self,
        *,
        client: Any,
        index_name: str,
        vector_field: str = "vector",
        request_timeout: int = 30,
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.vector_field = vector_field
        self.request_timeout = int(request_timeout)

    @classmethod
    def connect(
        cls,
        *,
        url: str,
        index_name: str,
        username: str = "",
        password: str = "",
        api_key: str = "",
        verify_certs: bool = True,
        request_timeout: int = 30,
        vector_field: str = "vector",
    ) -> "ElasticsearchStore":
        if Elasticsearch is None:
            raise ImportError(
                "elasticsearch package is not installed. "
                "Please run: pip install elasticsearch"
            )

        kwargs: Dict[str, Any] = {
            "hosts": [url],
            "verify_certs": bool(verify_certs),
            "request_timeout": int(request_timeout),
        }
        if api_key.strip():
            kwargs["api_key"] = api_key.strip()
        elif username.strip():
            kwargs["basic_auth"] = (username.strip(), password)

        client = Elasticsearch(**kwargs)
        return cls(
            client=client,
            index_name=index_name,
            vector_field=vector_field,
            request_timeout=request_timeout,
        )

    @staticmethod
    def _make_doc_id(meta: Dict[str, Any], fallback_idx: int) -> str:
        doc_id = meta.get("doc_id")
        if isinstance(doc_id, str) and doc_id.strip():
            return doc_id.strip()
        source = str(meta.get("source", "unknown"))
        chunk_id = meta.get("chunk_id", fallback_idx)
        raw = f"{source}#{chunk_id}"
        # Keep ES _id compact and deterministic even for long/odd source paths.
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()  # noqa: S324
        return f"doc-{digest}"

    def recreate_index(self, *, dim: int, recreate: bool = False) -> None:
        exists = bool(self.client.indices.exists(index=self.index_name))
        if exists and recreate:
            self.client.indices.delete(index=self.index_name)
            exists = False

        if exists:
            return

        body = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "modality": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "text": {"type": "text"},
                    self.vector_field: {
                        "type": "dense_vector",
                        "dims": int(dim),
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=body)

    def add(
        self,
        *,
        vectors: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
        batch_size: int = 200,
        refresh: bool = True,
    ) -> int:
        if len(vectors) != len(metadatas):
            raise ValueError("vectors and metadatas length mismatch")
        if not vectors:
            return 0
        if helpers is None:
            raise ImportError("elasticsearch.helpers is unavailable")

        batch_size = max(int(batch_size), 1)
        actions: List[Dict[str, Any]] = []
        for i, (vec, meta_raw) in enumerate(zip(vectors, metadatas)):
            meta = dict(meta_raw)
            doc_id_value = str(meta.get("doc_id", "")).strip()
            if not doc_id_value:
                source = str(meta.get("source", "unknown"))
                chunk_id = meta.get("chunk_id", i)
                doc_id_value = f"{source}#{chunk_id}"
                meta["doc_id"] = doc_id_value

            source = str(meta.get("source", "unknown"))
            chunk_id = _safe_int(meta.get("chunk_id", i), i)
            modality = str(meta.get("modality", "text"))
            file_type = str(meta.get("file_type", "unknown"))
            text = str(meta.get("text", ""))
            vector = [float(x) for x in vec]

            actions.append(
                {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_id": self._make_doc_id(meta, i),
                    "_source": {
                        "doc_id": doc_id_value,
                        "source": source,
                        "chunk_id": chunk_id,
                        "modality": modality,
                        "file_type": file_type,
                        "text": text,
                        self.vector_field: vector,
                        "metadata": meta,
                    },
                }
            )

        ok, _ = helpers.bulk(
            self.client,
            actions,
            chunk_size=batch_size,
            refresh="wait_for" if refresh else False,
            request_timeout=self.request_timeout,
        )
        return int(ok)

    def count(self) -> int:
        out = self.client.count(index=self.index_name)
        return int(out.get("count", 0))

    def search(
        self,
        *,
        query_vector: Sequence[float],
        top_k: int = 4,
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        top_k = max(int(top_k), 1)
        if num_candidates is None:
            num_candidates = max(top_k * 8, 100)
        num_candidates = max(int(num_candidates), top_k)

        resp = self.client.search(
            index=self.index_name,
            size=top_k,
            knn={
                "field": self.vector_field,
                "query_vector": [float(x) for x in query_vector],
                "k": top_k,
                "num_candidates": num_candidates,
            },
            source=["metadata", "doc_id", "source", "chunk_id", "modality", "file_type", "text"],
            request_timeout=self.request_timeout,
        )

        hits = resp.get("hits", {}).get("hits", [])
        out: List[Dict[str, Any]] = []
        for h in hits:
            src = h.get("_source", {}) if isinstance(h.get("_source"), dict) else {}
            meta = src.get("metadata")
            if not isinstance(meta, dict):
                meta = {
                    "doc_id": src.get("doc_id", ""),
                    "source": src.get("source", "unknown"),
                    "chunk_id": src.get("chunk_id", 0),
                    "modality": src.get("modality", "text"),
                    "file_type": src.get("file_type", "unknown"),
                    "text": src.get("text", ""),
                }
            score = float(h.get("_score", 0.0))
            out.append(
                {
                    "score": score,
                    # Keep compatibility with current downstream that expects a distance-like field.
                    "distance": float(1.0 - score),
                    "metadata": meta,
                }
            )
        return out
