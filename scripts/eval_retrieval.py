import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.reranker.bge_reranker import BGEReranker  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402
from app.vectordb.elasticsearch_store import ElasticsearchStore  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


def _normalize_doc_id(doc_id: str) -> str:
    d = (doc_id or "").strip()
    if not d:
        return d
    d = d.replace("\\", "/")
    while "//" in d:
        d = d.replace("//", "/")
    return d


def _as_vector(v: Any) -> List[float]:
    if hasattr(v, "tolist"):
        v = v.tolist()
    if not isinstance(v, list):
        raise TypeError(f"Expected list-like vector, got {type(v).__name__}")
    return [float(x) for x in v]


def _doc_key(meta: Dict[str, Any], fallback_idx: int) -> str:
    doc_id = meta.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return _normalize_doc_id(doc_id)
    source = str(meta.get("source", "unknown"))
    chunk_id = meta.get("chunk_id", fallback_idx)
    return _normalize_doc_id(f"{source}#{chunk_id}")


def _load_vector_store(args: argparse.Namespace) -> Any:
    if args.vector_backend == "elasticsearch":
        return ElasticsearchStore.connect(
            url=args.es_url,
            index_name=args.es_index,
            username=args.es_username,
            password=args.es_password,
            api_key=args.es_api_key,
            verify_certs=not args.es_no_verify_certs,
            request_timeout=int(args.es_request_timeout),
        )
    return FaissStore.load(args.index_path, args.meta_path)


def _parse_ks(ks: str) -> List[int]:
    out: List[int] = []
    for x in ks.split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k <= 0:
            raise ValueError(f"Invalid K: {k}")
        out.append(k)
    if not out:
        raise ValueError("No valid K provided in --ks")
    return sorted(set(out))


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid jsonl at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be an object: {path}:{line_no}")
            rows.append(obj)
    return rows


@dataclass
class QueryItem:
    query_id: str
    query: str
    query_image: str


def _load_queries(path: str) -> List[QueryItem]:
    rows = _load_jsonl(path)
    out: List[QueryItem] = []
    for i, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"queries line {i}: missing query_id")

        query = str(row.get("query", "")).strip()
        query_image = str(row.get("query_image", "")).strip()
        if not query and not query_image:
            raise ValueError(f"queries line {i}: both query and query_image are empty")

        out.append(QueryItem(query_id=query_id, query=query, query_image=query_image))
    return out


def _qrel_to_doc_id(row: Dict[str, Any], i: int) -> str:
    doc_id = str(row.get("doc_id", "")).strip()
    if doc_id:
        return _normalize_doc_id(doc_id)

    source = str(row.get("source", "")).strip()
    if not source:
        raise ValueError(f"qrels line {i}: missing doc_id or source")
    chunk_id = row.get("chunk_id", 0)
    return _normalize_doc_id(f"{source}#{chunk_id}")


def _load_qrels(path: str) -> Dict[str, Set[str]]:
    rows = _load_jsonl(path)
    out: Dict[str, Set[str]] = {}
    for i, row in enumerate(rows, start=1):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"qrels line {i}: missing query_id")

        relevance = row.get("relevance", 1)
        try:
            rel = float(relevance)
        except Exception as e:
            raise ValueError(f"qrels line {i}: invalid relevance={relevance}") from e
        if rel <= 0:
            continue

        doc_id = _qrel_to_doc_id(row, i)
        out.setdefault(query_id, set()).add(doc_id)
    return out


def _recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = len(set(pred[:k]).intersection(gold))
    return hit / float(len(gold))


def _mrr_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    for rank, doc_id in enumerate(pred[:k], start=1):
        if doc_id in gold:
            return 1.0 / float(rank)
    return 0.0


def _mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    return sum(vals) / float(len(vals)) if vals else 0.0


def _retrieve_vector(
    *,
    store: Any,
    vector_backend: str,
    embed_backend: str,
    embedder_openai: Optional[OpenAICompatibleEmbedder],
    embedder_clip: Optional[CLIPMultimodalEmbedder],
    query: str,
    query_image: str,
    top_k: int,
    output_k: Optional[int],
    candidate_k: int,
    max_distance: float,
    strict: bool,
    es_num_candidates: int,
) -> List[Dict[str, Any]]:
    if embed_backend == "openai":
        if not query:
            return []
        if embedder_openai is None:
            raise RuntimeError("OpenAI embedder is not initialized")
        qv = _as_vector(embedder_openai.embed_text(query, output_type="list"))
    else:
        if embedder_clip is None:
            raise RuntimeError("CLIP embedder is not initialized")
        if query_image:
            qv = _as_vector(embedder_clip.embed_image(query_image, output_type="list"))
        elif query:
            qv = _as_vector(embedder_clip.embed_text(query, output_type="list"))
        else:
            return []

    recall_k = max(top_k, candidate_k)
    if vector_backend == "elasticsearch":
        candidates = store.search(
            query_vector=qv,
            top_k=recall_k,
            num_candidates=max(int(es_num_candidates), recall_k),
        )
    else:
        candidates = store.search(query_vector=qv, top_k=recall_k)

    filtered: List[Dict[str, Any]] = []
    if vector_backend == "faiss":
        for item in candidates:
            d = item.get("distance")
            if isinstance(d, (int, float)) and d <= max_distance:
                filtered.append(item)
    else:
        filtered = candidates[:]

    limit = output_k if output_k is not None else top_k
    if filtered:
        return filtered[:limit]
    if strict:
        return []
    return candidates[:limit]


def _retrieve_bm25(
    *,
    bm25: BM25Store,
    query: str,
    top_k: int,
    output_k: Optional[int],
) -> List[Dict[str, Any]]:
    if not query:
        return []
    limit = output_k if output_k is not None else top_k
    return bm25.search(query=query, top_k=limit)


def _retrieve_hybrid(
    *,
    store: Any,
    vector_backend: str,
    bm25: BM25Store,
    embed_backend: str,
    embedder_openai: Optional[OpenAICompatibleEmbedder],
    embedder_clip: Optional[CLIPMultimodalEmbedder],
    query: str,
    query_image: str,
    top_k: int,
    output_k: Optional[int],
    candidate_k: int,
    max_distance: float,
    strict: bool,
    vector_k: int,
    bm25_k: int,
    rrf_k: int,
    es_num_candidates: int,
) -> List[Dict[str, Any]]:
    if query_image and not query:
        # image-only query fallback to vector
        return _retrieve_vector(
            store=store,
            vector_backend=vector_backend,
            embed_backend=embed_backend,
            embedder_openai=embedder_openai,
            embedder_clip=embedder_clip,
            query=query,
            query_image=query_image,
            top_k=top_k,
            output_k=output_k,
            candidate_k=candidate_k,
            max_distance=max_distance,
            strict=strict,
            es_num_candidates=es_num_candidates,
        )

    if not query:
        return []

    vec_items = _retrieve_vector(
        store=store,
        vector_backend=vector_backend,
        embed_backend=embed_backend,
        embedder_openai=embedder_openai,
        embedder_clip=embedder_clip,
        query=query,
        query_image="",
        top_k=max(top_k, vector_k),
        output_k=max(output_k or top_k, vector_k),
        candidate_k=max(candidate_k, vector_k),
        max_distance=max_distance,
        strict=strict,
        es_num_candidates=es_num_candidates,
    )
    bm25_items = _retrieve_bm25(
        bm25=bm25,
        query=query,
        top_k=max(top_k, bm25_k),
        output_k=max(output_k or top_k, bm25_k),
    )

    fused_scores: Dict[str, float] = {}
    best_item: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(vec_items, start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / float(rrf_k + rank)
        best_item[key] = item

    for rank, item in enumerate(bm25_items, start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / float(rrf_k + rank)
        if key not in best_item:
            best_item[key] = item
        else:
            merged = dict(best_item[key])
            if "score" not in merged and isinstance(item.get("score"), (int, float)):
                merged["score"] = item["score"]
            best_item[key] = merged

    ranked = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    out: List[Dict[str, Any]] = []
    limit = output_k if output_k is not None else top_k
    for key in ranked[:limit]:
        item = dict(best_item[key])
        item["hybrid_score"] = float(fused_scores[key])
        out.append(item)
    return out


def _rerank_items(
    *,
    items: List[Dict[str, Any]],
    query: str,
    use_rerank: bool,
    reranker: Optional[BGEReranker],
    final_top_k: int,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    if not use_rerank or reranker is None:
        return items[:final_top_k]
    if not query:
        return items[:final_top_k]
    reranked = reranker.rerank(query=query, retrieved=items, top_k=final_top_k)
    return reranked[:final_top_k]


def _extract_doc_ids(items: List[Dict[str, Any]]) -> List[str]:
    doc_ids: List[str] = []
    for i, item in enumerate(items, start=1):
        meta_raw = item.get("metadata")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        doc_ids.append(_doc_key(meta, i))
    return doc_ids


def _evaluate_backend(
    *,
    backend: str,
    queries: List[QueryItem],
    qrels: Dict[str, Set[str]],
    ks: List[int],
    store: Optional[Any],
    bm25: Optional[BM25Store],
    vector_backend: str,
    embed_backend: str,
    embedder_openai: Optional[OpenAICompatibleEmbedder],
    embedder_clip: Optional[CLIPMultimodalEmbedder],
    top_k_max: int,
    candidate_k: int,
    max_distance: float,
    strict: bool,
    vector_k: int,
    bm25_k: int,
    rrf_k: int,
    es_num_candidates: int,
    use_rerank: bool,
    reranker: Optional[BGEReranker],
    rerank_top_n: int,
) -> Dict[str, Any]:
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: Dict[int, List[float]] = {k: [] for k in ks}
    per_query: List[Dict[str, Any]] = []
    skipped_no_qrels = 0

    for q in queries:
        gold = qrels.get(q.query_id, set())
        if not gold:
            skipped_no_qrels += 1
            continue

        if backend == "vector":
            if store is None:
                raise RuntimeError("Vector store not initialized")
            items = _retrieve_vector(
                store=store,
                vector_backend=vector_backend,
                embed_backend=embed_backend,
                embedder_openai=embedder_openai,
                embedder_clip=embedder_clip,
                query=q.query,
                query_image=q.query_image,
                top_k=top_k_max,
                output_k=max(top_k_max, rerank_top_n) if use_rerank else top_k_max,
                candidate_k=candidate_k,
                max_distance=max_distance,
                strict=strict,
                es_num_candidates=es_num_candidates,
            )
        elif backend == "bm25":
            if bm25 is None:
                raise RuntimeError("BM25 store not initialized")
            items = _retrieve_bm25(
                bm25=bm25,
                query=q.query,
                top_k=top_k_max,
                output_k=max(top_k_max, rerank_top_n) if use_rerank else top_k_max,
            )
        elif backend == "hybrid":
            if store is None or bm25 is None:
                raise RuntimeError("Vector/BM25 store not initialized")
            items = _retrieve_hybrid(
                store=store,
                vector_backend=vector_backend,
                bm25=bm25,
                embed_backend=embed_backend,
                embedder_openai=embedder_openai,
                embedder_clip=embedder_clip,
                query=q.query,
                query_image=q.query_image,
                top_k=top_k_max,
                output_k=max(top_k_max, rerank_top_n) if use_rerank else top_k_max,
                candidate_k=candidate_k,
                max_distance=max_distance,
                strict=strict,
                vector_k=vector_k,
                bm25_k=bm25_k,
                rrf_k=rrf_k,
                es_num_candidates=es_num_candidates,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        items = _rerank_items(
            items=items,
            query=q.query,
            use_rerank=use_rerank,
            reranker=reranker,
            final_top_k=top_k_max,
        )

        pred = _extract_doc_ids(items)
        row: Dict[str, Any] = {"query_id": q.query_id, "pred_doc_ids": pred}
        for k in ks:
            r = _recall_at_k(pred, gold, k)
            m = _mrr_at_k(pred, gold, k)
            recalls[k].append(r)
            mrrs[k].append(m)
            row[f"recall@{k}"] = r
            row[f"mrr@{k}"] = m
        per_query.append(row)

    summary = {
        "backend": backend,
        "use_rerank": use_rerank,
        "n_queries": len(queries),
        "n_eval_queries": len(per_query),
        "n_skipped_no_qrels": skipped_no_qrels,
        "metrics": {},
    }
    for k in ks:
        summary["metrics"][f"recall@{k}"] = _mean(recalls[k])
        summary["metrics"][f"mrr@{k}"] = _mean(mrrs[k])
    summary["per_query"] = per_query
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with Recall@K and MRR@K")
    parser.add_argument("--queries", type=str, required=True, help="queries jsonl path")
    parser.add_argument("--qrels", type=str, required=True, help="qrels jsonl path")
    parser.add_argument("--backend", type=str, default="all", choices=["vector", "bm25", "hybrid", "all"])
    parser.add_argument("--ks", type=str, default="1,4,10", help="comma-separated K values")

    parser.add_argument(
        "--vector-backend",
        type=str,
        default="faiss",
        choices=["faiss", "elasticsearch"],
        help="Vector store backend used by vector/hybrid backends",
    )
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--es-url", type=str, default="http://localhost:9200")
    parser.add_argument("--es-index", type=str, default="personal_rag")
    parser.add_argument("--es-username", type=str, default="")
    parser.add_argument("--es-password", type=str, default="")
    parser.add_argument("--es-api-key", type=str, default="")
    parser.add_argument("--es-request-timeout", type=int, default=30)
    parser.add_argument("--es-num-candidates", type=int, default=100, help="ES knn num_candidates")
    parser.add_argument("--es-no-verify-certs", action="store_true", help="Disable TLS cert verification for ES client")
    parser.add_argument("--bm25-path", type=str, default="data/index/bm25.json")

    parser.add_argument("--embed-backend", type=str, default="openai", choices=["openai", "clip"])
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", type=str, default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)

    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--max-distance", type=float, default=0.8)
    parser.add_argument("--strict", dest="strict", action="store_true", default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")

    parser.add_argument("--vector-k", type=int, default=40, help="hybrid vector branch recall size")
    parser.add_argument("--bm25-k", type=int, default=40, help="hybrid bm25 branch recall size")
    parser.add_argument("--rrf-k", type=int, default=60, help="hybrid RRF constant")
    parser.add_argument("--jieba-user-dict", action="append", default=[])
    parser.add_argument("--use-rerank", action="store_true", help="Apply BGE reranker after retrieval")
    parser.add_argument("--rerank-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-device", type=str, default="", help='e.g. "cpu" or "cuda"')
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-top-n", type=int, default=40, help="Candidate size before rerank")
    parser.add_argument("--compare-rerank", action="store_true", help="Run both without and with rerank")

    parser.add_argument("--output-json", type=str, default="", help="optional output json report path")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    top_k_max = max(ks)
    queries = _load_queries(args.queries)
    qrels = _load_qrels(args.qrels)

    backends = ["vector", "bm25", "hybrid"] if args.backend == "all" else [args.backend]

    need_vector = any(b in ("vector", "hybrid") for b in backends)
    need_bm25 = any(b in ("bm25", "hybrid") for b in backends)
    need_embed = any(b in ("vector", "hybrid") for b in backends)

    store = _load_vector_store(args) if need_vector else None
    bm25 = BM25Store.load(args.bm25_path, user_dict_paths=args.jieba_user_dict) if need_bm25 else None

    embedder_openai = None
    embedder_clip = None
    if need_embed:
        if args.embed_backend == "openai":
            embedder_openai = OpenAICompatibleEmbedder()
        else:
            embedder_clip = CLIPMultimodalEmbedder(
                model_name=args.clip_model_name,
                device=args.clip_device or None,
                batch_size=args.clip_batch_size,
            )

    rerank_modes = [False, True] if args.compare_rerank else [bool(args.use_rerank)]
    reports: List[Dict[str, Any]] = []
    for rerank_on in rerank_modes:
        reranker: Optional[BGEReranker] = None
        if rerank_on:
            reranker = BGEReranker(
                model_name=args.rerank_model,
                device=args.rerank_device or None,
                batch_size=args.rerank_batch_size,
            )
        for backend in backends:
            report = _evaluate_backend(
                backend=backend,
                queries=queries,
                qrels=qrels,
                ks=ks,
                store=store,
                bm25=bm25,
                vector_backend=args.vector_backend,
                embed_backend=args.embed_backend,
                embedder_openai=embedder_openai,
                embedder_clip=embedder_clip,
                top_k_max=top_k_max,
                candidate_k=args.candidate_k,
                max_distance=args.max_distance,
                strict=args.strict,
                vector_k=args.vector_k,
                bm25_k=args.bm25_k,
                rrf_k=args.rrf_k,
                es_num_candidates=args.es_num_candidates,
                use_rerank=rerank_on,
                reranker=reranker,
                rerank_top_n=args.rerank_top_n,
            )
            reports.append(report)

    print("\n=== Retrieval Eval ===")
    print(f"queries={args.queries}")
    print(f"qrels={args.qrels}")
    print(f"ks={ks}")
    print(f"embed_backend={args.embed_backend}")
    print(f"vector_backend={args.vector_backend}")
    print("")
    for rep in reports:
        print(
            f"[backend={rep['backend']}, rerank={rep['use_rerank']}] "
            f"n_eval_queries={rep['n_eval_queries']} / {rep['n_queries']}"
        )
        metrics = rep["metrics"]
        for k in ks:
            print(
                f"  recall@{k}={metrics[f'recall@{k}']:.4f} "
                f"mrr@{k}={metrics[f'mrr@{k}']:.4f}"
            )
        print("")

    if args.output_json:
        payload = {
            "queries": args.queries,
            "qrels": args.qrels,
            "ks": ks,
            "backend": args.backend,
            "vector_backend": args.vector_backend,
            "compare_rerank": bool(args.compare_rerank),
            "reports": reports,
        }
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
