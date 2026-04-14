import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


def _build_direct_context(query_text: str) -> List[Dict[str, Any]]:
    return [
        {
            "metadata": {
                "source": "direct_query",
                "file_type": "query",
                "modality": "text",
                "chunk_id": -1,
                "start": 0,
                "end": len(query_text),
                "text": query_text,
            }
        }
    ]


def _as_vector(v: Any) -> List[float]:
    if hasattr(v, "tolist"):
        v = v.tolist()
    if not isinstance(v, list):
        raise TypeError(f"Expected list-like vector, got {type(v).__name__}")
    return [float(x) for x in v]


def _doc_key(meta: Dict[str, Any], fallback_idx: int) -> str:
    doc_id = meta.get("doc_id")
    if isinstance(doc_id, str) and doc_id.strip():
        return doc_id.strip()
    source = str(meta.get("source", "unknown"))
    chunk_id = meta.get("chunk_id", fallback_idx)
    return f"{source}#{chunk_id}"


def _retrieve_vector(args: argparse.Namespace, query_text: str, query_image: str) -> Tuple[List[Dict[str, Any]], str]:
    if args.embed_backend == "openai" and not query_text:
        raise ValueError("openai backend requires --query")
    if args.embed_backend == "clip" and not query_text and not query_image:
        raise ValueError("clip backend requires --query or --query-image")

    store = FaissStore.load(index_path=args.index_path, meta_path=args.meta_path)

    if args.embed_backend == "openai":
        embedder = OpenAICompatibleEmbedder()
        query_vector = _as_vector(embedder.embed_text(query_text, output_type="list"))
    else:
        clip_embedder = CLIPMultimodalEmbedder(
            model_name=args.clip_model_name,
            device=args.clip_device or None,
            batch_size=args.clip_batch_size,
        )
        if query_image:
            query_vector = _as_vector(clip_embedder.embed_image(query_image, output_type="list"))
            if not query_text:
                query_text = f"Image query: {os.path.basename(query_image)}"
        else:
            query_vector = _as_vector(clip_embedder.embed_text(query_text, output_type="list"))

    candidates = store.search(query_vector=query_vector, top_k=max(args.top_k, args.candidate_k))
    filtered: List[Dict[str, Any]] = []
    for item in candidates:
        d = item.get("distance")
        if isinstance(d, (int, float)) and d <= args.max_distance:
            filtered.append(item)

    if filtered:
        return filtered[: args.top_k], query_text
    if args.strict:
        return [], query_text
    return candidates[: args.top_k], query_text


def _retrieve_bm25(args: argparse.Namespace, query_text: str, query_image: str) -> Tuple[List[Dict[str, Any]], str]:
    if query_image:
        raise ValueError("bm25 retrieval does not support --query-image; use --query text")
    if not query_text:
        raise ValueError("bm25 retrieval requires --query")

    bm25 = BM25Store.load(
        path=args.bm25_path,
        user_dict_paths=args.jieba_user_dict,
    )
    retrieved = bm25.search(query=query_text, top_k=args.top_k)
    return retrieved, query_text


def _retrieve_hybrid(args: argparse.Namespace, query_text: str, query_image: str) -> Tuple[List[Dict[str, Any]], str]:
    # Hybrid is text-first. If image-only query is given, fallback to vector retrieval.
    if query_image and not query_text:
        print("[hybrid] image-only query detected; fallback to vector retrieval.")
        return _retrieve_vector(args, query_text, query_image)
    if not query_text:
        raise ValueError("hybrid retrieval requires --query text")

    vector_top_k = max(args.vector_k, args.top_k)
    bm25_top_k = max(args.bm25_k, args.top_k)

    vec_args = argparse.Namespace(**vars(args))
    vec_args.top_k = vector_top_k
    vec_retrieved, query_text = _retrieve_vector(vec_args, query_text, "")

    bm25_args = argparse.Namespace(**vars(args))
    bm25_args.top_k = bm25_top_k
    bm25_retrieved, _ = _retrieve_bm25(bm25_args, query_text, "")

    # Reciprocal Rank Fusion (RRF).
    fused_scores: Dict[str, float] = {}
    best_item: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(vec_retrieved, start=1):
        meta_raw = item.get("metadata")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (args.rrf_k + rank)
        best_item[key] = item

    for rank, item in enumerate(bm25_retrieved, start=1):
        meta_raw = item.get("metadata")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (args.rrf_k + rank)
        if key not in best_item:
            best_item[key] = item
        else:
            # Keep vector distance when available, but merge bm25 score.
            merged = dict(best_item[key])
            if "score" not in merged and isinstance(item.get("score"), (int, float)):
                merged["score"] = item["score"]
            best_item[key] = merged

    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    fused: List[Dict[str, Any]] = []
    for k in ranked_keys[: args.top_k]:
        item = dict(best_item[k])
        item["hybrid_score"] = float(fused_scores[k])
        fused.append(item)

    return fused, query_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG query demo (CLI)")
    parser.add_argument("--query", type=str, default="", help="Text query")
    parser.add_argument("--query-image", type=str, default="", help="Image query path (for clip backend)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=40, help="Initial recall size before filtering (vector only)")
    parser.add_argument("--max-distance", type=float, default=0.8, help="Keep results with distance <= threshold (vector only)")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True, help="Strict mode (default: on)")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Backfill nearest items when none pass threshold")
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--bm25-path", type=str, default="data/index/bm25.json")
    parser.add_argument("--no-retrieval", action="store_true", help="Disable retrieval and do direct generation")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunk texts")

    parser.add_argument(
        "--retrieval-backend",
        type=str,
        default="vector",
        choices=["vector", "bm25", "hybrid"],
        help="vector: faiss retrieval; bm25: lexical retrieval; hybrid: vector + bm25 (RRF)",
    )
    parser.add_argument(
        "--embed-backend",
        type=str,
        default="openai",
        choices=["openai", "clip"],
        help="openai: text query only; clip: text/image query (vector retrieval only)",
    )
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", type=str, default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--vector-k", type=int, default=20, help="Hybrid: vector branch recall size")
    parser.add_argument("--bm25-k", type=int, default=20, help="Hybrid: bm25 branch recall size")
    parser.add_argument("--rrf-k", type=int, default=60, help="Hybrid: RRF constant (larger = flatter)")
    parser.add_argument(
        "--jieba-user-dict",
        action="append",
        default=[],
        help="Path to jieba custom dictionary for bm25 query tokenization. Can be used multiple times.",
    )

    args = parser.parse_args()

    query_text = (args.query or "").strip()
    query_image = (args.query_image or "").strip()

    if args.no_retrieval and not query_text:
        raise ValueError("--no-retrieval requires --query")

    if args.no_retrieval:
        retrieved = _build_direct_context(query_text=query_text)
    elif args.retrieval_backend == "bm25":
        retrieved, query_text = _retrieve_bm25(args, query_text, query_image)
    elif args.retrieval_backend == "hybrid":
        retrieved, query_text = _retrieve_hybrid(args, query_text, query_image)
    else:
        retrieved, query_text = _retrieve_vector(args, query_text, query_image)

    if not retrieved and not args.no_retrieval:
        print("\n=== Answer ===")
        print("No sufficiently relevant evidence found. Try adjusting query settings.")
        print("\n=== Retrieved Sources ===")
        print("(empty)")
        return

    generator = OpenAICompatibleGenerator()
    result = generator.generate(query=query_text, retrieved_chunks=retrieved)

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Retrieved Sources ===")
    for i, s in enumerate(result["sources"], start=1):
        print(
            f"[{i}] citation={s.get('citation')} "
            f"source={s.get('source')} "
            f"span=({s.get('start')}, {s.get('end')})"
        )

    if args.show_chunks:
        print("\n=== Retrieved Chunks ===")
        for i, item in enumerate(retrieved, start=1):
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()

            modality = meta.get("modality", "text")
            image_path = meta.get("image_path")
            time_sec = meta.get("time_sec")
            distance = item.get("distance")
            score = item.get("score")
            hybrid_score = item.get("hybrid_score")

            extra = [f"modality={modality}"]
            if isinstance(score, (int, float)):
                extra.append(f"score={score:.4f}")
            if isinstance(hybrid_score, (int, float)):
                extra.append(f"hybrid_score={hybrid_score:.4f}")
            if isinstance(distance, (int, float)):
                extra.append(f"distance={distance:.4f}")
            if isinstance(time_sec, (int, float)):
                extra.append(f"time_sec={time_sec:.2f}")
            if image_path:
                extra.append(f"image_path={image_path}")

            extra_info = " ".join(extra)
            print(f"[{i}] source={meta.get('source')} chunk_id={meta.get('chunk_id')} {extra_info}")
            print(text)
            print("-" * 80)


if __name__ == "__main__":
    main()
