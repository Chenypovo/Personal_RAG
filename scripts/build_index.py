import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chunker.text_chunker import chunk_text  # noqa: E402
from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.loader import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402
from app.vectordb.elasticsearch_store import ElasticsearchStore  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


def collect_files(input_dir: str) -> List[str]:
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    files: List[str] = []
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(fp))
    return files


def _as_vector_matrix(vectors: Any) -> List[List[float]]:
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()

    if not isinstance(vectors, list):
        raise TypeError(f"Expected list-like vectors, got {type(vectors).__name__}")

    out: List[List[float]] = []
    for row in vectors:
        if not isinstance(row, list):
            raise TypeError("Each vector row must be a list")
        out.append([float(x) for x in row])
    return out


def _normalize_entry(entry: Dict[str, Any], fallback_source: str, fallback_file_type: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source": entry.get("source", fallback_source),
        "file_type": entry.get("file_type", fallback_file_type),
        "modality": entry.get("modality", "text"),
        "chunk_id": entry.get("chunk_id", 0),
        "text": entry.get("text", ""),
    }
    if "image_path" in entry:
        out["image_path"] = entry["image_path"]
    if "time_sec" in entry:
        out["time_sec"] = entry["time_sec"]
    if "start" in entry:
        out["start"] = entry["start"]
    if "end" in entry:
        out["end"] = entry["end"]
    return out


def _make_semantic_embed_fn(
    *,
    semantic_embed_backend: str,
    semantic_batch_size: int,
    semantic_embed_model: str,
    semantic_clip_model_name: str,
    semantic_clip_device: str,
) -> Optional[Callable[[Sequence[str]], List[List[float]]]]:
    backend = semantic_embed_backend.strip().lower()
    if backend == "openai":
        model = semantic_embed_model.strip() or None
        embedder = OpenAICompatibleEmbedder(model=model, batch_size=semantic_batch_size)

        def _embed(texts: Sequence[str]) -> List[List[float]]:
            vec_raw = embedder.embed_texts(texts, output_type="list")
            return _as_vector_matrix(vec_raw)

        return _embed

    if backend == "clip":
        clip_embedder = CLIPMultimodalEmbedder(
            model_name=semantic_clip_model_name.strip(),
            device=semantic_clip_device or None,
            batch_size=semantic_batch_size,
        )

        def _embed(texts: Sequence[str]) -> List[List[float]]:
            vec_raw = clip_embedder.embed_texts(texts, output_type="list")
            return _as_vector_matrix(vec_raw)

        return _embed

    raise ValueError(f"Unsupported semantic_embed_backend: {semantic_embed_backend}")


def build_records(
    file_paths: List[str],
    chunk_size: int,
    overlap: int,
    chunk_strategy: str,
    semantic_threshold: float,
    semantic_min_sentences: int,
    semantic_embed_fn: Optional[Callable[[Sequence[str]], List[List[float]]]],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for file_path in file_paths:
        doc = load_document(file_path)

        entries = doc.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                records.append(
                    _normalize_entry(
                        entry=entry,
                        fallback_source=str(doc.get("source", file_path)),
                        fallback_file_type=str(doc.get("file_type", "unknown")),
                    )
                )
            continue

        text = doc.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        chunks = chunk_text(
            text=text,
            source=str(doc.get("source", file_path)),
            chunk_size=chunk_size,
            overlap=overlap,
            chunk_strategy=chunk_strategy,
            semantic_threshold=semantic_threshold,
            semantic_min_sentences=semantic_min_sentences,
            semantic_embed_fn=semantic_embed_fn,
        )
        for c in chunks:
            c["modality"] = "text"
            c["file_type"] = doc.get("file_type", "text")
        records.extend(chunks)

    return records


def embed_records_openai(records: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    text_records = [r for r in records if str(r.get("modality", "text")) == "text"]
    skipped = len(records) - len(text_records)
    if skipped > 0:
        print(f"[openai] skip non-text records: {skipped}")

    if not text_records:
        return [], []

    embedder = OpenAICompatibleEmbedder()
    texts = [str(r.get("text", "")) for r in text_records]
    vectors_raw = embedder.embed_texts(texts, output_type="list")
    vectors = _as_vector_matrix(vectors_raw)
    return vectors, text_records


def embed_records_clip(
    records: List[Dict[str, Any]],
    clip_model_name: str,
    clip_device: str,
    clip_batch_size: int,
) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    embedder = CLIPMultimodalEmbedder(
        model_name=clip_model_name,
        device=clip_device or None,
        batch_size=clip_batch_size,
    )

    vectors: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    text_records = [r for r in records if str(r.get("modality", "text")) == "text"]
    image_records = [r for r in records if str(r.get("modality", "")) == "image" and r.get("image_path")]

    if text_records:
        text_vectors_raw = embedder.embed_texts(
            [str(r.get("text", "")) for r in text_records],
            output_type="list",
        )
        text_vectors = _as_vector_matrix(text_vectors_raw)
        vectors.extend(text_vectors)
        metadatas.extend(text_records)

    if image_records:
        image_vectors_raw = embedder.embed_images(
            [str(r["image_path"]) for r in image_records],
            output_type="list",
        )
        image_vectors = _as_vector_matrix(image_vectors_raw)
        vectors.extend(image_vectors)
        metadatas.extend(image_records)

    return vectors, metadatas


def build_bm25(
    records: List[Dict[str, Any]],
    bm25_path: str,
    jieba_user_dicts: List[str],
    enable_cjk_bigram_fallback: bool,
) -> Tuple[int, int]:
    bm25 = BM25Store(
        user_dict_paths=jieba_user_dicts,
        enable_cjk_bigram_fallback=enable_cjk_bigram_fallback,
    )
    bm25.build_from_records(records)

    if not bm25.corpus_tokens:
        return 0, 0

    bm25.save(bm25_path)
    return len(bm25.corpus_tokens), len(bm25.metadatas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index (FAISS/Elasticsearch) + BM25 from local multimodal files")
    parser.add_argument("--input-dir", type=str, default="data/uploads")
    parser.add_argument(
        "--vector-backend",
        type=str,
        default="faiss",
        choices=["faiss", "elasticsearch"],
        help="Vector index backend.",
    )
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--bm25-path", type=str, default="data/index/bm25.json")
    parser.add_argument("--es-url", type=str, default="http://localhost:9200")
    parser.add_argument("--es-index", type=str, default="personal_rag")
    parser.add_argument("--es-username", type=str, default="")
    parser.add_argument("--es-password", type=str, default="")
    parser.add_argument("--es-api-key", type=str, default="")
    parser.add_argument("--es-request-timeout", type=int, default=30)
    parser.add_argument("--es-batch-size", type=int, default=200)
    parser.add_argument("--es-recreate-index", action="store_true", help="Delete and recreate ES index before build")
    parser.add_argument("--es-no-verify-certs", action="store_true", help="Disable TLS cert verification for ES client")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--chunk-strategy", type=str, default="token", choices=["token", "semantic"])
    parser.add_argument("--semantic-threshold", type=float, default=0.78)
    parser.add_argument("--semantic-min-sentences", type=int, default=2)
    parser.add_argument("--semantic-embed-backend", type=str, default="openai", choices=["openai", "clip"])
    parser.add_argument("--semantic-embed-model", type=str, default="", help="Optional model for openai semantic chunking.")
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--semantic-clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--semantic-clip-device", type=str, default="cpu")

    parser.add_argument(
        "--embed-backend",
        type=str,
        default="openai",
        choices=["openai", "clip"],
        help="openai: text-only embeddings; clip: text+image/video-frames in one space",
    )
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", type=str, default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)

    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 build")
    parser.add_argument(
        "--jieba-user-dict",
        action="append",
        default=[],
        help="Path to jieba custom dictionary. Can be used multiple times.",
    )
    parser.add_argument(
        "--disable-cjk-bigram-fallback",
        action="store_true",
        help="Disable CJK 2-gram fallback in BM25 tokenizer",
    )

    args = parser.parse_args()

    file_paths = collect_files(args.input_dir)
    if not file_paths:
        raise ValueError(f"No supported files found in: {args.input_dir}")

    semantic_embed_fn: Optional[Callable[[Sequence[str]], List[List[float]]]] = None
    if args.chunk_strategy == "semantic":
        semantic_embed_fn = _make_semantic_embed_fn(
            semantic_embed_backend=args.semantic_embed_backend,
            semantic_batch_size=int(args.semantic_batch_size),
            semantic_embed_model=args.semantic_embed_model,
            semantic_clip_model_name=args.semantic_clip_model_name,
            semantic_clip_device=args.semantic_clip_device,
        )

    print(f"Found files: {len(file_paths)}")
    records = build_records(
        file_paths=file_paths,
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        chunk_strategy=args.chunk_strategy,
        semantic_threshold=float(args.semantic_threshold),
        semantic_min_sentences=int(args.semantic_min_sentences),
        semantic_embed_fn=semantic_embed_fn,
    )
    if not records:
        raise ValueError("No records generated from input files")

    text_count = sum(1 for r in records if str(r.get("modality", "text")) == "text")
    image_count = sum(1 for r in records if str(r.get("modality", "")) == "image")
    print(f"Generated records: {len(records)} (text={text_count}, image={image_count})")

    if args.embed_backend == "openai":
        vectors, metadatas = embed_records_openai(records)
    else:
        vectors, metadatas = embed_records_clip(
            records=records,
            clip_model_name=args.clip_model_name,
            clip_device=args.clip_device,
            clip_batch_size=args.clip_batch_size,
        )

    if not vectors:
        raise ValueError("No vectors generated. Check files and embed backend.")
    if len(vectors) != len(metadatas):
        raise ValueError("vectors and metadatas length mismatch")

    dim = len(vectors[0])
    if args.vector_backend == "faiss":
        store = FaissStore(dim=dim)
        store.add(vectors=vectors, metadatas=metadatas)
        store.save(index_path=args.index_path, meta_path=args.meta_path)

        print("FAISS build complete.")
        print(f"FAISS index saved: {args.index_path}")
        print(f"Metadata saved: {args.meta_path}")
        print(f"Vector dim: {dim}")
        print(f"Stored vectors: {len(vectors)}")
    else:
        es_store = ElasticsearchStore.connect(
            url=args.es_url,
            index_name=args.es_index,
            username=args.es_username,
            password=args.es_password,
            api_key=args.es_api_key,
            verify_certs=not args.es_no_verify_certs,
            request_timeout=int(args.es_request_timeout),
        )
        es_store.recreate_index(dim=dim, recreate=bool(args.es_recreate_index))
        n_indexed = es_store.add(
            vectors=vectors,
            metadatas=metadatas,
            batch_size=int(args.es_batch_size),
            refresh=True,
        )
        print("Elasticsearch build complete.")
        print(f"ES index: {args.es_index}")
        print(f"ES url: {args.es_url}")
        print(f"Vector dim: {dim}")
        print(f"Indexed vectors: {n_indexed}")
        print(f"Indexed docs (count API): {es_store.count()}")

    if args.skip_bm25:
        print("BM25 skipped (--skip-bm25).")
    else:
        bm25_docs, bm25_metas = build_bm25(
            records=records,
            bm25_path=args.bm25_path,
            jieba_user_dicts=args.jieba_user_dict,
            enable_cjk_bigram_fallback=not args.disable_cjk_bigram_fallback,
        )
        if bm25_docs == 0:
            print("BM25: no text records, nothing saved.")
        else:
            print("BM25 build complete.")
            print(f"BM25 saved: {args.bm25_path}")
            print(f"BM25 docs: {bm25_docs}, metadatas: {bm25_metas}")


if __name__ == "__main__":
    main()
