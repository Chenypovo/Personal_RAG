import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chunker.text_chunker import chunk_text  # noqa: E402
from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.loader import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


UPLOAD_TYPES = [
    "txt",
    "md",
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "webp",
    "bmp",
    "mp4",
    "mov",
    "avi",
    "mkv",
    "webm",
]

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]


def _save_uploaded_files(uploaded_files: List[Any], upload_dir: Path) -> List[Path]:
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []
    for f in uploaded_files:
        target = upload_dir / f.name
        target.write_bytes(f.getbuffer())
        saved_paths.append(target)
    return saved_paths


def _collect_files(input_dir: Path) -> List[str]:
    files: List[str] = []
    for fp in input_dir.rglob("*"):
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


def _build_records(file_paths: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
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
        )
        for c in chunks:
            c["modality"] = "text"
            c["file_type"] = doc.get("file_type", "text")
        records.extend(chunks)
    return records


def _build_index_openai(records: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
    text_records = [r for r in records if str(r.get("modality", "text")) == "text"]
    if not text_records:
        return [], []
    embedder = OpenAICompatibleEmbedder()
    vec_raw = embedder.embed_texts([str(r.get("text", "")) for r in text_records], output_type="list")
    vectors = _as_vector_matrix(vec_raw)
    return vectors, text_records


def _build_index_clip(
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
        vec_raw = embedder.embed_texts([str(r.get("text", "")) for r in text_records], output_type="list")
        vectors.extend(_as_vector_matrix(vec_raw))
        metadatas.extend(text_records)

    if image_records:
        vec_raw = embedder.embed_images([str(r["image_path"]) for r in image_records], output_type="list")
        vectors.extend(_as_vector_matrix(vec_raw))
        metadatas.extend(image_records)

    return vectors, metadatas


def _build_bm25(records: List[Dict[str, Any]], bm25_path: Path) -> Tuple[int, int]:
    bm25 = BM25Store()
    bm25.build_from_records(records)
    if not bm25.corpus_tokens:
        return 0, 0
    bm25.save(str(bm25_path))
    return len(bm25.corpus_tokens), len(bm25.metadatas)


def _query_vector(
    embed_backend: str,
    query_text: str,
    query_image_path: str,
    clip_model_name: str,
    clip_device: str,
    clip_batch_size: int,
) -> Tuple[List[float], str]:
    q_text = query_text

    if embed_backend == "openai":
        if not q_text.strip():
            raise ValueError("openai backend requires text query")
        embedder = OpenAICompatibleEmbedder()
        q = embedder.embed_text(q_text, output_type="list")
        return _as_vector(q), q_text

    clip_embedder = CLIPMultimodalEmbedder(
        model_name=clip_model_name,
        device=clip_device or None,
        batch_size=clip_batch_size,
    )

    if query_image_path:
        q = clip_embedder.embed_image(query_image_path, output_type="list")
        if not q_text.strip():
            q_text = f"Image query: {os.path.basename(query_image_path)}"
        return _as_vector(q), q_text

    if not q_text.strip():
        raise ValueError("clip backend requires text query or query image")

    q = clip_embedder.embed_text(q_text, output_type="list")
    return _as_vector(q), q_text


def _retrieve_vector(
    *,
    index_path: Path,
    meta_path: Path,
    query_vector: List[float],
    top_k: int,
    candidate_k: int,
    max_distance: float,
    strict_mode: bool,
) -> List[Dict[str, Any]]:
    store = FaissStore.load(index_path=str(index_path), meta_path=str(meta_path))
    candidates = store.search(query_vector=query_vector, top_k=max(top_k, candidate_k))

    filtered: List[Dict[str, Any]] = []
    for item in candidates:
        d = item.get("distance")
        if isinstance(d, (int, float)) and d <= float(max_distance):
            filtered.append(item)

    if filtered:
        return filtered[:top_k]
    if strict_mode:
        return []
    return candidates[:top_k]


def _retrieve_bm25(*, bm25_path: Path, query_text: str, top_k: int) -> List[Dict[str, Any]]:
    bm25 = BM25Store.load(str(bm25_path))
    return bm25.search(query=query_text, top_k=top_k)


def _retrieve_hybrid(
    *,
    index_path: Path,
    meta_path: Path,
    bm25_path: Path,
    query_text: str,
    query_vector: List[float],
    top_k: int,
    vector_k: int,
    bm25_k: int,
    rrf_k: int,
    candidate_k: int,
    max_distance: float,
    strict_mode: bool,
) -> List[Dict[str, Any]]:
    vec_items = _retrieve_vector(
        index_path=index_path,
        meta_path=meta_path,
        query_vector=query_vector,
        top_k=max(top_k, vector_k),
        candidate_k=max(candidate_k, vector_k),
        max_distance=max_distance,
        strict_mode=strict_mode,
    )
    bm25_items = _retrieve_bm25(
        bm25_path=bm25_path,
        query_text=query_text,
        top_k=max(top_k, bm25_k),
    )

    fused_scores: Dict[str, float] = {}
    chosen_item: Dict[str, Dict[str, Any]] = {}

    for rank, item in enumerate(vec_items, start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        chosen_item[key] = item

    for rank, item in enumerate(bm25_items, start=1):
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        key = _doc_key(meta, rank)
        fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        if key not in chosen_item:
            chosen_item[key] = item
        else:
            merged = dict(chosen_item[key])
            if "score" not in merged and isinstance(item.get("score"), (int, float)):
                merged["score"] = item["score"]
            chosen_item[key] = merged

    ranked_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    fused: List[Dict[str, Any]] = []
    for key in ranked_keys[:top_k]:
        item = dict(chosen_item[key])
        item["hybrid_score"] = float(fused_scores[key])
        fused.append(item)
    return fused


def main() -> None:
    st.set_page_config(page_title="Personal RAG", layout="wide")
    st.title("Personal RAG (Multimodal)")
    st.caption("Upload files -> build index -> ask with text/image query")

    data_dir = Path(PROJECT_ROOT) / "data"
    upload_dir = data_dir / "uploads"
    query_dir = data_dir / "queries"
    index_path = data_dir / "index" / "faiss.index"
    meta_path = data_dir / "index" / "metadatas.json"
    bm25_path = data_dir / "index" / "bm25.json"

    with st.sidebar:
        st.header("Index Settings")
        embed_backend = st.selectbox("embed_backend", options=["openai", "clip"], index=0)
        chunk_size = st.number_input("chunk_size", min_value=100, max_value=4000, value=700, step=50)
        overlap = st.number_input("overlap", min_value=0, max_value=1000, value=120, step=10)

        st.subheader("CLIP")
        clip_model_name = st.text_input("clip_model_name", value="openai/clip-vit-base-patch32")
        clip_device = st.selectbox("clip_device", options=["cpu", "cuda"], index=0)
        clip_batch_size = st.number_input("clip_batch_size", min_value=1, max_value=128, value=32, step=1)

        st.subheader("Retrieval")
        retrieval_backend = st.selectbox("retrieval_backend", options=["vector", "hybrid", "bm25"], index=1)
        top_k = st.number_input("top_k", min_value=1, max_value=20, value=4, step=1)
        candidate_k = st.number_input("candidate_k", min_value=1, max_value=200, value=40, step=1)
        max_distance = st.number_input("max_distance", min_value=0.0, max_value=10.0, value=0.8, step=0.05)
        strict_mode = st.checkbox("strict_mode (no backfill)", value=True)
        vector_k = st.number_input("vector_k (hybrid)", min_value=1, max_value=200, value=40, step=1)
        bm25_k = st.number_input("bm25_k (hybrid)", min_value=1, max_value=200, value=40, step=1)
        rrf_k = st.number_input("rrf_k (hybrid)", min_value=1, max_value=500, value=60, step=1)

    st.subheader("1) Upload Files")
    uploaded_files = st.file_uploader(
        "Supported: txt/md/pdf/image/video",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("Save Uploads", use_container_width=True):
            if not uploaded_files:
                st.warning("No files selected.")
            else:
                saved = _save_uploaded_files(uploaded_files, upload_dir)
                st.success(f"Saved {len(saved)} files to {upload_dir}")

    with col2:
        if st.button("Build Index", use_container_width=True):
            try:
                with st.spinner("Building index..."):
                    if uploaded_files:
                        _save_uploaded_files(uploaded_files, upload_dir)

                    file_paths = _collect_files(upload_dir)
                    if not file_paths:
                        raise ValueError(f"No supported files found in: {upload_dir}")

                    records = _build_records(file_paths, int(chunk_size), int(overlap))
                    if not records:
                        raise ValueError("No records generated from input files")

                    if embed_backend == "openai":
                        vectors, metadatas = _build_index_openai(records)
                    else:
                        vectors, metadatas = _build_index_clip(
                            records=records,
                            clip_model_name=clip_model_name.strip(),
                            clip_device=clip_device,
                            clip_batch_size=int(clip_batch_size),
                        )

                    if not vectors:
                        text_count = sum(1 for r in records if str(r.get("modality", "text")) == "text")
                        image_count = sum(1 for r in records if str(r.get("modality", "")) == "image")
                        if embed_backend == "openai":
                            raise ValueError(
                                "No vectors generated in openai mode. "
                                f"Current records: text={text_count}, image={image_count}. "
                                "Use clip mode for image/video or upload text/PDF with extractable text."
                            )
                        raise ValueError("No vectors generated")

                    dim = len(vectors[0])
                    store = FaissStore(dim=dim)
                    store.add(vectors=vectors, metadatas=metadatas)
                    store.save(index_path=str(index_path), meta_path=str(meta_path))

                    bm25_docs, _ = _build_bm25(records=records, bm25_path=bm25_path)

                text_count = sum(1 for r in metadatas if str(r.get("modality", "text")) == "text")
                image_count = sum(1 for r in metadatas if str(r.get("modality", "")) == "image")
                st.success(
                    f"Index built: files={len(file_paths)} vectors={len(vectors)} dim={dim} "
                    f"(text={text_count}, image={image_count}) | bm25_docs={bm25_docs}"
                )
            except Exception as e:
                st.error(f"Build failed: {e}")

    st.divider()
    st.subheader("2) Ask")
    query_text = st.text_input("Text query", placeholder="e.g. What is FAISS?")
    query_image_file = st.file_uploader("Optional query image (for clip)", type=IMAGE_TYPES, accept_multiple_files=False)
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

    if st.button("Run Query", type="primary"):
        saved_query_image = ""
        if query_image_file is not None:
            query_dir.mkdir(parents=True, exist_ok=True)
            target = query_dir / query_image_file.name
            target.write_bytes(query_image_file.getbuffer())
            saved_query_image = str(target)

        try:
            with st.spinner("Retrieving and generating..."):
                retrieved: List[Dict[str, Any]]
                effective_query_text = query_text

                if retrieval_backend == "bm25":
                    if not bm25_path.exists():
                        st.error("BM25 index file not found. Build index first.")
                        return
                    if saved_query_image:
                        st.error("BM25 backend does not support image-only query.")
                        return
                    if not effective_query_text.strip():
                        st.error("BM25 backend requires text query.")
                        return
                    retrieved = _retrieve_bm25(
                        bm25_path=bm25_path,
                        query_text=effective_query_text,
                        top_k=int(top_k),
                    )
                else:
                    if not index_path.exists() or not meta_path.exists():
                        st.error("FAISS index files not found. Build index first.")
                        return

                    query_vector, effective_query_text = _query_vector(
                        embed_backend=embed_backend,
                        query_text=effective_query_text,
                        query_image_path=saved_query_image,
                        clip_model_name=clip_model_name.strip(),
                        clip_device=clip_device,
                        clip_batch_size=int(clip_batch_size),
                    )

                    if retrieval_backend == "vector":
                        retrieved = _retrieve_vector(
                            index_path=index_path,
                            meta_path=meta_path,
                            query_vector=query_vector,
                            top_k=int(top_k),
                            candidate_k=int(candidate_k),
                            max_distance=float(max_distance),
                            strict_mode=bool(strict_mode),
                        )
                    else:
                        if not bm25_path.exists():
                            st.error("BM25 index file not found. Build index first.")
                            return
                        if not effective_query_text.strip():
                            st.error("Hybrid backend requires text query.")
                            return
                        retrieved = _retrieve_hybrid(
                            index_path=index_path,
                            meta_path=meta_path,
                            bm25_path=bm25_path,
                            query_text=effective_query_text,
                            query_vector=query_vector,
                            top_k=int(top_k),
                            vector_k=int(vector_k),
                            bm25_k=int(bm25_k),
                            rrf_k=int(rrf_k),
                            candidate_k=int(candidate_k),
                            max_distance=float(max_distance),
                            strict_mode=bool(strict_mode),
                        )

                if not retrieved:
                    st.warning(
                        "No sufficiently relevant evidence found. "
                        "Try hybrid backend, increase top_k/candidate_k, or disable strict_mode."
                    )
                    return

                generator = OpenAICompatibleGenerator()
                result = generator.generate(query=effective_query_text, retrieved_chunks=retrieved)

            st.markdown("### Answer")
            st.write(result.get("answer", ""))

            st.markdown("### Sources")
            for i, s in enumerate(result.get("sources", []), start=1):
                st.write(
                    f"[{i}] {s.get('citation')} | source={s.get('source')} "
                    f"| span=({s.get('start')}, {s.get('end')})"
                )

            if show_chunks:
                with st.expander("Retrieved Chunks"):
                    for i, item in enumerate(retrieved, start=1):
                        meta_raw = item.get("metadata")
                        meta = meta_raw if isinstance(meta_raw, dict) else {}
                        text_raw = meta.get("text", "")
                        text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
                        text = text.strip()
                        distance = item.get("distance")
                        score = item.get("score")
                        hybrid_score = item.get("hybrid_score")

                        extra: List[str] = []
                        if isinstance(score, (int, float)):
                            extra.append(f"score={score:.4f}")
                        if isinstance(hybrid_score, (int, float)):
                            extra.append(f"hybrid_score={hybrid_score:.4f}")
                        if isinstance(distance, (int, float)):
                            extra.append(f"distance={distance:.4f}")
                        extra_text = (" " + " ".join(extra)) if extra else ""

                        st.markdown(
                            f"**[{i}] source={meta.get('source')} chunk_id={meta.get('chunk_id')}{extra_text}**"
                        )
                        st.write(text)
                        if meta.get("image_path"):
                            st.caption(f"image_path: {meta.get('image_path')}")
                        st.divider()
        except Exception as e:
            st.error(f"Query failed: {e}")


if __name__ == "__main__":
    main()
