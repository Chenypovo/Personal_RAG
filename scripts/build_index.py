import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chunker.text_chunker import chunk_text  # noqa: E402
from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.loader import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
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


def build_records(file_paths: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from local multimodal files")
    parser.add_argument("--input-dir", type=str, default="data/uploads")
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)

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

    args = parser.parse_args()

    file_paths = collect_files(args.input_dir)
    if not file_paths:
        raise ValueError(f"No supported files found in: {args.input_dir}")

    print(f"Found files: {len(file_paths)}")
    records = build_records(file_paths, args.chunk_size, args.overlap)
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
    store = FaissStore(dim=dim)
    store.add(vectors=vectors, metadatas=metadatas)
    store.save(index_path=args.index_path, meta_path=args.meta_path)

    print("Index build complete.")
    print(f"Index saved: {args.index_path}")
    print(f"Metadata saved: {args.meta_path}")
    print(f"Vector dim: {dim}")
    print(f"Stored vectors: {len(vectors)}")


if __name__ == "__main__":
    main()
