import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.chunker.text_chunker import chunk_text  # noqa: E402
from app.embedder.embedder import OpenAICompatibleEmbedder  # noqa: E402
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


def build_chunks(file_paths: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for file_path in file_paths:
        doc = load_document(file_path)
        chunks = chunk_text(
            text=doc["text"],
            source=doc["source"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from local documents")
    parser.add_argument("--input-dir", type=str, default="data/uploads")
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()

    file_paths = collect_files(args.input_dir)
    if not file_paths:
        raise ValueError(f"No supported files found in: {args.input_dir}")

    print(f"Found files: {len(file_paths)}")
    chunks = build_chunks(file_paths, args.chunk_size, args.overlap)
    if not chunks:
        raise ValueError("No chunks generated from input files")

    print(f"Generated chunks: {len(chunks)}")
    embedder = OpenAICompatibleEmbedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts)
    if not vectors:
        raise ValueError("No vectors generated")

    dim = len(vectors[0])
    store = FaissStore(dim=dim)
    store.add(vectors=vectors, metadatas=chunks)
    store.save(index_path=args.index_path, meta_path=args.meta_path)

    print("Index build complete.")
    print(f"Index saved: {args.index_path}")
    print(f"Metadata saved: {args.meta_path}")
    print(f"Vector dim: {dim}")


if __name__ == "__main__":
    main()
