import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.embedder.embedder import OpenAICompatibleEmbedder  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.retriever.faiss_retriever import FaissRetriever  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG query demo (CLI)")
    parser.add_argument("--query", type=str, required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--no-retrieval", action="store_true", help="Disable retrieval and do direct chat")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunk texts")
    args = parser.parse_args()

    generator = OpenAICompatibleGenerator()
    if args.no_retrieval:
        retrieved = [
            {
                "metadata": {
                    "source": "direct_query",
                    "chunk_id": -1,
                    "start": 0,
                    "end": len(args.query),
                    "text": args.query,
                }
            }
        ]
    else:
        store = FaissStore.load(index_path=args.index_path, meta_path=args.meta_path)
        embedder = OpenAICompatibleEmbedder()
        retriever = FaissRetriever(store=store, embedder=embedder)
        retrieved = retriever.retrieve(query=args.query, top_k=args.top_k)

    result = generator.generate(query=args.query, retrieved_chunks=retrieved)

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
            meta = item.get("metadata", {})
            text = (meta.get("text", "") or "").strip()
            print(f"[{i}] source={meta.get('source')} chunk_id={meta.get('chunk_id')}")
            print(text)
            print("-" * 80)


if __name__ == "__main__":
    main()
