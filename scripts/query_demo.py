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

    # Rerank options
    parser.add_argument("--use-rerank", action="store_true", help="Enable cross-encoder rerank")
    parser.add_argument("--rerank-model", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-top-n", type=int, default=20, help="Retrieve top-N from FAISS before rerank")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-device", type=str, default="", help='e.g. "cpu" or "cuda"')

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

        retrieve_k = args.top_k
        if args.use_rerank:
            retrieve_k = max(args.top_k, args.rerank_top_n)

        retrieved = retriever.retrieve(query=args.query, top_k=retrieve_k)

        if args.use_rerank and retrieved:
            from app.reranker.bge_reranker import BGEReranker  # lazy import

            reranker = BGEReranker(
                model_name=args.rerank_model,
                device=args.rerank_device or None,
                batch_size=args.rerank_batch_size,
            )
            retrieved = reranker.rerank(query=args.query, retrieved=retrieved, top_k=args.top_k)

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
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()
            score = item.get("rerank_score")
            score_part = f" rerank_score={score:.4f}" if isinstance(score, (int, float)) else ""
            print(f"[{i}] source={meta.get('source')} chunk_id={meta.get('chunk_id')}{score_part}")
            print(text)
            print("-" * 80)


if __name__ == "__main__":
    main()
