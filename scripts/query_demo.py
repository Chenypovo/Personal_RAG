import argparse
import os
import sys
from typing import Any, Dict, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.embedder.embedder import CLIPMultimodalEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG query demo (CLI)")
    parser.add_argument("--query", type=str, default="", help="Text query")
    parser.add_argument("--query-image", type=str, default="", help="Image query path (for clip backend)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=40, help="Initial recall size before filtering")
    parser.add_argument("--max-distance", type=float, default=0.8, help="Keep results with distance <= threshold")
    parser.add_argument("--strict", dest="strict", action="store_true", default=True, help="Strict mode (default: on)")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Backfill nearest items when none pass threshold")
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index")
    parser.add_argument("--meta-path", type=str, default="data/index/metadatas.json")
    parser.add_argument("--no-retrieval", action="store_true", help="Disable retrieval and do direct generation")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunk texts")

    parser.add_argument(
        "--embed-backend",
        type=str,
        default="openai",
        choices=["openai", "clip"],
        help="openai: text query only; clip: text/image query",
    )
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-device", type=str, default="cpu")
    parser.add_argument("--clip-batch-size", type=int, default=32)

    args = parser.parse_args()

    query_text = (args.query or "").strip()
    query_image = (args.query_image or "").strip()

    if not args.no_retrieval:
        if args.embed_backend == "openai" and not query_text:
            raise ValueError("openai backend requires --query")
        if args.embed_backend == "clip" and not query_text and not query_image:
            raise ValueError("clip backend requires --query or --query-image")

    generator = OpenAICompatibleGenerator()

    if args.no_retrieval:
        if not query_text:
            query_text = "Please answer the user query directly."
        retrieved = _build_direct_context(query_text=query_text)
    else:
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
            retrieved = filtered[: args.top_k]
        else:
            retrieved = [] if args.strict else candidates[: args.top_k]

    if not retrieved and not args.no_retrieval:
        print("\n=== Answer ===")
        print("No sufficiently relevant evidence found. Try adjusting query or --max-distance.")
        print("\n=== Retrieved Sources ===")
        print("(empty)")
        return

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

            extra = [f"modality={modality}"]
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
