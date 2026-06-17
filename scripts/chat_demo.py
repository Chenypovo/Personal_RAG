"""Interactive multi-turn chat that demonstrates the memory-augmented agent.

Each turn the agent:
  1. routes (decide use_docs / use_memory / write_memory)
  2. recalls relevant long-term memories
  3. retrieves document evidence (FAISS vector search, if an index exists)
  4. generates an evidence-constrained answer personalized with memory
  5. extracts new facts from your message and merges them into long-term memory

Run:
    python scripts/chat_demo.py
Type 'exit' / 'quit' to leave, ':mem' to list stored memories.
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.factory import build_memory_agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-augmented RAG agent (interactive)")
    parser.add_argument("--memory-dir", default="data/memory")
    parser.add_argument("--index-path", default="data/index/faiss.index")
    parser.add_argument("--meta-path", default="data/index/metadatas.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-distance", type=float, default=0.8)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Rerank document evidence with BGE")
    args = parser.parse_args()

    bundle = build_memory_agent(
        memory_dir=args.memory_dir,
        index_path=args.index_path,
        meta_path=args.meta_path,
        top_k=args.top_k,
        max_distance=args.max_distance,
        recall_k=args.recall_k,
        use_rerank=args.rerank,
    )
    agent, store = bundle.agent, bundle.store

    print("=" * 70)
    print("Memory-augmented RAG agent. Type 'exit' to quit, ':mem' to list memories.")
    print("=" * 70)

    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break
        if user_msg == ":mem":
            facts = store.list_active()
            print(f"\n[stored memories: {len(facts)}]")
            for f in facts:
                obj = f"({f.fact_object}) " if f.fact_object else ""
                print(f"  - {obj}{f.fact_content}  [{f.visibility}, {f.source}]")
            continue

        result = agent.chat(user_msg)

        print(f"\nAssistant: {result.answer}")

        if result.recalled_memories:
            print("\n  ↳ recalled about you:")
            for m in result.recalled_memories:
                print(f"      • {m.fact_content}")
        if result.sources:
            cites = ", ".join(str(s.get("citation")) for s in result.sources)
            print(f"  ↳ sources: {cites}")
        if result.memory_ops:
            applied = [f"{o.type}:{o.fact_content or o.id}" for o in result.memory_ops if o.applied]
            if applied:
                print(f"  ↳ memory updated: {', '.join(applied)}")

        bundle.save()  # persist memory vector index after each turn

    bundle.save()
    print("Memory saved. Bye.")


if __name__ == "__main__":
    main()
