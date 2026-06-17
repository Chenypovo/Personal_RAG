"""Evaluate the memory subsystem.

Two parts (the "unified" eval alongside scripts/eval_retrieval.py, same metrics):
  1. Memory recall: Recall@K / MRR@K of recalling the right stored facts.
  2. Merge decision (optional): accuracy of add/update/delete decisions.

Recall eval seeds a memory store from a facts dataset (each fact has an id),
then runs labeled queries and scores which facts come back.

Run:
    python scripts/eval_memory.py \
        --facts data/eval_memory/facts_example.jsonl \
        --queries data/eval_memory/queries_example.jsonl \
        --qrels data/eval_memory/qrels_example.jsonl \
        --ks 1,3,5
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.llm import make_embed_fn  # noqa: E402
from app.eval.memory_eval import MemQuery, evaluate_recall  # noqa: E402
from app.memory.models import MemoryFact  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.memory.vector_index import NumpyVectorIndex  # noqa: E402


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
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be an object: {path}:{line_no}")
            rows.append(obj)
    return rows


def _parse_ks(ks: str) -> List[int]:
    out = [int(x) for x in ks.split(",") if x.strip()]
    if not out:
        raise ValueError("No valid K provided")
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory recall with Recall@K / MRR@K")
    parser.add_argument("--facts", required=True, help="jsonl: {id, fact_content, fact_object?}")
    parser.add_argument("--queries", required=True, help="jsonl: {query_id, query}")
    parser.add_argument("--qrels", required=True, help="jsonl: {query_id, mem_id}")
    parser.add_argument("--ks", default="1,3,5")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)

    facts = _load_jsonl(args.facts)
    query_rows = _load_jsonl(args.queries)
    qrel_rows = _load_jsonl(args.qrels)

    queries = [MemQuery(query_id=str(r["query_id"]), query=str(r["query"])) for r in query_rows]
    qrels: Dict[str, Set[str]] = {}
    for r in qrel_rows:
        qrels.setdefault(str(r["query_id"]), set()).add(str(r["mem_id"]))

    embed_fn = make_embed_fn()
    dim = len(embed_fn("dimension probe"))

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(
            db_path=str(Path(tmp) / "eval_mem.db"),
            vector_index=NumpyVectorIndex(dim=dim),
            embed_fn=embed_fn,
        )
        for fr in facts:
            store.add(
                MemoryFact(
                    id=str(fr["id"]),
                    fact_content=str(fr["fact_content"]),
                    fact_object=str(fr.get("fact_object", "")),
                )
            )

        def search_fn(query: str, top_k: int) -> List[str]:
            return [f.id for f, _score in store.search(query, top_k=top_k)]

        report = evaluate_recall(search_fn, queries, qrels, ks=ks)

    print("\n=== Memory Recall Eval ===")
    print(f"facts={len(facts)}  queries={len(queries)}  eval_queries={report['n_eval']}  ks={ks}")
    for k in ks:
        print(f"  recall@{k}={report['metrics'][f'recall@{k}']:.4f}  mrr@{k}={report['metrics'][f'mrr@{k}']:.4f}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report: {args.output_json}")


if __name__ == "__main__":
    main()
