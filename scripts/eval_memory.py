import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.llm import make_embed_fn  # noqa: E402
from app.eval.metrics import mean, summarize_rank_metrics  # noqa: E402
from app.memory.models import MemoryFact, new_id  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.memory.vector_index import NumpyVectorIndex  # noqa: E402


def _parse_ks(ks: str) -> List[int]:
    out: List[int] = []
    for raw in ks.split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Invalid K: {value}")
        out.append(value)
    if not out:
        raise ValueError("No valid K provided in --ks")
    return sorted(set(out))


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid jsonl at {path}:{line_no}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be an object: {path}:{line_no}")
            rows.append(obj)
    return rows


def _load_facts(path: str) -> List[MemoryFact]:
    facts: List[MemoryFact] = []
    for i, row in enumerate(_load_jsonl(path), start=1):
        fact_content = str(row.get("fact_content", "")).strip()
        if not fact_content:
            raise ValueError(f"facts line {i}: missing fact_content")
        facts.append(
            MemoryFact(
                id=str(row.get("fact_id") or row.get("id") or "").strip() or new_id(),
                fact_object=str(row.get("fact_object", "")).strip(),
                fact_content=fact_content,
                visibility=str(row.get("visibility", "PUBLIC")).strip() or "PUBLIC",
                state=str(row.get("state", "ACTIVE")).strip() or "ACTIVE",
                source=str(row.get("source", "eval")).strip() or "eval",
            )
        )
    return facts


def _load_queries(path: str) -> List[Dict[str, str]]:
    queries: List[Dict[str, str]] = []
    for i, row in enumerate(_load_jsonl(path), start=1):
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            raise ValueError(f"queries line {i}: missing query_id or query")
        queries.append({"query_id": query_id, "query": query})
    return queries


def _load_qrels(path: str) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    for i, row in enumerate(_load_jsonl(path), start=1):
        query_id = str(row.get("query_id", "")).strip()
        fact_id = str(row.get("fact_id") or row.get("doc_id") or "").strip()
        if not query_id or not fact_id:
            raise ValueError(f"qrels line {i}: missing query_id or fact_id")
        relevance = float(row.get("relevance", 1))
        if relevance <= 0:
            continue
        qrels.setdefault(query_id, set()).add(fact_id)
    return qrels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate memory retrieval with Recall@K and MRR@K")
    parser.add_argument("--facts", required=True, help="memory facts jsonl path")
    parser.add_argument("--queries", required=True, help="queries jsonl path")
    parser.add_argument("--qrels", required=True, help="qrels jsonl path")
    parser.add_argument("--ks", default="1,3,5", help="comma-separated K values")
    parser.add_argument("--output-json", default="", help="optional output json report path")
    args = parser.parse_args()

    ks = _parse_ks(args.ks)
    top_k_max = max(ks)

    embed_fn = make_embed_fn()
    dim = len(embed_fn("dimension probe"))
    store = MemoryStore(
        db_path=":memory:",
        vector_index=NumpyVectorIndex(dim=dim),
        embed_fn=embed_fn,
    )

    facts = _load_facts(args.facts)
    for fact in facts:
        if fact.state == "ACTIVE":
            store.add(fact)

    queries = _load_queries(args.queries)
    qrels = _load_qrels(args.qrels)

    per_query: List[Dict[str, Any]] = []
    recalls: Dict[int, List[float]] = {k: [] for k in ks}
    mrrs: Dict[int, List[float]] = {k: [] for k in ks}
    skipped_no_qrels = 0

    for item in queries:
        gold = qrels.get(item["query_id"], set())
        if not gold:
            skipped_no_qrels += 1
            continue

        hits = store.search(item["query"], top_k=top_k_max)
        predicted = [fact.id for fact, _score in hits]
        metrics = summarize_rank_metrics(predicted, gold, ks)
        row: Dict[str, Any] = {
            "query_id": item["query_id"],
            "pred_fact_ids": predicted,
        }
        row.update(metrics)
        for k in ks:
            recalls[k].append(metrics[f"recall@{k}"])
            mrrs[k].append(metrics[f"mrr@{k}"])
        per_query.append(row)

    report = {
        "n_facts": len(facts),
        "n_queries": len(queries),
        "n_eval_queries": len(per_query),
        "n_skipped_no_qrels": skipped_no_qrels,
        "metrics": {
            f"recall@{k}": mean(recalls[k]) for k in ks
        },
        "per_query": per_query,
    }
    for k in ks:
        report["metrics"][f"mrr@{k}"] = mean(mrrs[k])

    print("\n=== Memory Eval ===")
    print(f"facts={args.facts}")
    print(f"queries={args.queries}")
    print(f"qrels={args.qrels}")
    print(f"ks={ks}")
    print(f"n_eval_queries={report['n_eval_queries']} / {report['n_queries']}")
    for k in ks:
        print(
            f"  recall@{k}={report['metrics'][f'recall@{k}']:.4f} "
            f"mrr@{k}={report['metrics'][f'mrr@{k}']:.4f}"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
