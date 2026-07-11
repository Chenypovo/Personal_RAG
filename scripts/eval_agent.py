"""离线评测：单次基线（MemoryAgent）vs 工具调用循环（ToolAgent）。

对每个任务分别用两种 agent 跑一遍，程序化判定任务成功率、多跳 Coverage@k、
工具选择 P/R 与成本（LLM 调用/工具调用/步数/解析失败率），打印对照表并落盘 JSON 报告。

用法：
    python scripts/eval_agent.py --tasks data/eval/agent_tasks_synth.jsonl
    python scripts/eval_agent.py --tasks data/eval/agent_tasks_example.jsonl --agent loop --limit 1
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

from app.agent.agent import MemoryAgent  # noqa: E402
from app.agent.factory import _build_doc_retriever  # noqa: E402
from app.agent.llm import make_complete_fn, make_embed_fn  # noqa: E402
from app.agent.loop import ToolAgent  # noqa: E402
from app.agent.registry import ToolRegistry  # noqa: E402
from app.agent.router import Router  # noqa: E402
from app.agent.tools.calculator import CalculatorTool  # noqa: E402
from app.agent.tools.memory_tools import ReadMemoryTool, WriteMemoryTool  # noqa: E402
from app.agent.tools.retrieve import RetrieveDocsTool  # noqa: E402
from app.eval.agent_eval import coverage_at_k, judge_success, tool_precision_recall  # noqa: E402
from app.eval.metrics import mean  # noqa: E402
from app.generator.generator import OpenAICompatibleGenerator  # noqa: E402
from app.memory.extractor import MemoryExtractor  # noqa: E402
from app.memory.merger import MemoryMerger  # noqa: E402
from app.memory.models import MemoryFact  # noqa: E402
from app.memory.store import MemoryStore  # noqa: E402
from app.memory.vector_index import NumpyVectorIndex  # noqa: E402


class CountingComplete:
    """包一层 complete_fn，统计每任务 LLM 调用次数。"""

    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.count += 1
        return self.fn(system_prompt, user_prompt)


class RecordingRetrieve:
    """包一层 retrieve_docs_fn，按检索顺序累积所有取回的 chunk（评 Coverage 用）。"""

    def __init__(self, fn):
        self.fn = fn
        self.chunks: List[Dict[str, Any]] = []

    def __call__(self, query: str) -> List[Dict[str, Any]]:
        out = self.fn(query)
        self.chunks.extend(out)
        return out


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Each line must be an object: {path}:{line_no}")
            rows.append(obj)
    return rows


def _make_store(tmp_dir: str, embed_fn, dim: int, setup_memories: List[str]) -> MemoryStore:
    store = MemoryStore(
        db_path=str(Path(tmp_dir) / "mem.db"),
        vector_index=NumpyVectorIndex(dim=dim),
        embed_fn=embed_fn,
    )
    for text in setup_memories:
        store.add(MemoryFact(fact_content=text))
    return store


def _baseline_tools(route) -> Set[str]:
    """把基线路由决策映射为伪工具集合，与循环的工具口径对齐。"""
    called: Set[str] = set()
    if route is None:
        return called
    if route.use_docs:
        called.add("retrieve_docs")
    if route.use_memory:
        called.add("read_memory")
    if route.write_memory:
        called.add("write_memory")
    return called


def run_task(kind: str, task: Dict[str, Any], ctx: Dict[str, Any], k: int) -> Dict[str, Any]:
    complete = CountingComplete(ctx["complete_fn"])
    retrieve = RecordingRetrieve(ctx["retrieve"])
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(tmp, ctx["embed_fn"], ctx["dim"], task.get("setup_memories", []))
        extractor = MemoryExtractor(complete_fn=complete)
        merger = MemoryMerger(store=store, complete_fn=complete)

        if kind == "baseline":
            agent = MemoryAgent(
                router=Router(complete_fn=complete), store=store,
                extractor=extractor, merger=merger,
                retrieve_docs_fn=retrieve, generate_fn=ctx["generate_fn"],
            )
            result = agent.chat(task["question"])
            called = _baseline_tools(result.route)
            steps = 1
            tool_calls = len(called)
            parse_fail_steps = 0
        else:
            registry = ToolRegistry()
            registry.register(RetrieveDocsTool(retrieve_docs_fn=retrieve))
            registry.register(ReadMemoryTool(store=store))
            registry.register(WriteMemoryTool(extractor=extractor, merger=merger))
            registry.register(CalculatorTool())
            agent = ToolAgent(
                complete_fn=complete, registry=registry, generate_fn=ctx["generate_fn"],
                max_steps=ctx["max_steps"], max_tool_calls=ctx["max_tool_calls"],
            )
            result = agent.chat(task["question"])
            called = {s.tool for s in result.trajectory if s.tool and s.error != "duplicate call"}
            steps = len(result.trajectory)
            tool_calls = sum(1 for s in result.trajectory if s.tool and s.error != "duplicate call")
            parse_fail_steps = sum(1 for s in result.trajectory if s.error == "invalid decision JSON")

    gold = set(task.get("gold_chunk_ids", []))
    expected_tools = set(task.get("expected_tools", []))
    precision, recall = tool_precision_recall(called, expected_tools)
    record = {
        "task_id": task.get("task_id", ""),
        "type": task.get("type", ""),
        "success": judge_success(task, result.answer, retrieve.chunks, k),
        "coverage": coverage_at_k(retrieve.chunks, gold, k) if gold else None,
        "tool_precision": precision,
        "tool_recall": recall,
        "tools_called": sorted(called),
        "llm_calls": complete.count,
        "tool_calls": tool_calls,
        "steps": steps,
        "parse_fail_rate": (parse_fail_steps / steps) if steps else 0.0,
        "stop_reason": getattr(result, "stop_reason", None),
        "answer": result.answer,
    }
    return record


def summarize(records: List[Dict[str, Any]]) -> Dict[str, float]:
    coverages = [r["coverage"] for r in records if r["coverage"] is not None]
    return {
        "task_success": mean([1.0 if r["success"] else 0.0 for r in records]),
        "coverage": mean(coverages),
        "tool_precision": mean([r["tool_precision"] for r in records]),
        "tool_recall": mean([r["tool_recall"] for r in records]),
        "llm_calls": mean([r["llm_calls"] for r in records]),
        "tool_calls": mean([r["tool_calls"] for r in records]),
        "steps": mean([r["steps"] for r in records]),
        "parse_fail_rate": mean([r["parse_fail_rate"] for r in records]),
    }


def print_table(k: int, n: int, summaries: Dict[str, Dict[str, float]]) -> None:
    kinds = list(summaries)
    print(f"\n== agent eval ({n} tasks, coverage@{k}) ==")
    header = f"{'metric':<24}" + "".join(f"{kind:>12}" for kind in kinds)
    print(header)
    rows = [
        ("task_success", "task_success"),
        (f"coverage@{k}", "coverage"),
        ("tool_precision", "tool_precision"),
        ("tool_recall", "tool_recall"),
        ("llm_calls/task", "llm_calls"),
        ("tool_calls/task", "tool_calls"),
        ("steps/task", "steps"),
        ("parse_fail_rate", "parse_fail_rate"),
    ]
    for label, key in rows:
        print(f"{label:<24}" + "".join(f"{summaries[kind][key]:>12.3f}" for kind in kinds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval: single-shot baseline vs tool-using agent loop")
    parser.add_argument("--tasks", default="data/eval/agent_tasks_synth.jsonl")
    parser.add_argument("--agent", choices=["both", "baseline", "loop"], default="both")
    parser.add_argument("--coverage-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N tasks")
    parser.add_argument("--out", default="data/eval/agent_eval_report.json")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    # 检索装配参数（默认与 factory 一致）
    parser.add_argument("--vector-store", default="lancedb")
    parser.add_argument("--index-path", default="data/index/faiss.index")
    parser.add_argument("--meta-path", default="data/index/metadatas.json")
    parser.add_argument("--lancedb-uri", default="data/index/lancedb")
    parser.add_argument("--lancedb-table", default="chunks")
    parser.add_argument("--bm25-path", default="data/index/bm25.json")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-distance", type=float, default=0.8)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--no-parent-child", action="store_true")
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks to run", file=sys.stderr)
        sys.exit(1)

    embed_fn = make_embed_fn()
    complete_fn = make_complete_fn()
    dim = len(embed_fn("dimension probe"))
    generator = OpenAICompatibleGenerator()

    def generate_fn(query: str, chunks: List[Dict[str, Any]], user_memory: str) -> Dict[str, Any]:
        if not chunks and not user_memory.strip():
            return {"answer": "没有可用证据。", "sources": []}
        return generator.generate(query=query, retrieved_chunks=chunks, user_memory=user_memory)

    retrieve = _build_doc_retriever(
        args.vector_store, args.index_path, args.meta_path, args.lancedb_uri,
        args.lancedb_table, args.bm25_path, embed_fn,
        args.top_k, args.max_distance, args.rerank, args.rerank_model,
        args.rerank_candidates, 20, 20, 60, not args.no_parent_child, 6,
    )

    ctx = {
        "embed_fn": embed_fn, "complete_fn": complete_fn, "dim": dim,
        "generate_fn": generate_fn, "retrieve": retrieve,
        "max_steps": args.max_steps, "max_tool_calls": args.max_tool_calls,
    }

    kinds = ["baseline", "loop"] if args.agent == "both" else [args.agent]
    all_records: Dict[str, List[Dict[str, Any]]] = {}
    for kind in kinds:
        records = []
        for i, task in enumerate(tasks, start=1):
            print(f"[{kind}] {i}/{len(tasks)} {task.get('task_id', '')} ...", file=sys.stderr)
            records.append(run_task(kind, task, ctx, args.coverage_k))
        all_records[kind] = records

    summaries = {kind: summarize(records) for kind, records in all_records.items()}
    print_table(args.coverage_k, len(tasks), summaries)

    report = {
        "tasks_file": args.tasks,
        "n_tasks": len(tasks),
        "coverage_k": args.coverage_k,
        "summary": summaries,
        "per_task": all_records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
