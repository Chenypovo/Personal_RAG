"""用 LLM 从现有语料合成多步 agent 评测任务（需 API；生成后务必人工抽查）。

沿用现有评测集做法：喂 metadatas.json 里的真实 chunk 给 LLM，让它出
「必须 ≥2 次工具调用」的复合任务，gold_chunk_ids 用真实 source#chunk_id。

用法：
    python scripts/gen_agent_tasks.py --n 24 --out data/eval/agent_tasks_synth.jsonl
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.llm import make_complete_fn  # noqa: E402

_SYSTEM_PROMPT = (
    "你在为一个工具调用 agent 造离线评测任务。agent 有四个工具：retrieve_docs（文档检索）、"
    "read_memory（读用户记忆）、write_memory（写用户记忆）、calculator（算术）。\n"
    "根据给出的文档片段，生成指定类型的一个任务，要求完成任务必须 ≥2 次工具调用。\n"
    "- multihop：问题必须同时问到两个不同片段里的信息（需要两次不同子查询的 retrieve_docs）。\n"
    "- calc：问题要求先从片段里找到数值，再做一步算术（retrieve_docs + calculator）；"
    "expected_value 是最终数值。\n"
    "- memory：先给用户一条个人事实 setup_memories，问题需要同时用到该事实和文档内容"
    "（read_memory + retrieve_docs）；answer_contains 是正确回答必然包含的词。\n"
    "gold_chunk_ids 只能用给出的真实 id。问题用中文，不要在问题里透露 chunk id。\n"
    '只返回 JSON：{"question": str, "gold_chunk_ids": [str], "expected_tools": [str], '
    '"expected_value": number|null, "setup_memories": [str], "answer_contains": [str]}'
)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-step agent eval tasks with an LLM")
    parser.add_argument("--meta-path", default="data/index/metadatas.json")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--out", default="data/eval/agent_tasks_synth.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metas = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    by_source = defaultdict(list)
    for m in metas:
        if str(m.get("text", "")).strip():
            by_source[str(m.get("source", ""))].append(m)

    sources = sorted(by_source)
    if len(sources) < 2:
        print("need at least 2 sources", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    complete_fn = make_complete_fn()
    # 三类循环生成：multihop 需要两个不同来源的片段，calc/memory 一个来源即可
    types = ["multihop", "calc", "memory"]
    tasks = []
    attempts = 0
    while len(tasks) < args.n and attempts < args.n * 4:
        attempts += 1
        task_type = types[len(tasks) % len(types)]
        if task_type == "multihop":
            s1, s2 = rng.sample(sources, 2)
            picked = [rng.choice(by_source[s1]), rng.choice(by_source[s2])]
        else:
            s1 = rng.choice(sources)
            picked = rng.sample(by_source[s1], min(2, len(by_source[s1])))
        chunk_lines = [
            f"[{m.get('source')}#{m.get('chunk_id')}]\n{str(m.get('text', ''))[:800]}"
            for m in picked
        ]
        user_prompt = f"任务类型: {task_type}\n\n文档片段:\n\n" + "\n\n---\n\n".join(chunk_lines)
        try:
            raw = complete_fn(_SYSTEM_PROMPT, user_prompt)
            obj = json.loads(_strip_code_fences(raw))
            question = str(obj.get("question", "")).strip()
            gold = [g for g in obj.get("gold_chunk_ids", [])
                    if any(g == f"{m.get('source')}#{m.get('chunk_id')}" for m in picked)]
            if not question or not gold:
                continue
            task = {
                "task_id": f"{task_type[:4]}_{len(tasks):03d}",
                "type": task_type,
                "question": question,
                "gold_chunk_ids": gold,
                "expected_tools": obj.get("expected_tools") or (
                    ["retrieve_docs", "calculator"] if task_type == "calc"
                    else ["read_memory", "retrieve_docs"] if task_type == "memory"
                    else ["retrieve_docs"]
                ),
            }
            if task_type == "calc":
                if obj.get("expected_value") is None:
                    continue
                task["expected_value"] = float(obj["expected_value"])
            if task_type == "memory":
                setup = [str(x) for x in obj.get("setup_memories", []) if str(x).strip()]
                needles = [str(x) for x in obj.get("answer_contains", []) if str(x).strip()]
                if not setup or not needles:
                    continue
                task["setup_memories"] = setup
                task["answer_contains"] = needles
            tasks.append(task)
            print(f"ok {task['task_id']}: {question[:60]}", file=sys.stderr)
        except Exception as e:
            print(f"skip ({e})", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {out}（请人工抽查 gold 与题面后再用于评测）")


if __name__ == "__main__":
    main()
