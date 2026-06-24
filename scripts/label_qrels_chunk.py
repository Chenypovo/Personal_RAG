"""Refine document-level qrels to chunk-level with an LLM judge.

For each (query, gold source) pair, show the LLM every chunk of that source and
ask which chunk(s) actually contain the answer. Output chunk-level qrels
(``doc_id = source#chunk_id``). Retrieval uses embeddings, so labeling with the
chat LLM is not circular. Single-chunk docs are passed through unchanged.

    python scripts/label_qrels_chunk.py \
        --queries data/eval/queries_synth.jsonl \
        --qrels data/eval/qrels_synth.jsonl \
        --meta-path data/index_eval/metadatas.json \
        --output data/eval/qrels_synth_chunk.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.agent.llm import make_complete_fn  # noqa: E402

_SYS = (
    "你是检索评测标注员。给定一个查询和同一篇文档的若干编号 chunk，判断哪些 chunk 的"
    "正文真正包含回答该查询所需的关键信息（数值、定义、规则等）。只选真正含答案的，"
    '可多选但要克制。只返回 JSON：{"chunk_ids": [整数,...]}。'
)


def _load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--meta-path", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    metas = json.load(open(args.meta_path, encoding="utf-8"))
    by_src = {}
    for m in metas:
        by_src.setdefault(m["source"], {})[int(m["chunk_id"])] = m.get("text", "")

    queries = {r["query_id"]: r["query"] for r in _load_jsonl(args.queries)}
    qrels = _load_jsonl(args.qrels)
    complete = make_complete_fn()

    out = []
    for r in qrels:
        qid, src = r["query_id"], r["source"]
        chunks = by_src.get(src, {})
        if not chunks:
            continue
        if len(chunks) == 1:
            cid = next(iter(chunks))
            out.append({"query_id": qid, "doc_id": f"{src}#{cid}", "relevance": 1})
            continue

        listing = "\n\n".join(f"[chunk {cid}]\n{chunks[cid]}" for cid in sorted(chunks))
        prompt = f"查询：{queries.get(qid, '')}\n\n候选 chunks：\n{listing}"
        try:
            raw = complete(_SYS, prompt).strip().strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            ids = [int(x) for x in json.loads(raw).get("chunk_ids", []) if int(x) in chunks]
        except Exception:
            ids = []
        if not ids:
            ids = [min(chunks)]  # conservative fallback: first chunk
        for cid in ids:
            out.append({"query_id": qid, "doc_id": f"{src}#{cid}", "relevance": 1})
        print(f"  {qid} [{src.split('/')[-1]}] -> chunks {ids}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(out)} chunk-level qrels -> {args.output}")


if __name__ == "__main__":
    main()
