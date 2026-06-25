"""Ablation: does parent-child context expansion help, and at what window size?

Heading-less fine-grained corpus. Retrieve children, then expand each hit to a
parent window of W consecutive chunks. Measure, per W:
  - answer coverage: does the assembled context contain the gold answer span?
  - context size: average tokens fed to the LLM (the noise / cost of expansion).

W=1 is the child-only baseline. This isolates the recall(coverage)/cost tradeoff
of parent expansion and the effect of the window size.
"""

import json
import os
import statistics
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tiktoken  # noqa: E402

from app.agent.llm import make_embed_fn  # noqa: E402
from app.retriever.hybrid import rrf_fuse  # noqa: E402
from app.vectordb.bm25_store import BM25Store  # noqa: E402
from app.vectordb.faiss_store import FaissStore  # noqa: E402

# Gold answer span per query (a distinctive verbatim substring of the answer).
GOLD = {
    "q001": "五层流水线", "q002": "route_confidence_threshold 设为 0.62", "q003": "进入保守模式",
    "q004": "文档库偏向可追溯资料", "q005": "450 token", "q006": "start_char、end_char",
    "q007": "退化为普通滑窗切分", "q008": "表头会复制到每个被拆分的表格片段中", "q009": "created_at",
    "q010": "source#chunk_id 生成", "q011": "自动降级到 FAISSStore", "q012": "FAISS 适合快速原型",
    "q013": "Reciprocal Rank Fusion", "q014": "rrf_k 为 60", "q015": "把排名转换成",
    "q016": "退化为向量检索", "q017": "BAAI/bge-reranker-base", "q018": "rerank_candidate_k 默认是 20",
    "q019": "精排后仍会执行 source 去重", "q020": "自动跳过精排", "q021": "visibility、state、source 和 updated_at",
    "q022": "preference、skill、location", "q023": "标记为 SUPERSEDED", "q024": "Recall@1、Recall@3 和 MRR@3",
    "q025": "每条记忆证据包含 fact_id", "q026": "我没有在当前知识库中找到足够依据", "q027": "必须优先引用包含具体数值的证据",
    "q028": "模型不能从常识补齐", "q029": "query_id、doc_id 或 source/chunk_id",
    "q030": "文档检索建议报告 Recall@1、Recall@4", "q031": "必须用 metadatas.json 校验",
    "q032": "qrels 可以包含多个相关 chunk", "q033": "统一成 entries 列表", "q034": "保留 image_path",
    "q035": "frame_interval_sec 默认为 5 秒", "q036": "openai/clip-vit-base-patch32",
    "q037": "BAAI/bge-small-zh-v1.5", "q038": "系统应选择 local embedding",
    "q039": "合成评测文档放在 data/uploads_eval", "q040": "不建议把 data/memory/memory.db 提交到公开仓库",
}

INDEX = "data/index_plain"
WINDOWS = [1, 2, 3, 5]


def main():
    store = FaissStore.load(f"{INDEX}/faiss.index", f"{INDEX}/metadatas.json")
    bm25 = BM25Store.load(f"{INDEX}/bm25.json")
    by_src = {}
    for m in store.all_metadatas():
        by_src.setdefault(m["source"], {})[int(m["chunk_id"])] = m.get("text", "")

    embed = make_embed_fn()
    enc = tiktoken.get_encoding("cl100k_base")
    queries = {json.loads(l)["query_id"]: json.loads(l)["query"] for l in open("data/eval/queries_synth.jsonl", encoding="utf-8") if l.strip()}

    def window_context(children, w):
        picked = set()
        for src, cid in children:
            g = cid // w
            for c in range(g * w, (g + 1) * w):
                if c in by_src.get(src, {}):
                    picked.add((src, c))
        return " ".join(by_src[s][c] for s, c in sorted(picked))

    for label, k in [("top1", 1), ("top4", 4)]:
        cov = {w: 0 for w in WINDOWS}
        toks = {w: [] for w in WINDOWS}
        n = 0
        for qid, gold in GOLD.items():
            q = queries.get(qid)
            if not q:
                continue
            n += 1
            vhits = store.search(query_vector=embed(q), top_k=20)
            bhits = bm25.search(query=q, top_k=20)
            fused = rrf_fuse(vhits, bhits, rrf_k=60, top_k=k)
            children = [(h["metadata"]["source"], int(h["metadata"]["chunk_id"])) for h in fused]
            for w in WINDOWS:
                ctx = window_context(children, w)
                if gold in ctx:
                    cov[w] += 1
                toks[w].append(len(enc.encode(ctx)))
        print(f"\n=== retrieve {label} children, expand to parent window W ===  (queries={n})")
        print(f"{'W':>3} | {'answer_coverage':>16} | {'avg_context_tokens':>18}")
        for w in WINDOWS:
            print(f"{w:>3} | {cov[w] / n:>16.3f} | {statistics.mean(toks[w]):>18.0f}")


if __name__ == "__main__":
    main()
