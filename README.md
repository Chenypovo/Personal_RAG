# Personal RAG + 长期记忆 Agent

轻量级个人 RAG + 长期记忆 Agent：多模态文档检索增强问答，跨会话沉淀长期记忆，由路由层每轮自主决定「查文档 / 调记忆 / 写记忆」。

**技术栈**：Python · LanceDB/FAISS · BM25 · BGE Reranker · OpenAI-compatible / 本地 HF embedding · FastAPI

## 功能

- **多模态接入**：`txt / md / pdf / image(OCR) / video`
- **结构感知切块**：词元感知（tiktoken）+ 段落/句子边界；chunk 保留**标题层级**、**页码**、**块类型**（heading/list/table/code/paragraph）、切分边界、多模态路径等 metadata
- **向量库**：LanceDB 统一行存（id/vector/text/metadata），与 FAISS（IndexFlatL2）两套后端结果等价；Agent 默认走 LanceDB
- **混合检索**：向量 + BM25 经 RRF 融合 → BGE Reranker 精排
- **parent-child 召回**：索引小子块保精度，命中后按**标题章节**扩展回父块（无标题则相邻窗口），父块去重
- **受控生成**：证据约束回答、按 source/chunk 展示依据，证据不足时拒答
- **长期记忆**：从对话抽取事实三元组 → LLM 判定 add/update/delete 归并 → SQLite + 向量索引持久化；提问时召回注入
- **对话 Agent**：路由决策 + 记忆增强生成
- **Web**：FastAPI 后端 + 自研前端（`webui/`），另提供 Streamlit
- **评测**：检索 & 记忆的 `Recall@K` / `MRR@K`

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 LLM key；embedding 可选远端或本地
```

`.env` 关键项（embedding 二选一）：

```bash
LLM_PROVIDER=openai_compatible      # 回答/抽取/路由走 API
OPENAI_COMPAT_API_KEY=...
OPENAI_COMPAT_BASE_URL=...
EMBED_PROVIDER=local                # 本地：BAAI/bge-small-zh-v1.5（或 openai_compatible）
EMBED_MODEL=BAAI/bge-small-zh-v1.5
```

建库 → 跑应用：

```bash
# 1) 把文档放进 data/uploads，建索引（向量 + BM25）
python scripts/build_index.py --input-dir data/uploads --embed-backend local

# 2) Web 端（聊天 + 实时展示沉淀的记忆）
uvicorn app.api.server:app --port 8000      # 浏览器打开 http://127.0.0.1:8000

# 或命令行多轮对话
python scripts/chat_demo.py
```

检索单测：`python scripts/query_demo.py --retrieval-backend hybrid --rerank --query "..." --show-chunks`
（`--vector-store faiss|lancedb`、`--embed-backend local|openai|clip` 等参数见 `--help`）

## 评测

```bash
# 检索 — 无精排基线（→ data/eval/retrieval_synth_report.json）
python scripts/eval_retrieval.py --vector-store lancedb --lancedb-uri data/index_eval/lancedb \
  --bm25-path data/index_eval/bm25.json --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth.jsonl --backend all --ks 1,4 --embed-backend local --no-strict

# 检索 — 完整链路，加 --rerank（→ data/eval/retrieval_synth_rerank_report.json）
python scripts/eval_retrieval.py --vector-store lancedb --lancedb-uri data/index_eval/lancedb \
  --bm25-path data/index_eval/bm25.json --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth.jsonl --backend all --ks 1,4 --embed-backend local --no-strict --rerank

# 记忆召回（→ data/eval/memory_synth_report.json）
python scripts/eval_memory.py --facts data/eval/memory_facts_synth.jsonl \
  --queries data/eval/memory_queries_synth.jsonl --qrels data/eval/memory_qrels_synth.jsonl --ks 1,3
```

实验结果（合成语料 40 查询，本地 bge-small，LanceDB；FAISS 数字逐位相同）。
左两列＝无精排基线（`retrieval_synth_report.json`），右两列＝加 `--rerank` 完整链路（`retrieval_synth_rerank_report.json`）：

| Backend | 基线 Recall@1 | 基线 MRR@4 | +精排 Recall@1 | +精排 MRR@4 |
| --- | ---: | ---: | ---: | ---: |
| 向量 | 0.3125 | 0.5437 | 0.4750 | 0.6792 |
| BM25 | 0.4500 | 0.6417 | 0.5000 | 0.7042 |
| **混合** | 0.4000 | 0.6604 | **0.5000** | **0.7042** |

- **`eval_retrieval.py` 默认不精排；加 `--rerank` 才走 BGE 精排。** 上表两组分别对应两份报告文件。
- 完整链路（混合 + BGE 精排）：**Recall@1=0.50 · Recall@4=0.89 · MRR@4=0.70**；精排把 top-1 命中从 0.40 提到 0.50。
- 记忆召回（`memory_synth_report.json`，31 事实 / 20 查询带干扰，**与 reranker 无关**）：**Recall@1=0.90 · Recall@3=0.95 · MRR@1=0.90**

> 评测集为自建标注集（`data/eval/`）。可把建库后的 `metadatas.json` + 原文喂给 LLM 自动生成 `queries/qrels`。

## 结构

`app/`：`loader`（多模态+OCR）· `chunker`（结构感知切块）· `embedder` · `vectordb`（LanceDB/FAISS/BM25）· `retriever`（含 parent-child）· `reranker` · `generator` · `memory`（抽取/归并/召回/存储）· `agent`（路由+编排）· `api`（FastAPI）· `eval`
