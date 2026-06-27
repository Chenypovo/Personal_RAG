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
- **长期记忆**：从对话抽取事实三元组 → LLM 判定 add/update/delete 归并 → SQLite（事实）+ 独立向量索引（embedding）持久化；提问时按 embedding cosine 召回注入。**防误判（规则化判定）**：UPDATE 需**同槽位**（fact_object 一致）+ cosine ≥ 0.8 → 非销毁 supersede（旧置 SUPERSEDED 可恢复），否则降级 ADD；DELETE 更危险——需**矛盾证据**（带 content）+ 同槽位 + 更高门槛 0.9，否则保留旧事实。把 add/update 做成确定性判定，只把"矛盾型删除"留给 LLM
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
# 检索（chunk 级 qrels，标到含答案的 chunk；加 --rerank 走完整链路）
python scripts/eval_retrieval.py --vector-store lancedb --lancedb-uri data/index_eval/lancedb \
  --bm25-path data/index_eval/bm25.json --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl --backend all --ks 1,4 --embed-backend local --no-strict [--rerank]

# 记忆召回
python scripts/eval_memory.py --facts data/eval/memory_facts_synth.jsonl \
  --queries data/eval/memory_queries_synth.jsonl --qrels data/eval/memory_qrels_synth.jsonl --ks 1,3
```

实验结果（合成语料 40 查询，本地 bge-small，LanceDB；FAISS 数字逐位相同）。
左两列＝无精排基线，右两列＝加 `--rerank` 完整链路：

| Backend | 基线 Recall@1 | 基线 MRR@4 | +精排 Recall@1 | +精排 MRR@4 |
| --- | ---: | ---: | ---: | ---: |
| 向量 | 0.5125 | 0.6896 | 0.7625 | 0.8792 |
| BM25 | 0.7375 | 0.8521 | 0.7875 | 0.9042 |
| **混合** | 0.6000 | 0.8042 | **0.7875** | **0.9042** |

- **完整链路（混合 + BGE 精排）：Recall@1=0.79 · Recall@4=0.975 · MRR@4=0.90**；混合无精排时 Recall@4=1.00。
- **`eval_retrieval.py` 默认不精排；加 `--rerank` 才走 BGE 精排。**
- **qrels 标到 chunk 级**（含答案的 chunk）。若只标到文档级（脚本会塌成 `#0`），会把"召回到正确文档的其它 chunk"误判为错，**低估** MRR/Recall@1（文档级会让混合 MRR@4 从 0.90 假性掉到 0.66）。

### parent-child 父块窗口消融（`scripts/eval_parent_window.py`）

heading-less 细粒度语料（104 chunk），命中后扩展到 W 个连续子块的父块，测**答案覆盖率**与**上下文 token**：

| W | 覆盖率(top-4) | 上下文 tokens | 说明 |
| ---: | ---: | ---: | --- |
| 1（child-only） | 0.900 | 349 | 基线 |
| 2 | 0.900 | 612 | 零增益、翻倍 token |
| **3** | **0.975** | 891 | 覆盖率拐点 |
| 5 | 0.975 | 1143 | 不再涨、纯增成本 |

结论：**W=3 是覆盖率/成本的拐点**（W=2 无增益、W=5 仅增 token），故无标题文档默认窗口取 3；有标题文档则按章节边界分组。

父块大小统一**封顶**（`max_parent_chunks`，默认 6，居中命中）：窗口模式天然 ≤W，章节模式也不会因某节过大而灌爆 LLM 上下文。
- 完整链路（混合 + BGE 精排）：**Recall@1=0.50 · Recall@4=0.89 · MRR@4=0.70**；精排把 top-1 命中从 0.40 提到 0.50。
- 记忆召回（`memory_synth_report.json`，31 事实 / 20 查询带干扰，**与 reranker 无关**）：**Recall@1=0.90 · Recall@3=0.95 · MRR@1=0.90**

> 评测集为自建标注集（`data/eval/`）。可把建库后的 `metadatas.json` + 原文喂给 LLM 自动生成 `queries/qrels`。

## 结构

`app/`：`loader`（多模态+OCR）· `chunker`（结构感知切块）· `embedder` · `vectordb`（LanceDB/FAISS/BM25）· `retriever`（含 parent-child）· `reranker` · `generator` · `memory`（抽取/归并/召回/存储）· `agent`（路由+编排）· `api`（FastAPI）· `eval`
