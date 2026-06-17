# Personal RAG + 长期记忆 Agent

一个轻量级个人 RAG + 长期记忆 Agent：多模态文档检索增强问答，跨会话沉淀长期记忆，并由路由层自主决定「查文档 / 调记忆 / 写记忆」。  
核心技术栈：Python + LanceDB/FAISS + BM25 + BGE Reranker + OpenAI-compatible / 本地 HF embedding + FastAPI + Streamlit。

## 功能

- 文件输入：`txt / md / pdf / image(OCR) / video`
- 文本切块：`chunk_size=700`，`overlap=120`
- 向量库：LanceDB（默认）或 FAISS
- 关键词检索：BM25（`jieba` + regex 保护 token + CJK 2-gram fallback）
- 混合检索：向量 + BM25（RRF 融合），可选 BGE Reranker 精排
- 图片 OCR：`rapidocr` 注入，缺失/失败自动降级
- 长期记忆：从对话/笔记抽取事实三元组 → add/update/delete 归并 → SQLite 持久化；提问时召回注入「关于你」上下文
- 对话 Agent：路由决策（查文档 / 调记忆 / 写记忆）+ 证据约束生成、证据不足拒答
- Web：FastAPI 后端 + 自研前端（`webui/`），或 Streamlit
- 评测：检索 & 记忆 `Recall@K` / `MRR@K`，归并决策准确率

## 安装

```bash
pip install -r requirements.txt
```

复制并配置 `.env`：

```bash
cp .env.example .env
```

## 建库与查询

`.env` 支持两种文本 embedding：

```bash
# 远端 OpenAI-compatible
EMBED_PROVIDER=openai_compatible
EMBED_MODEL=embedding-3

# 本地 Hugging Face
# EMBED_PROVIDER=local
# EMBED_MODEL=BAAI/bge-small-zh-v1.5
# EMBED_DEVICE=cpu
```

建库（同时生成 FAISS + BM25）：

```bash
python scripts/build_index.py --input-dir data/uploads --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --embed-backend openai
```

仅向量检索（FAISS）：

```bash
python scripts/query_demo.py --retrieval-backend vector --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --query "总结这个pdf核心内容，用中文" --top-k 4 --show-chunks
```

混合检索（FAISS + BM25）：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --vector-k 40 --bm25-k 40 --rrf-k 60 --show-chunks
```

混合检索后再精排：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --rerank --rerank-candidates 40 --show-chunks
```

文档检索网页端（Streamlit）：

```bash
streamlit run web/streamlit_app.py
```

## 对话 Agent（长期记忆）

命令行多轮对话（每轮自动召回记忆、按需检索文档、并从你的话里沉淀新记忆）：

```bash
python scripts/chat_demo.py
```

FastAPI 后端 + 自研 Web 前端（聊天，并展示召回到的记忆与本轮记忆写入操作）：

```bash
uvicorn app.api.server:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

记忆持久化在 `data/memory/`（SQLite 主存 + 向量索引），跨会话保留。

## 真机自检（OCR / reranker 真模型）

```bash
python scripts/verify_real.py    # 首次会下载 BGE 模型(~1.1GB) 与 rapidocr 模型
```

## 检索评测

评测脚本：

- [eval_retrieval.py](scripts/eval_retrieval.py)

运行示例：

```bash
python scripts/eval_retrieval.py --queries data/eval/queries_example.jsonl --qrels data/eval/qrels_example.jsonl --backend all --ks 1,4 --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --vector-k 40 --bm25-k 40 --rrf-k 60
```

记忆评测示例：

```bash
python scripts/eval_memory.py --facts data/eval/memory_facts_example.jsonl --queries data/eval/memory_queries_example.jsonl --qrels data/eval/memory_qrels_example.jsonl --ks 1,3,5
```

### 最近一次实验结果（基于我本地上传的论文）

| Backend | Recall@1 | MRR@1 | Recall@4 | MRR@4 |
| --- | ---: | ---: | ---: | ---: |
| vector | 0.0882 | 0.1176 | 0.1471 | 0.1373 |
| bm25 | 0.4706 | 0.5294 | 0.9412 | 0.7010 |
| hybrid | **0.5294** | **0.6471** | 0.9118 | **0.7794** |

## 如何人工构建评测集

1. 准备查询文件：`data/eval/queries.jsonl`
   每行一个 JSON，至少包含 `query_id` 和 `query`。  
2. 准备标注文件：`data/eval/qrels.jsonl`
   每行一个 JSON，包含 `query_id` + 相关文档标识。  
   你可以用 `doc_id`，或者用 `source + chunk_id`。  
3. 相关文档标识必须和索引里一致。
   可从 [metadatas.json](data/index/metadatas.json) 查看真实 `source/chunk_id`。  

## 评测集示例（建议将建库后的metadatas.json与原始文档，以及这两个jsonl文件喂给LLM让其生成评测集）

- [queries_example.jsonl](data/eval/queries_example.jsonl)
- [qrels_example.jsonl](data/eval/qrels_example.jsonl)
