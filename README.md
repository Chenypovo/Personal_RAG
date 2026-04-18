# Personal RAG

一个简洁的个人 RAG 项目，支持文本与多模态检索。  
核心技术栈：Python + Streamlit + FAISS + BM25 + OpenAI-compatible API。

## 功能

- 文件输入：`txt / md / pdf / image / video`
- 文本切块：`chunk_size=700`，`overlap=120`
- 向量检索：FAISS
- 关键词检索：BM25（`jieba` + regex 保护 token + CJK 2-gram fallback）
- 混合检索：FAISS + BM25（RRF 融合）
- 检索评测：`Recall@K`、`MRR@K`

## 安装

```bash
pip install -r requirements.txt
```

复制并配置 `.env`：

```bash
cp .env.example .env
```

## 建库与查询

建库（同时生成 FAISS + BM25）：

```bash
python scripts/build_index.py --input-dir data/uploads --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --embed-backend openai
```

仅向量检索（FAISS）：

```bash
python scripts/query_demo.py --retrieval-backend vector --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --query "总结这个pdf核心内容，用中文" --top-k 4 --show-chunks
```

仅向量检索 + rerank：

```bash
python scripts/query_demo.py --retrieval-backend vector --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --query "总结这个pdf核心内容，用中文" --top-k 4 --candidate-k 20 --use-rerank --rerank-top-n 20 --show-chunks
```

混合检索（FAISS + BM25）：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --vector-k 40 --bm25-k 40 --rrf-k 60 --show-chunks
```

混合检索 + rerank：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --vector-k 40 --bm25-k 40 --rrf-k 60 --use-rerank --rerank-top-n 20 --show-chunks
```

网页端：

```bash
streamlit run web/streamlit_app.py
```

网页端支持在侧边栏勾选 `use_rerank`，并设置 `rerank_model` / `rerank_device` / `rerank_top_n`。

## 检索评测

评测脚本：

- [eval_retrieval.py](/D:/personal_rag/scripts/eval_retrieval.py)

运行示例：

```bash
python scripts/eval_retrieval.py --queries data/eval/queries_example.jsonl --qrels data/eval/qrels_example.jsonl --backend all --ks 1,4 --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --vector-k 40 --bm25-k 40 --rrf-k 60
```
本地运行时请去掉`_example`字样并建议将文档发送给大模型(GPT, Gemini, Qwen等) 让其根据example以及个人文档生成jsonl文件并放入eval。

### 最近一次实验结果（`queries.jsonl` / `qrels.jsonl`，`n_eval_queries=29`，`embed_backend=openai`）

| Backend | Rerank | Recall@1 | MRR@1 | Recall@4 | MRR@4 | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vector | False | 0.3448 | 0.4828 | 0.7414 | 0.6178 | 0.8621 | 0.6305 |
| bm25 | False | 0.3793 | 0.4828 | 0.7931 | 0.6580 | 0.9310 | 0.6693 |
| hybrid | False | 0.4310 | 0.5517 | 0.8621 | 0.7471 | 0.9138 | 0.7510 |
| vector | True | **0.5862** | **0.6897** | 0.8276 | **0.7989** | 0.9138 | **0.8046** |
| bm25 | True | 0.5517 | 0.6552 | **0.8448** | 0.7816 | **0.9310** | 0.7885 |
| hybrid | True | 0.5517 | 0.6552 | **0.8448** | 0.7816 | **0.9310** | 0.7885 |

## 如何人工构建评测集

1. 准备查询文件：`data/eval/queries.jsonl`
   每行一个 JSON，至少包含 `query_id` 和 `query`。  
2. 准备标注文件：`data/eval/qrels.jsonl`
   每行一个 JSON，包含 `query_id` + 相关文档标识。  
   你可以用 `doc_id`，或者用 `source + chunk_id`。  
3. 相关文档标识必须和索引里一致。
   可从 [metadatas.json](/D:/personal_rag/data/index/metadatas.json) 查看真实 `source/chunk_id`。  

## 评测集示例（建议将建库后的metadatas.json与原始文档，以及这两个jsonl文件喂给LLM让其生成评测集）

- [queries_example.jsonl](/D:/personal_rag/data/eval/queries_example.jsonl)
- [qrels_example.jsonl](/D:/personal_rag/data/eval/qrels_example.jsonl)


The *_example files are generic templates and do not contain project-private data.

