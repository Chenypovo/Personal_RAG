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

混合检索（FAISS + BM25）：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --vector-k 40 --bm25-k 40 --rrf-k 60 --show-chunks
```

网页端：

```bash
streamlit run web/streamlit_app.py
```

## 检索评测

评测脚本：

- [eval_retrieval.py](/D:/personal_rag/scripts/eval_retrieval.py)

运行示例：

```bash
python scripts/eval_retrieval.py --queries data/eval/queries_example.jsonl --qrels data/eval/qrels_example.jsonl --backend all --ks 1,4 --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --vector-k 40 --bm25-k 40 --rrf-k 60
```

### 最近一次实验结果（你提供的数据）

| Backend | Recall@1 | MRR@1 | Recall@4 | MRR@4 |
| --- | ---: | ---: | ---: | ---: |
| vector | 0.0882 | 0.1176 | 0.1471 | 0.1373 |
| bm25 | 0.4706 | 0.5294 | 0.9412 | 0.7010 |
| hybrid | **0.5294** | **0.6471** | 0.9118 | **0.7794** |

## 如何人工构建评测集（最小流程）

1. 准备查询文件：`data/eval/queries.jsonl`
   每行一个 JSON，至少包含 `query_id` 和 `query`。  
2. 准备标注文件：`data/eval/qrels.jsonl`
   每行一个 JSON，包含 `query_id` + 相关文档标识。  
   你可以用 `doc_id`，或者用 `source + chunk_id`。  
3. 相关文档标识必须和索引里一致。
   可从 [metadatas.json](/D:/personal_rag/data/index/metadatas.json) 查看真实 `source/chunk_id`。  

## 评测集示例（可复制模板）

- [queries_example.jsonl](/D:/personal_rag/data/eval/queries_example.jsonl)
- [qrels_example.jsonl](/D:/personal_rag/data/eval/qrels_example.jsonl)

## 隐私与提交

- `data/uploads/*.pdf` 默认忽略，避免私人 PDF 误提交。
- `data/eval/*.jsonl` 默认忽略，仅允许提交 `*_example.jsonl` 模板文件。
