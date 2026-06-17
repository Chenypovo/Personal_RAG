# Synthetic Retrieval Eval

本目录新增一套小型但完整的文档检索合成评测数据，用于快速验证 Personal RAG 的 `Recall@K` 和 `MRR@K`。

## 文件

- 文档库：`data/uploads_eval/`
- 查询文件：`data/eval/queries_synth.jsonl`
- 标注文件：`data/eval/qrels_synth.jsonl`

文档库共 10 篇中文 Markdown，主题覆盖多模态 RAG Agent 的结构化 chunking、向量库、BM25/RRF 混合检索、BGE Reranker、长期记忆、证据约束生成、评测方法、多模态接入和部署配置。

## 规模

- 文档数：10
- 查询数：40
- qrels 行数：46
- 标注粒度：文档级 `source`

部分查询有两个相关文档，用来覆盖对比查询和跨模块综合问题。

## 字段约定

`queries_synth.jsonl` 每行包含：

```json
{"query_id":"q001","query":"自然语言问题","query_image":""}
```

`qrels_synth.jsonl` 每行包含：

```json
{"query_id":"q001","source":"data/uploads_eval/rag_architecture.md","relevance":1}
```

当前 qrels 是建索引前的文档级标注，不包含 `chunk_id`。如果直接用于当前 `scripts/eval_retrieval.py`，脚本会把只有 `source` 的 qrels 解释为 `source#0`。建议建库后读取 `data/index/metadatas.json`，再把文档级 source 展开为真实的 chunk 级 `doc_id` 标注。

## 建库示例

```bash
python scripts/build_index.py \
  --input-dir data/uploads_eval \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json \
  --embed-backend local
```

## 评测示例

chunk 级 qrels 生成后，可运行：

```bash
python scripts/eval_retrieval.py \
  --queries data/eval/queries_synth.jsonl \
  --qrels data/eval/qrels_synth_chunk.jsonl \
  --backend all \
  --ks 1,4 \
  --embed-backend local \
  --index-path data/index_eval/faiss.index \
  --meta-path data/index_eval/metadatas.json \
  --bm25-path data/index_eval/bm25.json
```

建议简历展示指标：文档检索 `Recall@1`、`Recall@4`、`MRR@4`。
