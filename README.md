# Personal RAG

一个简单可跑的个人 RAG（文本 + 多模态）项目。  
核心栈：Python + Streamlit + FAISS + OpenAI-compatible API + BM25。

## 功能

- 文件上传与解析：`txt / md / pdf / image / video`
- 文本切块：`chunk_size=700`，`overlap=120`
- 向量检索：FAISS
- 关键词检索：BM25（`jieba` + 规则 token + 中文 2-gram fallback）
- 混合检索：FAISS + BM25（RRF 融合）
- 生成回答：OpenAI-compatible chat model
- 输出引用：基于检索 chunk 的 `source/chunk_id`

## 安装

```bash
pip install -r requirements.txt
```

复制并配置环境变量：

```bash
cp .env.example .env
```

## 使用流程（推荐）

1. 把文件放到 `data/uploads/`
2. 先建库（FAISS + BM25）
3. 再查询（vector 或 hybrid 对比）

## 建库命令

```bash
python scripts/build_index.py --input-dir data/uploads --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --embed-backend openai
```

## 查询命令

仅向量检索（FAISS）：

```bash
python scripts/query_demo.py --retrieval-backend vector --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --query "总结这个pdf核心内容，用中文" --top-k 4 --show-chunks
```

混合检索（FAISS + BM25）：

```bash
python scripts/query_demo.py --retrieval-backend hybrid --embed-backend openai --index-path data/index/faiss.index --meta-path data/index/metadatas.json --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --vector-k 40 --bm25-k 40 --rrf-k 60 --show-chunks
```

仅 BM25 baseline：

```bash
python scripts/query_demo.py --retrieval-backend bm25 --bm25-path data/index/bm25.json --query "总结这个pdf核心内容，用中文" --top-k 4 --show-chunks
```

## Streamlit

```bash
streamlit run web/streamlit_app.py
```

页面内建议流程：

- 上传后先点 `Save Uploads`，再点 `Build Index`
- 查询阶段优先用 `hybrid`
- 如果结果为空，先提高 `top_k/candidate_k`，再关闭 `strict_mode`

## 隐私说明

`data/uploads` 是本地目录，建议放个人文件。  
仓库默认忽略 `data/uploads/*.pdf`，避免误提交私人 PDF。
