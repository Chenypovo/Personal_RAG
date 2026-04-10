# Personal RAG（Python + Streamlit + FAISS + OpenAI-Compatible API）

一个最小可用的个人 RAG 项目，支持本地文档入库与问答检索，当前默认走 OpenAI 兼容接口（可配置为 ZAI/OpenRouter 等兼容服务）。

## 功能清单

- 支持文档类型：`txt` / `md` / `pdf`
- 文本切块：`chunk_size=700`，`overlap=120`
- Embedding：通过 OpenAI-Compatible `embeddings` 接口
- 向量库：本地 `FAISS`
- 检索：Top-K（默认 `top_k=4`）
- 生成：基于检索上下文回答，并返回 chunk 级证据来源
- CLI 已可用，Streamlit 页面预留在 `web/streamlit_app.py`

## 目录结构

```text
app/
  loader/       # 文档加载（txt/md/pdf）
  chunker/      # 切块
  embedder/     # embedding 客户端
  vectordb/     # FAISS 存取
  retriever/    # 检索器
  generator/    # 生成器
scripts/
  build_index.py   # 建索引
  query_demo.py    # 查询问答
data/
  uploads/      # 待入库文档
  index/        # 生成的索引（默认运行时产物）
```

## 环境准备

1. Python 3.10+
2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置环境变量（从示例复制）：

```bash
cp .env.example .env
```

编辑 `.env`，填入你自己的 key：

```env
LLM_PROVIDER=openai_compatible
EMBED_PROVIDER=openai_compatible

LLM_MODEL=glm-4-flash
EMBED_MODEL=embedding-3

OPENAI_COMPAT_API_KEY=your_api_key
OPENAI_COMPAT_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

## 快速开始

### 1) 准备文档

把你的 `txt/md/pdf` 文件放到 `data/uploads/` 下。

### 2) 建立向量索引

```bash
python scripts/build_index.py --input-dir data/uploads --index-path data/index/faiss.index --meta-path data/index/metadatas.json
```

### 3) 检索问答（RAG）

```bash
python scripts/query_demo.py --query "什么是 FAISS？"
```

### 4) 打印检索到的 chunk 原文

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --show-chunks
```

### 5) 关闭检索，仅测试生成接口

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --no-retrieval
```

## 示例

### 示例一：正常 RAG 查询

```bash
python scripts/query_demo.py --query "FAISS 适合解决什么问题？"
```

预期输出包含：

- `=== Answer ===`：模型回答
- `=== Retrieved Sources ===`：chunk 证据（例如 `Chunk 0, xxx.txt`）

### 示例二：查看证据文本

```bash
python scripts/query_demo.py --query "FAISS 的核心能力有哪些？" --show-chunks
```

预期输出额外包含：

- `=== Retrieved Chunks ===`：每个检索 chunk 的原文

## 常见问题

- 报错 `could not open data/index/faiss.index`  
  说明还未建库，先运行 `scripts/build_index.py`。

- 报错 `No supported files found`  
  检查 `--input-dir` 路径下是否有 `txt/md/pdf` 文件。

## 后续建议

- 在 `web/streamlit_app.py` 接入上传、建库、问答页面
- 加入多文件来源过滤和重排序（reranker）
- 支持持久化会话与历史问答
