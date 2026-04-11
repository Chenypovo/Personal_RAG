# Personal RAG

一个简单的个人 RAG 项目。

## 支持内容

- 文档格式：`txt` / `md` / `pdf`
- 切块：`chunk_size=700`，`overlap=120`
- 向量：Embedding 接口
- 检索：FAISS（Top-K，默认 4）
- 生成：基于检索结果回答，并返回 chunk 来源

## 项目结构

```text
app/
  loader/
  chunker/
  embedder/
  vectordb/
  retriever/
  generator/
scripts/
  build_index.py
  query_demo.py
data/
  uploads/
```

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

填写 `.env`：

```env
LLM_PROVIDER=openai_compatible
EMBED_PROVIDER=openai_compatible
LLM_MODEL=
EMBED_MODEL=
OPENAI_COMPAT_API_KEY=your_api_key
OPENAI_COMPAT_BASE_URL=
```

3. 放入文档

把 `txt/md/pdf` 文件放到 `data/uploads/`。

4. 建索引

```bash
python scripts/build_index.py --input-dir data/uploads
```

5. 查询

```bash
python scripts/query_demo.py --query "你的问题"
```

## 示例命令

- 显示检索到的 chunk 文本

```bash
python scripts/query_demo.py --query "FAISS 的作用是什么？" --show-chunks
```

- 关闭检索（只测生成接口）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --no-retrieval
```

## 常见报错

- `could not open data/index/faiss.index`：先运行建索引命令。
- `No supported files found`：检查输入目录里是否有 `txt/md/pdf`。
