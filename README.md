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
Personal_RAG/
├── app/
│   ├── loader/
│   ├── chunker/
│   ├── embedder/
│   ├── vectordb/
│   ├── retriever/
│   ├── generator/
│   └── reranker/
├── scripts/
│   ├── build_index.py
│   └── query_demo.py
└── data/
    └── uploads/
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

- 启用 Cross-Encoder 重排（先召回再 rerank）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --top-k 4 --use-rerank --rerank-top-n 20 --rerank-device cpu --show-chunks
```

- 启用 Cross-Encoder 重排（GPU）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --top-k 4 --use-rerank --rerank-top-n 20 --rerank-device cuda --show-chunks
```

- 关闭检索（只测生成接口）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --no-retrieval
```

## 用户指南（CPU / GPU）

1. 启动 Web 界面

```bash
streamlit run web/streamlit_app.py
```

2. 仅 CPU 用户（推荐先用这个验证流程）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --top-k 4 --use-rerank --rerank-top-n 20 --rerank-device cpu --show-chunks
```

3. 有 NVIDIA GPU 用户

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --top-k 4 --use-rerank --rerank-top-n 20 --rerank-device cuda --show-chunks
```

4. 如果 GPU 内核不兼容报错（如 `fmha_cutlass`），先改回 CPU

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --top-k 4 --use-rerank --rerank-top-n 20 --rerank-device cpu
```

## 界面示例

![Streamlit Example](./image.png)

## 示例输出

```text
>python scripts/query_demo.py --query "文档没写的内容是什么？"
=== Answer ===
根据现有资料无法确定具体遗漏的内容。提供的上下文主要涵盖 Faiss 的定义、功能、代码示例和发展历程，但未提及具体应用场景、技术局限性或最新更新细节。
```

```text
>python scripts/query_demo.py --query "什么是faiss"
=== Answer ===
Faiss 是一个专门用于快速处理大规模向量数据的相似度搜索的算法库，主要解决传统暴力搜索在处理大数据时的性能问题。它可以高效地在向量空间中找到相似向量，常用于推荐系统、图像匹配等场景。根据资料，Faiss 是向量数据库的核心引擎，尤其在大模型时代和检索增强生成（RAG）技术中发挥重要作用。[1][2][3]
```

## 常见报错

- `could not open data/index/faiss.index`：先运行建索引命令。
- `No supported files found`：检查输入目录里是否有 `txt/md/pdf`。

## 联系方式

- 邮箱：`yipeng003@e.ntu.edu.sg`
