# Personal RAG

一个简单的个人 RAG 项目，支持文本与多模态检索，适合隐私文件检索。

## 功能

- 文档加载：`txt / md / pdf / image / video`
- PDF OCR 回退：扫描版/图片版 PDF 会自动尝试 OCR 提取文本
- 文本切块：`chunk_size=700`，`overlap=120`
- 向量检索：`FAISS`
- 检索模式：
  - `openai`：文本 embedding
  - `clip`：文本 + 图片同空间检索（视频先抽帧）

## 快速开始

```bash
pip install -r requirements.txt
```

复制并填写 `.env`：

```bash
cp .env.example .env
```

把数据放到 `data/uploads/` 后执行。

## 模型说明

- 聊天模型（生成回答）
  - 来源：`.env` 中 `LLM_MODEL`
  - 默认值：`glm-4-flash`
  - 调用方式：OpenAI-compatible Chat Completions

- 文本 Embedding 模型（`openai` 检索模式）
  - 来源：`.env` 中 `EMBED_MODEL`
  - 默认值：`embedding-3`
  - 调用方式：OpenAI-compatible Embeddings

- 多模态 Embedding 模型（`clip` 检索模式）
  - 模型：`openai/clip-vit-base-patch32`
  - 用途：文本与图片同向量空间检索（视频先抽帧再检索）

- Reranker（可选模块）
  - 模型：`BAAI/bge-reranker-base`
  - 代码位置：`app/reranker/bge_reranker.py`
  - 说明：当前默认流程未开启，可按需接入到检索后重排

## 常用命令

1. 文本建库（OpenAI embedding）

```bash
python scripts/build_index.py --input-dir data/uploads --embed-backend openai
```

2. 多模态建库（CLIP，CPU）

```bash
python scripts/build_index.py --input-dir data/uploads --embed-backend clip --clip-device cpu
```

3. 文本问答（OpenAI）

```bash
python scripts/query_demo.py --query "什么是 FAISS？" --embed-backend openai
```

4. 图片检索问答（CLIP，CPU）

```bash
python scripts/query_demo.py --query-image "data/uploads/cat.png" --query "这张图描述了什么？" --embed-backend clip --clip-device cpu
```

5. 严格检索（不补齐 top-k）+ 打印召回内容

```bash
python scripts/query_demo.py --query-image "data/uploads/cat.png" --query "这张图是什么？" --embed-backend clip --clip-device cpu --top-k 4 --candidate-k 50 --max-distance 0.5 --strict --show-chunks
```

6. Streamlit 页面（上传文本/图片/视频并检索）

```bash
streamlit run web/streamlit_app.py
```

## 网页版推荐用法

- 推荐优先使用网页版，操作更直观。
- 上传文件后，先点 `Save Uploads`，再点 `Build Index`。
- 如果显示没有结果：
- 先把 `top_k` 调高（如果你启用了 rerank，这里指 rerank 阶段保留结果用的 `top_k`）。
- 再尝试关闭 `strict_mode`（等价于取消严格过滤，通常能解决）。
- 如果希望回答是中文，直接在 query 里写明：`请用中文回答`。
