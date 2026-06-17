# 跑通记忆增强 RAG core（给运行者 / 另一个 CC 会话）

这份指南让你在**真实环境**跑通"带长期记忆的 RAG agent"核心：多轮对话里，助手会从你说的话沉淀长期记忆，并在后续提问时召回。

> 当前分支：`feature/long-term-memory`。请在原仓库目录运行，不要和正在改代码的会话共用同一份工作区同时改文件。

## 1. 环境

```bash
# 建议用虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 跑“记忆 core 对话”最少只需要这些（不需要 torch/transformers/rapidocr）：
pip install openai faiss-cpu numpy python-dotenv rank-bm25 jieba tiktoken pypdf pymupdf

# 或者直接装全量（含 reranker/OCR 依赖，较重）：
# pip install -r requirements.txt
```

## 2. 配置 .env

```bash
cp .env.example .env
```

填入你的 OpenAI 兼容服务（例如智谱）：

```
LLM_PROVIDER=openai_compatible
EMBED_PROVIDER=openai_compatible
LLM_MODEL=glm-4-flash
EMBED_MODEL=embedding-3
OPENAI_COMPAT_API_KEY=你的key
OPENAI_COMPAT_BASE_URL=你的base_url
```

## 3. 先跑单测（确认 core 逻辑健康，无需联网）

```bash
pip install pytest
python -m pytest tests/ -q
# 期望：41 passed
```

## 4.（可选）建文档索引

有文档证据时回答更像 RAG；没有也能跑（纯记忆模式）。

```bash
# 把文件放进 data/uploads/ 后：
python scripts/build_index.py --input-dir data/uploads \
  --index-path data/index/faiss.index --meta-path data/index/metadatas.json \
  --bm25-path data/index/bm25.json --embed-backend openai
```

## 5. 跑对话 core（重点）

```bash
python scripts/chat_demo.py
```

建议这样验证"长期记忆"确实生效：

1. 先**陈述事实**（会被写入记忆）：
   - `我在南洋理工读人工智能硕士，最近在做一个带长期记忆的 RAG 项目`
   - 观察输出末尾的 `↳ memory updated: add:...`
2. **换个话题**问一句无关的，再回来问：
   - `我在读什么专业？` / `我最近在做什么项目？`
   - 观察 `↳ recalled about you:` 是否召回了之前说的事实，回答是否用上了。
3. 输入 `:mem` 查看当前所有已存记忆。
4. 退出再进来 `python scripts/chat_demo.py`，再问一次——记忆应**跨会话保留**（存在 `data/memory/`）。

## 期望看到的现象

- 陈述事实那轮：`↳ memory updated: add:...`
- 后续相关提问：`↳ recalled about you: • ...` 并且回答个性化
- `:mem` 能列出结构化事实
- 重启后记忆仍在（`data/memory/memory.db` + `mem_index.json`）

## 反馈给改代码的会话

把这些贴回来最有用：
- 记忆**该抽没抽 / 抽错 / 重复**的例子
- 召回**该召没召 / 召回不相关**的例子
- 归并**该合并没合并、该更新没更新、误删**的例子
- 任何报错堆栈

这些正好对应后续要做的"记忆评测"维度。
