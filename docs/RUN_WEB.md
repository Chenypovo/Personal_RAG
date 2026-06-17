# 跑 Web 前端（FastAPI + 自研界面）

在 CLI 对话之外，提供一个有设计感的网页聊天界面，左侧对话、右侧实时展示「长期记忆」。

## 依赖

在已装最小依赖（见 [RUN_CORE.md](RUN_CORE.md)）基础上，再装 Web 依赖：

```bash
pip install fastapi uvicorn
# 想让网页里的文档检索走 BGE 精排，再装：pip install torch transformers
```

## 配置

同 RUN_CORE：`.env` 里填好 `OPENAI_COMPAT_API_KEY` / `OPENAI_COMPAT_BASE_URL`。
记忆持久化在 `data/memory/`，与 CLI（chat_demo）共用同一份记忆。

## 启动

```bash
uvicorn app.api.server:app --reload --port 8000
# 可选：USE_RERANK=1 uvicorn app.api.server:app --port 8000   # 文档检索加 BGE 精排
```

打开 http://localhost:8000

## 接口

- `POST /api/chat`  body `{"message": "..."}` → `{answer, sources, recalled_memories, memory_ops, route}`
- `GET  /api/memories` → 当前所有 active 记忆
- `GET  /` → 网页界面（`webui/`）

## 验证（同 core 剧本）

1. 输入「我在南洋理工读人工智能硕士，最近在做带长期记忆的 RAG」→ 右侧「长期记忆」应新增一条并高亮。
2. 换话题，再问「我在做什么项目？」→ 回答下方出现「想起关于你」chips，且回答个性化。
3. 刷新页面，记忆仍在（跨会话持久化）。
