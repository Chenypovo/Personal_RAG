# 设计：带长期记忆的多模态检索增强问答 Agent

日期：2026-06-17
状态：已批准（用户确认 3 项关键决策）

## 目标

把现有无状态文档 RAG（Personal_RAG）升级为一个**带长期记忆的个人 AI Agent**：

1. 能从**对话**和**笔记**中沉淀结构化长期记忆（事实三元组），跨会话保留。
2. 提问时把相关记忆作为"关于你的已知信息"注入生成，做到个性化回答。
3. 一个轻量**路由层**让每轮自主决定：查文档 / 调记忆 / 写记忆——使其名副其实是 agent。
4. 配套**检索 + 记忆的统一评测**。

并把简历里此前"声称了但未生效"的两点做实：**图片 OCR** 与 **BGE Reranker 精排接入主链路**。

## 关键决策（用户确认）

- 记忆用法：**单独注入"关于你"上下文块**（与文档证据分开），不与文档 RRF 混合。
- Agent：**新增真·路由决策层**撑起 "Agent" 命名。
- 范围：**全做**（记忆子系统 + 记忆评测 + OCR + reranker 接入）。

## 设计原则

- 思想借自脱敏后的 `mem/` 笔记（事实三元组、召回旧记忆、LLM 归并 add/update/delete），**全新实现**，不引用任何外部代码/prompt/命名。
- 本地、单用户、不上线：无 MySQL/Milvus/消息队列/灰度。主存用 SQLite，向量索引复用现有 `FaissStore`。
- 复用现有模块风格（dataclass、OpenAI 兼容 client、JSON/文件持久化），小而专的单元，可独立测试。

## 架构

```
                 ┌─────────────── Agent 路由层 (app/agent/router.py) ───────────────┐
用户一轮输入 ──▶ │  LLM 决策：需要 {检索文档?  召回记忆?  写记忆?}                    │
                 └───┬───────────────┬────────────────────────┬─────────────────────┘
                     ▼               ▼                         ▼
            文档检索(已有)      记忆召回(新)              记忆写入(新)
         hybrid+reranker     memory.recall          extractor→merger→store
                     │               │                         │
                     └──────┬────────┘                         │
                            ▼                                   ▼
                  生成器(已有, 改造)                     记忆主存(SQLite)
        证据块 + "关于你"记忆块 → LLM 证据约束生成        + 记忆向量索引(FAISS)
```

## 组件与接口

### 1. 记忆数据层 `app/memory/`

**`models.py`** — 记忆事实记录：
```python
@dataclass
class MemoryFact:
    id: str                  # uuid
    fact_object: str         # 宾语，可空
    fact_content: str        # 正文（含规范化时间），核心
    visibility: str          # "PUBLIC" | "PRIVATE"（存储用，本地不做受众过滤）
    state: str               # "ACTIVE" | "DELETED"
    source: str              # "chat" | "note"
    created_at: str          # ISO
    updated_at: str
```

**`store.py` — `MemoryStore`**：事实主存 + 向量索引，职责分离（主存为真值源，向量索引可重建）。
- 主存：SQLite（表 `memory_facts`）。
- 向量索引：复用 `FaissStore`（独立的 index/meta 文件，存 `fact_content` 的 embedding）。
- 接口：`add(fact) / update(id, fact) / soft_delete(id) / get(id) / list_active() / search(query_vector, top_k) / rebuild_index()`。
- 不变式：写主存与写索引在同一方法内同步；`rebuild_index()` 可从主存重建 FAISS。

**`extractor.py` — `MemoryExtractor`**：从一段文本（对话轮/笔记）抽事实三元组。
- 复用 OpenAI 兼容 client；prompt 含：四道过滤（本人/自我披露/说自己/有实质内容）、相对时间→绝对日期规范化、可见性分级。
- 输出：`list[ExtractedFact(fact_object, fact_content, visibility)]`。
- 健壮解析：兼容 `{"facts":[...]}` / 裸数组 / 代码围栏。空输入与解析失败返回 `[]`。

**`merger.py` — `MemoryMerger`**：给"新事实"决定 add/update/delete。
- 对每条新事实，用其 embedding 在 `MemoryStore` 召回 top-K 相似旧记忆。
- LLM 输出操作集 `[{type: add|update|delete, id?, fact_content, fact_object, visibility}]`。
- 应用到 store：add→新建；update→改正文/时间；delete→软删。
- 兜底：解析失败时退化为"全部 add"（不丢信息），并记录告警。

**`recall.py` — `recall_memories(query, top_k)`**：embed query → `store.search` → 返回 active 记忆列表（供注入）。

### 2. Agent 路由层 `app/agent/`

**`router.py` — `route(turn, history) -> RouteDecision`**：
- LLM 轻量分类，返回 `{use_docs: bool, use_memory: bool, write_memory: bool, rewritten_query: str}`。
- 规则兜底：解析失败时默认 `use_docs=True, use_memory=True, write_memory=True`（安全保守）。

**`agent.py` — `MemoryAgent.chat(user_msg) -> AgentResult`**：编排一轮：
1. `route()` 决策。
2. 若 `use_memory`：`recall.recall_memories`。
3. 若 `use_docs`：现有 hybrid 检索 + reranker。
4. 生成：把"关于你"记忆块 + 文档证据块交给生成器。
5. 若 `write_memory`：`extractor` 抽取 → `merger` 归并入库（可同步或回合末）。
- 返回 answer + sources + recalled_memories + memory_ops（便于前端展示"记住了什么"）。

### 3. 生成器改造 `app/generator/generator.py`
- `generate(query, retrieved_chunks, user_memories=None)`：新增可选 `user_memories`，在 system/user prompt 里加 `[关于用户的已知信息]` 块，明确"记忆是关于用户的背景，文档是回答证据"，证据不足仍拒答。向后兼容（不传 memory 时行为不变）。

### 4. OCR 接入 `app/loader/image_loader.py`
- 用已声明依赖 `rapidocr-onnxruntime` 对图片做 OCR，把识别文字写进 entry 的 `text`（替换占位 `Image source: ...`）。
- 失败/无文字降级：保留原占位文本，`modality` 仍为 image，CLIP 路不受影响。
- 让图片文字进入 BM25 + embedding，真正可检索。

### 5. Reranker 接入主链路
- 在 `scripts/query_demo.py` 与 `web/streamlit_app.py` 的 hybrid（及 vector）检索后，加可选 `--rerank` / 复选框：用 `BGEReranker.rerank(query, retrieved, top_k)` 精排。
- 召回阶段取较大候选（如 vector_k/bm25_k=40 融合后取 top-N），reranker 精排到 top_k。
- 默认开关可关（CPU 上较慢），但提供端到端可用路径，使"RRF 融合→BGE 精排"真实成立。

### 6. 统一评测
- 复用现有 `scripts/eval_retrieval.py`（检索三路 Recall@K/MRR@K）。
- 新增 `scripts/eval_memory.py`：
  - **记忆召回**：自建 `queries→预期记忆id` 标注集，算 Recall@K / MRR@K。
  - **归并决策**：自建"旧记忆+新事实→期望操作集"用例，算 add/update/delete 准确率。
- 产出真实数字回填简历占位 `[]`（检索 hybrid 现有：Recall@1=0.5294 / MRR@4=0.7794）。

### 7. 对话入口
- 新增 `scripts/chat_demo.py`（CLI 多轮）+ Streamlit 增加 "Chat" 标签，展示回答、引用、召回到的记忆、本轮记忆写入操作。

## 数据流（一轮对话）

```
user_msg → router → (recall memory) + (hybrid retrieve + rerank docs)
        → generator(docs evidence + user memory) → answer
        → extractor(user_msg) → merger(recall similar → LLM ops) → store(add/update/delete)
```

## 错误处理

- LLM 抽取/归并/路由解析失败：各自有保守兜底（空列表 / 全 add / 全开），不崩溃、不丢用户信息。
- OCR 失败：降级占位文本。
- 记忆向量维度需与文档索引一致（同一 embedding 模型）；store 校验维度。
- SQLite 与 FAISS 不一致时，`rebuild_index()` 以 SQLite 为准重建。

## 测试策略

- 引入 `pytest`（加入 requirements，新建 `tests/`）。
- 单元测试（不打真实 LLM/网络，mock client）：
  - `MemoryStore`：add/update/soft_delete/list_active/search/rebuild_index、维度校验。
  - `MemoryExtractor`：响应解析三种格式、空输入、解析失败兜底。
  - `MemoryMerger`：add/update/delete 应用正确、解析失败退化为 add。
  - `router`：解析与兜底。
  - `image_loader` OCR：mock OCR 引擎，验证 text 写入与失败降级。
- 集成 smoke：用假 client 跑一轮 `MemoryAgent.chat`，断言记忆被写入、召回被注入。

## 简历 claim 映射（做完后逐条为真）

| 简历句 | 由哪个组件落实 |
|---|---|
| 图片（OCR）接入 | image_loader OCR |
| RRF 融合 + BGE Reranker 精排 | reranker 接入主链路 |
| 从对话沉淀长期记忆 | extractor + merger + store |
| 个人 AI Agent（自主决策） | agent/router |
| 检索与记忆的统一评测 | eval_retrieval + eval_memory |
| 混合检索最优 (Recall@1/MRR@4) | 真实 eval 数字回填 |

## 非目标（YAGNI）

- 不做受众可见性过滤、主动推送、prompt 管理平台、灰度、多用户、并发服务化。
- non_self 关系事实抽取暂不做（单用户场景价值低），只做 self 事实。
