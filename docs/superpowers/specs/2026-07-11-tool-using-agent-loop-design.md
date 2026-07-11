# 设计文档:工具调用 Agent 循环（Tool-Using Agent Loop）

> 日期：2026-07-11
> 状态：已通过产品评审，待实现
> 交接方式：本文档自包含，实现方（另一个 Claude Code）仅凭此文档即可拆计划并执行，无需原始对话上下文。

---

## 0. TL;DR（给实现方）

把现有的「一次 LLM 调用出 3 个开关」的路由（`app/agent/router.py`）升级成一个**真正的工具调用 agent 循环**：agent 每步自己决定调哪个工具、看结果、更新一个轻量待办计划、遇到失败/证据不足就反思重试，直到收尾。工具调用**全程 prompt-based JSON**（不碰 provider 原生 function calling），所有外部能力**依赖注入**，因此整条循环可在**无网络**下用假 LLM + 假工具单测。

交付三样：
1. `ToolAgent` 循环 + 工具注册表 + 4 个可分发工具（`retrieve_docs` / `read_memory` / `write_memory` / `calculator`）+ 1 个**终止动作** `final_answer`（在 prompt 里向模型广告为"可用动作"，但由循环特判处理、**不**走注册表分发，见 4.3/4.6）。
2. 失败处理与护栏（非法 JSON、未知工具、参数错、步数预算、死循环检测、兜底回答）+ 轨迹日志。
3. 离线评测 `scripts/eval_agent.py`：基线（单次）vs 循环的**任务成功率**、**多跳 Coverage@k**、**成本（调用/步数）**。

---

## 1. 背景与目标

### 1.1 现状
现有 `MemoryAgent.chat`（`app/agent/agent.py`）单轮流程是「路由 → 召回+检索 → 生成 → 学习」。`Router`（`app/agent/router.py`）只做**一次** LLM 调用，输出 `{use_docs, use_memory, write_memory, rewritten_query}` 四个字段。它**不会多步、不会调用异构工具、不会看结果再决策、不会失败重试**。项目面试清单第 4 题已诚实承认：「没有 tool/skill 管理，只有固定三动作路由」。

### 1.2 目标
面向 **agent 开发岗**，把这个固定路由升级成一个真实的、可讲清楚、可度量的工具调用 agent，补上「工具/技能管理 + 多步控制循环 + 失败处理」这块能力，同时保住项目已有的三条招牌：**provider 无关、依赖注入可测试、凡事有度量**。

### 1.3 简历目标句（衡量成败的北极星）
> 「把一个固定路由升级成真正的工具调用 agent：自研控制循环 + 工具注册表 + 失败重试与步数预算 + 轨迹日志，全程 prompt-based、provider 无关、可离线单测；用离线评测证明多步任务成功率与多跳覆盖率优于单次基线。」

若最终产物不能支撑这句话（尤其是「有数字」），即为未达标。

---

## 2. 范围

### 2.1 In scope（本期做）
- 一个智能体控制循环 `ToolAgent`（下称"循环"）。
- 工具接口 `Tool` + 工具注册表 `ToolRegistry`。
- 5 个工具：`retrieve_docs`、`read_memory`、`write_memory`、`calculator`、`final_answer`。
- 轻量待办计划（plan）+ 失败反思重试。
- 护栏：非法 JSON、未知工具、参数校验、步数/工具调用预算、死循环检测、兜底回答。
- 轨迹（trajectory）结构化记录，随结果返回。
- 离线评测脚本 + 小评测集（含多跳、记忆依赖、计算三类任务）。
- 单元测试（无网络，注入假 LLM / 假工具）。

### 2.2 Non-goals（本期明确不做，写清避免实现方跑偏）
- **不**用 provider 原生 function calling（保持 provider 无关；理由见面试清单 Q94）。
- **不**做多智能体（multi-agent）。
- **不**接联网工具（web search 等）——`calculator` 作为"异构工具"的离线演示，联网留 Phase 2。
- **不**做图记忆（graph memory）——已在产品评审阶段否决，记忆模块保持现状。
- **不**做流式输出（streaming）——Phase 2。
- **不**动 `loader/chunker/embedder/vectordb/retriever/reranker/generator/memory` 的内部实现；只**包装**它们成工具。
- **不**删除或改写 `Router`、现有 `MemoryAgent`（保留作为"单次基线"，评测要用；现有测试不能挂）。

### 2.3 循环架构选型（"cutting edge" 的落地解释）
采用**简单智能体循环 + 模型自驱**范式（现代共识是"简单循环胜过重框架"），具体是 **ReAct 式单步决策 + 轻量待办计划 + 失败反思**的融合体：
- 每步模型只做一个动作：调一个工具，或收尾。
- 模型维护一个可覆盖的 `plan`（待办列表），支持中途重规划。
- 工具失败 / 证据不足时，把错误作为观察喂回，模型下一步自行纠正。
- 这一形态天然覆盖 ReAct、链式多跳（依赖步骤）、自我纠错三种行为，但**不引入** plan-execute-replan 的重编排框架。

---

## 3. 架构总览

```
用户消息
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ ToolAgent 循环 (app/agent/loop.py)                       │
│                                                          │
│  state = {user_msg, history, plan=[], steps=[]}          │
│  repeat (≤ max_steps):                                   │
│    1) 渲染 prompt（系统指令 + 工具清单 + history +         │
│       plan + 已有 steps 观察）                            │
│    2) complete_fn(prompt) -> raw                         │
│    3) 防御式解析 JSON -> decision                         │
│    4) decision 含 final_answer? → 走终局合成，返回         │
│       decision 含 plan?         → 覆盖 state.plan          │
│       否则 tool+args           → registry.dispatch        │
│    5) 记录 TrajectoryStep（thought/tool/args/result）     │
│    6) result 失败 → 观察带错误，进入下一轮（反思重试）      │
│  预算耗尽 → 兜底回答                                       │
└─────────────────────────────────────────────────────────┘
   │dispatch                       │final
   ▼                               ▼
┌──────────────┐          ┌──────────────────────────┐
│ ToolRegistry │          │ 终局合成（复用现成生成器） │
│  校验 + 分发  │          │ generate_fn(query,        │
└──────┬───────┘          │  累积chunks, memory_block)│
       │                  │ → 受控/带引用/证据不足拒答 │
   ┌───┴───────────────┐  └──────────────────────────┘
   ▼   ▼   ▼   ▼   ▼
 retrieve read  write calc final
 _docs   _memory _mem       _answer
   │       │      │
   ▼       ▼      ▼
 现成检索  现成召回 现成 extractor+merger（非破坏归并）
```

**关键设计点：evidence 收集与终局合成分离。**
- agent 用工具**收集证据**（多次 `retrieve_docs` 不同子查询 = 多跳/拆解自然涌现；`read_memory` 拿用户事实）。
- agent 决定"够了"时调 `final_answer`（**信号:收尾**），由编排层用**现成 `generate_fn`** 对累积的全部 chunks + 记忆块做**受控、带引用、证据不足则拒答**的合成。
- 这样保住项目已有的"受控生成 + 引用 + 拒答"保证，且把"判断该不该收尾"（agent 职责）与"落地合成"（生成器职责）解耦。

---

## 4. 组件与接口

> 语言 Python 3.9+（与仓库一致，注意 `from __future__ import annotations`）。风格沿用仓库：`@dataclass` + 依赖注入 + 防御式 JSON 解析（参考 `app/agent/router.py` 的 `_strip_code_fences` / try-except / 缺字段默认）。

### 4.1 工具接口 `Tool`（`app/agent/tools/base.py`）

```python
@dataclass
class ToolResult:
    ok: bool
    content: str                       # 喂回模型的文本观察（人读的摘要）
    data: Dict[str, Any] = field(default_factory=dict)  # 结构化载荷（chunks/memories/ops/数值）
    error: Optional[str] = None        # ok=False 时的错误说明（也会进观察）

class Tool(Protocol):
    name: str
    description: str                   # 进系统 prompt 的一句话说明
    args_schema: Dict[str, Any]        # 参数说明（名→类型/含义），渲染进 prompt 指导模型
    def run(self, args: Dict[str, Any]) -> ToolResult: ...
```

- `content` 是给模型看的自然语言观察；`data` 是给编排层用的结构化数据（例如终局合成要用的累积 chunks）。
- 工具内部**自己 try/except**，任何异常都转成 `ToolResult(ok=False, error=...)`，绝不抛出中断循环。

### 4.2 工具注册表 `ToolRegistry`（`app/agent/registry.py`）
职责：
- 持有 `name -> Tool`。
- `render_tools() -> str`：把所有工具的 `name/description/args_schema` 渲染成一段供系统 prompt 用的清单文本。
- `dispatch(name, args) -> ToolResult`：
  - 未知工具名 → `ToolResult(ok=False, error="unknown tool: ...")`。
  - 参数缺失/类型不符 → `ToolResult(ok=False, error="bad args: ...")`（**不**信任模型，注册表侧校验）。
  - 正常 → 调 `tool.run(args)`，把异常也兜成 `ok=False`。

### 4.3 五个工具

| 工具 | 参数 | 内部实现（包装现成能力） | content 观察 | data |
| --- | --- | --- | --- | --- |
| `retrieve_docs` | `query: str`, `k?: int` | 注入的 `retrieve_docs_fn(query)`（现成 hybrid+rerank+parent-child） | 命中 chunk 的编号+摘要+来源 | `{"chunks": [...]}` |
| `read_memory` | `query: str`, `k?: int` | `recall_memories(store, query, top_k)`（现成） | 召回的用户事实文本块 | `{"memories": [...]}` |
| `write_memory` | `text?: str`（缺省用本轮 user_msg） | `extractor.extract(...)` + `merger.merge(...)`（现成，非破坏归并/相似度门控/supersede） | 归并操作摘要（add/update/delete 各几条） | `{"ops": [...]}` |
| `calculator` | `expression: str` | **AST 安全求值**（见 4.4），禁 `eval` | 计算结果或错误 | `{"result": <number>}` |

**`final_answer` 是终止动作，不是可分发工具**：它在系统 prompt 的可用动作清单里出现（让模型知道可以收尾），但循环里**特判**（`decision.final_answer` 为真即走终局合成），**不**经 `ToolRegistry.dispatch`，也**不**建 `tools/final_answer.py`。注册表只装 4 个真工具。

- `retrieve_docs` 可被**多次**调用（不同子查询）——这是多跳/拆解的实现方式，不需要单独的"decompose"工具；拆解体现在 agent 连续发起多个 `retrieve_docs`。
- `write_memory` 严格复用现有 `MemoryExtractor` / `MemoryMerger`，**保持非破坏 + 门控**，本期不改其逻辑。

### 4.4 `calculator` 安全求值
- **禁止** `eval()` / `exec()`。用 `ast.parse(expr, mode="eval")` + 白名单遍历：仅允许 `+ - * / // % **`、一元正负、括号、数字字面量。
- 除零 → `ToolResult(ok=False, error="division by zero")`。
- 出现名称/调用/属性/下标等非算术节点 → `ok=False, error="unsupported expression"`。
- 目的：演示"异构工具"且**确定性可单测**。

### 4.5 循环 `ToolAgent`（`app/agent/loop.py`）

构造（全依赖注入，可离线测）：
```python
class ToolAgent:
    def __init__(
        self,
        complete_fn: CompleteFn,          # (system_prompt, user_prompt) -> str，与 Router 同签名
        registry: ToolRegistry,
        generate_fn: GenerateFn,          # 终局合成，签名同现有 (query, chunks, memory_block) -> {answer, sources}
        max_steps: int = 6,
        max_tool_calls: int = 8,          # 与 max_steps 独立的总调用上限
        max_parse_retries: int = 2,       # 连续 JSON 解析失败上限
        history_max_turns: int = 6,
    ): ...
    def chat(self, user_msg: str) -> AgentResult: ...
```

`AgentResult`（可扩展现有 `app/agent/agent.py` 的 `AgentResult`，或新建；至少含）：
```python
@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]]
    trajectory: List[TrajectoryStep]     # 全过程
    recalled_memories: List[MemoryFact]  # read_memory 累积
    memory_ops: List[MergeOp]            # write_memory 累积
    stop_reason: str                     # "final_answer" | "budget" | "parse_abort" | "loop_abort"
    plan: List[str]                      # 终态计划
```

### 4.6 每步决策的 JSON 协议
模型每步**只返回**下列之一（防御式解析，允许 ```json 围栏）：
```json
{"thought": "...", "tool": "retrieve_docs", "args": {"query": "..."}}
{"thought": "...", "plan": ["查A", "查B", "对比后回答"]}      // 可选：更新待办
{"thought": "...", "final_answer": true}
```
- 解析规则（复用/仿 `router._strip_code_fences`）：去围栏 → `json.loads` → 非 dict 报错 → 按字段优先级判定 `final_answer` > `plan` > `tool`。
- `args` 缺字段由注册表侧校验兜。
- 一次响应里 `plan` 与 `tool` 可**同时**出现（先更新计划再执行一个工具）——按"先存 plan 再执行 tool"处理。

### 4.7 系统 prompt 要点（实现方据此写，不要照抄字面）
- 说明有哪些工具（`registry.render_tools()` 注入）、每个工具何时用。
- 要求：每步只做一个动作；证据够了才 `final_answer`；证据不足优先再检索/换查询/拆子问题；工具报错要根据错误改正。
- 强约束**只输出 JSON**、给出上面三种 schema。
- 注入 `history`、当前 `plan`、已执行 `steps`（thought+tool+观察）作为上下文。

---

## 5. 控制流（伪代码，实现方按此写）

```python
def chat(user_msg):
    state = AgentState(user_msg, history=self._format_history(), plan=[], steps=[])
    acc_chunks, acc_memories, acc_ops = [], [], []
    parse_fails = 0
    seen_calls = set()  # (tool, canonical_args) 死循环检测

    for _ in range(self.max_steps):
        if total_tool_calls >= self.max_tool_calls: break
        raw = self.complete_fn(SYSTEM_PROMPT, render_user_prompt(state))
        decision = parse_decision(raw)          # 防御式
        if decision is None:                    # 解析失败
            parse_fails += 1
            state.steps.append(TrajectoryStep(observation="format error: return valid JSON"))
            if parse_fails > self.max_parse_retries:
                return fallback(state, acc_chunks, acc_memories, acc_ops, reason="parse_abort")
            continue
        parse_fails = 0

        if decision.plan is not None:
            state.plan = decision.plan
        if decision.final_answer:
            return synthesize(state, acc_chunks, acc_memories, acc_ops, reason="final_answer")

        key = (decision.tool, canonical(decision.args))
        if key in seen_calls:                   # 完全相同调用重复 → 注入提示，避免死循环
            state.steps.append(TrajectoryStep(observation="duplicate call; try a different action"))
            # 可选：连续重复 N 次 → loop_abort 兜底
            continue
        seen_calls.add(key)

        result = self.registry.dispatch(decision.tool, decision.args)
        total_tool_calls += 1
        _accumulate(result, acc_chunks, acc_memories, acc_ops)   # 按工具类型归集
        state.steps.append(TrajectoryStep(decision.thought, decision.tool, decision.args, result))
        # result.ok=False 时不特殊处理：错误已在观察里，下一轮模型自行纠正（反思）

    return fallback(state, acc_chunks, acc_memories, acc_ops, reason="budget")


def synthesize(state, chunks, memories, ops, reason):
    memory_block = format_memory_block(memories)
    gen = self.generate_fn(state.user_msg, chunks, memory_block)   # 受控/带引用/证据不足拒答
    return AgentResult(answer=gen["answer"], sources=gen.get("sources", []),
                       trajectory=state.steps, recalled_memories=memories,
                       memory_ops=ops, stop_reason=reason, plan=state.plan)


def fallback(state, chunks, memories, ops, reason):
    # 预算/异常兜底：用已收集到的证据尽力合成；无证据则生成器按"证据不足"拒答
    return synthesize(state, chunks, memories, ops, reason)
```

---

## 6. 与现有代码的关系

- **新增**（不动老文件内部）：
  - `app/agent/tools/base.py`、`app/agent/tools/retrieve.py`、`app/agent/tools/memory_tools.py`（read/write）、`app/agent/tools/calculator.py`（`final_answer` 不建文件，由 `loop.py` 特判）
  - `app/agent/registry.py`、`app/agent/loop.py`、`app/agent/trajectory.py`
  - `app/agent/factory.py` 里加一个装配函数（把现成 retriever/store/extractor/merger/generator 包成工具 + registry + ToolAgent），沿用现有 factory 风格。
- **保留不改**：`app/agent/router.py`、`app/agent/agent.py`（`MemoryAgent`）——作为"单次基线"，评测与既有测试都要用；现有测试全绿是硬约束。
- **API/Web**：`app/api/server.py` 可加一条开关（走 `MemoryAgent` 还是 `ToolAgent`），默认可先保持 `MemoryAgent` 不变，避免破坏现有前端；本期不强制改前端。

---

## 7. 失败处理与护栏（agent 开发岗最看重，务必都覆盖且有测试）

| 场景 | 处理 |
| --- | --- |
| LLM 返回非法 JSON | 防御解析失败 → 观察"format error" → 连续超 `max_parse_retries` → `parse_abort` 兜底 |
| 未知工具名 | 注册表 `ok=False, error` → 观察 → 模型改正 |
| 参数缺失/类型错 | 注册表侧校验 `ok=False, error` → 观察 → 模型改正 |
| 工具内部异常 | 工具 try/except 兜成 `ok=False` → 观察 → 模型改正 |
| 步数/调用超预算 | `budget` 兜底：用已收集证据合成或拒答 |
| 完全相同的重复调用 | 死循环检测：注入提示；连续重复超阈值 → `loop_abort` 兜底 |
| calculator 除零/非算术 | `ok=False, error` |

**兜底不崩**：任何终止路径都返回一个合法 `AgentResult`（答案可能是生成器的"证据不足"拒答），绝不抛异常给上层。

---

## 8. 轨迹日志 / 可观测

- `TrajectoryStep`（`app/agent/trajectory.py`）：`{step_index, thought, tool, args, observation, ok, error, latency_ms?}`。
- 全轨迹随 `AgentResult.trajectory` 返回；提供一个 `format_trajectory()` 便于命令行/日志打印。
- 这是 Phase 2 上线做 LLM-as-judge / 可观测看板的地基（本期只需结构化记录 + 可打印）。

---

## 9. 评测（差异化与"有数字"的关键，`scripts/eval_agent.py`）

### 9.1 评测集（新建，放 `data/eval/`）
在现有语料上造 ~20–30 个**需要 ≥2 次工具调用**的任务，三类：
1. **多跳检索**：答案需跨 ≥2 章节/文档 → 需多次 `retrieve_docs` 不同子查询。gold = 需要的 chunk 并集。
2. **记忆依赖**：需 `read_memory` + `retrieve_docs`（或先 `write_memory` 再 `read_memory`）。
3. **计算**：需先 `retrieve_docs` 取数值再 `calculator`。gold 含期望数值。
- 生成方式沿用现有做法（喂 `metadatas.json` + 原文给 LLM 生成任务），但 prompt 明确要"多步/复合"，再**人工抽查**。

### 9.2 指标
- **任务成功率**：尽量**程序化判定**（多跳看 gold chunk 覆盖、计算看数值命中）；需语义判定的用 LLM-as-judge，但标注为"线上/可选"，离线主指标用程序化的。
- **多跳 Coverage@k**：最终累积证据 top-k 里覆盖了多少 gold chunk（头条检索数字）。
- **工具选择正确率**：实际调用工具集 vs 期望工具集（precision/recall）。
- **成本**：每任务平均 LLM 调用数 / 工具调用数 / 步数 / 解析失败率。
- **基线对照**：同一批任务上跑**单次基线**（现成 `MemoryAgent`：路由→检索一次→生成）vs `ToolAgent`。头条：多跳覆盖率 X%→Y%、任务成功率 A%→B%、代价 +N 次调用。

### 9.3 复用
复用 `scripts/eval_retrieval.py` / `app/eval` 里的 Recall/Coverage/MRR 计算逻辑，勿重复造轮子。

---

## 10. 测试（无网络，注入假件；镜像 `tests/test_agent.py`、`tests/test_router.py` 风格）

必须覆盖：
1. **循环编排**：假 `complete_fn` 返回脚本化 JSON 序列 → 断言按序调用工具、观察进上下文、遇 `final_answer` 正确收尾并触发合成。
2. **注册表**：未知工具 / 缺参 / 类型错 → 返回 `ok=False` 且不抛异常。
3. **防御解析**：非法 JSON → 观察 format error；连续超限 → `parse_abort` 兜底返回合法结果。
4. **预算**：步数/调用超限 → `budget` 兜底。
5. **死循环检测**：重复相同调用 → 注入提示 / `loop_abort`。
6. **calculator**：合法式、除零、禁用节点（`__import__`、名称、调用）逐一断言。
7. **write_memory 工具**：用假 extractor/merger，断言返回 ops、非破坏（复用现有 memory 测试的假件模式）。
8. **集成**：`ToolAgent` + 假 LLM + 内存态真工具（tiny 语料）→ 多跳任务能连发两次 `retrieve_docs` 并覆盖两个 gold chunk。

现有全部测试必须继续通过（`pytest` 全绿）。

---

## 11. 验收标准（可勾验，缺一不可）

1. `ToolAgent.chat` 跑通完整循环，5 个工具可用，能在 `final_answer` 或预算处终止，任何路径都返回合法 `AgentResult`。
2. 全程 prompt-based JSON，`complete_fn` 可注入，**不**依赖任何 provider 原生 function calling。
3. 失败处理六类场景（第 7 节）均有对应测试且通过；新老测试 `pytest` 全绿。
4. `scripts/eval_agent.py` 可**离线**运行，打印：基线 vs 循环的任务成功率、多跳 Coverage@k、平均成本。评测集与脚本入库 `data/eval/`。
5. 轨迹随结果返回且可打印。
6. `README.md` 增补一节：agent 循环链路 + 评测结果表（基线 vs 循环）+ 诚实边界（第 12 节）。

---

## 12. 诚实边界（写进 README / 面试主动说）

- Prompt-based JSON 工具调用**不如**原生 function calling 稳，靠防御解析+重试兜；会有偶发解析失败——用**解析失败率**量化，不藏。
- 评测集小、LLM 合成，仅信**相对结论**（与现有评测口径一致）。
- 多步比单次**更慢更贵**——量化 +N 次调用，不美化。
- `calculator` 是演示工具，"异构工具"的故事真实但体量有限；联网工具留 Phase 2。
- 记忆工具复用现有非破坏归并，未在本期做记忆侧新消融。

---

## 13. 线下 → 线上（Phase 2，不在本期实现，仅记录方向）

- 接联网/外部工具（web search 等）。
- 用记录下来的轨迹做 LLM-as-judge（任务成功 + 工具选择质量）+ 轨迹看板。
- 流式：把每步 thought/action/observation 推给 `webui/`。
- 记忆写入异步化。

---

## 14. 建议实现顺序（给实现方的落地节奏）

1. `Tool`/`ToolResult`/`ToolRegistry` + `calculator`（最独立，先立骨架 + 测试）。
2. `retrieve_docs`/`read_memory`/`write_memory` 包装现成能力 + 测试。
3. `ToolAgent` 循环 + 防御解析 + 护栏 + 轨迹 + 兜底 + 测试（核心）。
4. `factory` 装配 + 终局合成接现成 `generate_fn`。
5. 评测集 + `scripts/eval_agent.py` + 基线对照。
6. README 增补 + 全量 `pytest`。
