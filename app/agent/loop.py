from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.agent import GenerateFn
from app.agent.registry import ToolRegistry
from app.agent.tools.base import ToolResult
from app.agent.trajectory import TrajectoryStep
from app.memory.extractor import CompleteFn
from app.memory.merger import MergeOp
from app.memory.models import MemoryFact
from app.memory.recall import format_memory_block


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


@dataclass
class Decision:
    thought: str = ""
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    plan: Optional[List[str]] = None
    final_answer: bool = False


def parse_decision(raw: str) -> Optional[Decision]:
    """防御式解析每步决策；解析不出可执行动作一律返回 None（观察 format error）。

    字段优先级 final_answer > plan > tool；plan 可与 tool 同时出现。
    """
    try:
        parsed = json.loads(_strip_code_fences(raw or ""))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    thought = str(parsed.get("thought", "")).strip()
    raw_plan = parsed.get("plan")
    plan = [str(p).strip() for p in raw_plan if str(p).strip()] if isinstance(raw_plan, list) else None

    if parsed.get("final_answer"):
        return Decision(thought=thought, plan=plan, final_answer=True)

    tool = parsed.get("tool")
    if isinstance(tool, str) and tool.strip():
        args = parsed.get("args")
        return Decision(thought=thought, tool=tool.strip(),
                        args=args if isinstance(args, dict) else {}, plan=plan)

    if plan is not None:
        return Decision(thought=thought, plan=plan)
    return None


@dataclass
class AgentResult:
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[TrajectoryStep] = field(default_factory=list)
    recalled_memories: List[MemoryFact] = field(default_factory=list)
    memory_ops: List[MergeOp] = field(default_factory=list)
    stop_reason: str = "final_answer"   # "final_answer" | "budget" | "parse_abort" | "loop_abort"
    plan: List[str] = field(default_factory=list)


_SYSTEM_PROMPT_TEMPLATE = """You are the control loop of a personal assistant agent. Solve the user's \
request step by step by calling tools and reading their observations.

Available tools:
{tools}

Terminating action (not a tool):
- final_answer: signal that the collected evidence is sufficient. The system will then \
compose the final grounded answer from everything you retrieved so far.

Rules:
- Take exactly ONE action per step.
- Return ONLY a JSON object, no prose, in one of these shapes:
  {{"thought": "...", "tool": "<tool name>", "args": {{...}}}}
  {{"thought": "...", "plan": ["todo 1", "todo 2"]}}
  {{"thought": "...", "final_answer": true}}
- You may include "plan" alongside "tool" to update your todo list while acting.
- For multi-part questions, retrieve evidence per sub-question with separate retrieve_docs \
calls, each with a different standalone query.
- If an observation reports an error, fix your next call (different tool, corrected args, \
or a reformulated query) instead of repeating the same call.
- Use final_answer only when the evidence is sufficient; if it is not, retrieve more first.
"""


class ToolAgent:
    """工具调用 agent 循环：每步一个动作，观察结果，失败反思重试，直到收尾或预算耗尽。

    全依赖注入（complete_fn / registry / generate_fn），可离线单测；
    工具调用全程 prompt-based JSON，不依赖 provider 原生 function calling。
    """

    def __init__(
        self,
        complete_fn: CompleteFn,
        registry: ToolRegistry,
        generate_fn: GenerateFn,
        max_steps: int = 6,
        max_tool_calls: int = 8,
        max_parse_retries: int = 2,
        max_duplicate_calls: int = 2,
        history_max_turns: int = 6,
    ) -> None:
        self.complete_fn = complete_fn
        self.registry = registry
        self.generate_fn = generate_fn
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_parse_retries = max_parse_retries
        self.max_duplicate_calls = max_duplicate_calls
        self.history_max_turns = max(int(history_max_turns), 0)
        self.history: List[tuple[str, str]] = []
        self.system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tools=registry.render_tools())

    def _format_history(self) -> str:
        lines: List[str] = []
        for user, assistant in self.history[-self.history_max_turns:]:
            lines.append(f"用户: {user}")
            lines.append(f"助手: {assistant}")
        return "\n".join(lines)

    def _render_user_prompt(self, user_msg: str, plan: List[str], steps: List[TrajectoryStep]) -> str:
        parts: List[str] = []
        history = self._format_history()
        if history:
            parts.append("Conversation so far:\n" + history)
        parts.append("User request:\n" + user_msg)
        if plan:
            parts.append("Current plan:\n" + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(plan)))
        if steps:
            rendered = []
            for s in steps:
                if s.tool:
                    rendered.append(
                        f"step {s.step_index}: called {s.tool} args={json.dumps(s.args, ensure_ascii=False)}\n"
                        f"  observation: {s.observation}"
                    )
                else:
                    rendered.append(f"step {s.step_index}: {s.observation}")
            parts.append("Executed steps and observations:\n" + "\n\n".join(rendered))
        parts.append("Decide the next single action. Return ONLY JSON.")
        return "\n\n".join(parts)

    def chat(self, user_msg: str) -> AgentResult:
        msg = (user_msg or "").strip()
        if not msg:
            return AgentResult(answer="", stop_reason="final_answer")

        plan: List[str] = []
        steps: List[TrajectoryStep] = []
        acc_chunks: List[Dict[str, Any]] = []
        seen_chunk_keys: set = set()
        acc_memories: List[MemoryFact] = []
        seen_memory_ids: set = set()
        acc_ops: List[MergeOp] = []
        seen_calls: set = set()
        parse_fails = 0
        tool_calls = 0
        consecutive_dups = 0

        def finish(reason: str) -> AgentResult:
            # 任何终止路径都走同一条合成：受控/带引用/证据不足拒答由 generate_fn 保证
            try:
                gen = self.generate_fn(msg, acc_chunks, format_memory_block(acc_memories))
            except Exception:
                gen = {"answer": "抱歉，这一轮处理出错，暂时无法回答。", "sources": []}
            answer = str(gen.get("answer", ""))
            if self.history_max_turns > 0:
                self.history.append((msg, answer))
                self.history = self.history[-self.history_max_turns:]
            return AgentResult(
                answer=answer,
                sources=gen.get("sources", []) or [],
                trajectory=steps,
                recalled_memories=acc_memories,
                memory_ops=acc_ops,
                stop_reason=reason,
                plan=plan,
            )

        for step_index in range(self.max_steps):
            if tool_calls >= self.max_tool_calls:
                return finish("budget")

            raw = self.complete_fn(self.system_prompt, self._render_user_prompt(msg, plan, steps))
            decision = parse_decision(raw)

            if decision is None:
                parse_fails += 1
                steps.append(TrajectoryStep(
                    step_index=step_index,
                    observation="format error: return ONLY one valid JSON action object",
                    ok=False, error="invalid decision JSON",
                ))
                if parse_fails > self.max_parse_retries:
                    return finish("parse_abort")
                continue
            parse_fails = 0

            if decision.plan is not None:
                plan = decision.plan
            if decision.final_answer:
                steps.append(TrajectoryStep(step_index=step_index, thought=decision.thought,
                                            observation="final_answer"))
                return finish("final_answer")
            if decision.tool is None:      # plan-only 更新
                steps.append(TrajectoryStep(step_index=step_index, thought=decision.thought,
                                            observation="plan updated"))
                continue

            args = dict(decision.args)
            if decision.tool == "write_memory":
                args.setdefault("text", msg)

            call_key = (decision.tool, json.dumps(args, sort_keys=True, ensure_ascii=False))
            if call_key in seen_calls:
                consecutive_dups += 1
                steps.append(TrajectoryStep(
                    step_index=step_index, thought=decision.thought, tool=decision.tool, args=args,
                    observation="duplicate call: this exact call was already made; "
                                "take a different action or final_answer",
                    ok=False, error="duplicate call",
                ))
                if consecutive_dups > self.max_duplicate_calls:
                    return finish("loop_abort")
                continue
            consecutive_dups = 0
            seen_calls.add(call_key)

            start = time.time()
            result = self.registry.dispatch(decision.tool, args)
            tool_calls += 1
            _accumulate(decision.tool, result, acc_chunks, seen_chunk_keys,
                        acc_memories, seen_memory_ids, acc_ops)
            steps.append(TrajectoryStep(
                step_index=step_index, thought=decision.thought, tool=decision.tool, args=args,
                observation=result.content if result.ok else f"error: {result.error}",
                ok=result.ok, error=result.error,
                latency_ms=(time.time() - start) * 1000.0,
            ))
            # result.ok=False 不特殊处理：错误已进观察，下一轮模型自行纠正（反思）

        return finish("budget")


def _accumulate(
    tool: str,
    result: ToolResult,
    acc_chunks: List[Dict[str, Any]],
    seen_chunk_keys: set,
    acc_memories: List[MemoryFact],
    seen_memory_ids: set,
    acc_ops: List[MergeOp],
) -> None:
    """按工具类型把结构化载荷归集到终局合成用的证据池（去重）。"""
    if not result.ok:
        return
    if tool == "retrieve_docs":
        for c in result.data.get("chunks", []):
            meta = c.get("metadata", {}) if isinstance(c, dict) else {}
            key = (str(meta.get("source", "")), str(meta.get("chunk_id", id(c))))
            if key not in seen_chunk_keys:
                seen_chunk_keys.add(key)
                acc_chunks.append(c)
    elif tool == "read_memory":
        for f in result.data.get("memories", []):
            fid = getattr(f, "id", None) or id(f)
            if fid not in seen_memory_ids:
                seen_memory_ids.add(fid)
                acc_memories.append(f)
    elif tool == "write_memory":
        acc_ops.extend(result.data.get("ops", []))
