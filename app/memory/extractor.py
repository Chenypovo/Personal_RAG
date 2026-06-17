from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

# (system_prompt, user_prompt) -> raw model text
CompleteFn = Callable[[str, str], str]


@dataclass
class ExtractedFact:
    fact_content: str
    fact_object: str = ""
    visibility: str = "PUBLIC"


_SYSTEM_PROMPT = (
    "You extract durable personal facts about the USER from their own messages or notes, "
    "for a long-term memory system. Apply these filters, only emit a fact if ALL hold:\n"
    "1. Speaker is the user themselves (self-disclosure), not a quote or someone else.\n"
    "2. It is a statement about the user, not a question, request, or fleeting reaction.\n"
    "3. It has durable substance (preferences, facts, plans, relationships, events), "
    "not a vague momentary mood.\n"
    "4. Time expressions count as substance and must be kept.\n\n"
    "Normalize relative time to absolute dates using the provided MESSAGE TIME and write the "
    "date into fact_content (e.g. 'today' -> the message date).\n"
    "Classify visibility as PRIVATE only if the fact is explicitly sensitive (health, finances, "
    "romance/sex, precise location, contact details, legal/political/religious views); otherwise PUBLIC.\n"
    "Return ONLY JSON: {\"facts\": [{\"fact_object\": str, \"fact_content\": str, \"visibility\": "
    "\"PUBLIC\"|\"PRIVATE\"}]}. fact_content must not contain the subject pronoun. "
    "If nothing qualifies, return {\"facts\": []}."
)


def _normalize_visibility(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().upper()
        if v in {"PUBLIC", "PRIVATE"}:
            return v
        if v == "1":
            return "PUBLIC"
        if v == "2":
            return "PRIVATE"
    if value == 1:
        return "PUBLIC"
    if value == 2:
        return "PRIVATE"
    return "PUBLIC"


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # drop first fence line (``` or ```json) and trailing fence
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _coerce_fact_list(parsed: Any) -> List[dict]:
    # {"facts": [...]}
    if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
        return [f for f in parsed["facts"] if isinstance(f, dict)]
    # bare [...]
    if isinstance(parsed, list):
        # could be [{...}] or nested [{"facts": [...]}]
        out: List[dict] = []
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("facts"), list):
                out.extend(f for f in item["facts"] if isinstance(f, dict))
            elif isinstance(item, dict):
                out.append(item)
        return out
    return []


class MemoryExtractor:
    def __init__(self, complete_fn: CompleteFn) -> None:
        self.complete_fn = complete_fn

    def extract(
        self,
        text: str,
        source: str = "chat",
        message_time: Optional[str] = None,
    ) -> List[ExtractedFact]:
        content = (text or "").strip()
        if not content:
            return []

        user_prompt_parts = []
        if message_time:
            user_prompt_parts.append(f"MESSAGE TIME: {message_time}")
        user_prompt_parts.append(f"SOURCE: {source}")
        user_prompt_parts.append("CONTENT:\n" + content)
        user_prompt = "\n\n".join(user_prompt_parts)

        try:
            raw = self.complete_fn(_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(_strip_code_fences(raw))
        except Exception:
            return []

        facts: List[ExtractedFact] = []
        for item in _coerce_fact_list(parsed):
            fc = str(item.get("fact_content", "")).strip()
            if not fc:
                continue
            facts.append(
                ExtractedFact(
                    fact_content=fc,
                    fact_object=str(item.get("fact_object", "")).strip(),
                    visibility=_normalize_visibility(item.get("visibility")),
                )
            )
        return facts
