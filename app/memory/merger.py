from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

from app.memory.extractor import CompleteFn, ExtractedFact
from app.memory.models import MemoryFact
from app.memory.store import MemoryStore


@dataclass
class MergeOp:
    type: str  # "add" | "update" | "delete"
    fact_content: str = ""
    fact_object: str = ""
    visibility: str = "PUBLIC"
    id: str = ""
    applied: bool = False


_SYSTEM_PROMPT = (
    "You maintain a personal long-term memory. Given EXISTING memories (each with an id) and "
    "NEW candidate facts (no id), decide operations to keep memory deduplicated and current.\n"
    "- add: a genuinely new fact.\n"
    "- update: a NEW fact that refines/replaces an EXISTING one (reference its id).\n"
    "- delete: an EXISTING fact now contradicted or obsolete (reference its id).\n"
    "Do not duplicate facts already present. Return ONLY JSON: "
    '{"operations": [{"type": "add|update|delete", "id": "<existing id, for update/delete>", '
    '"fact_object": str, "fact_content": str, "visibility": "PUBLIC|PRIVATE"}]}'
)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _normalize_visibility(value, default: str = "PUBLIC") -> str:
    if isinstance(value, str) and value.strip().upper() in {"PUBLIC", "PRIVATE"}:
        return value.strip().upper()
    if value == 2:
        return "PRIVATE"
    if value == 1:
        return "PUBLIC"
    return default


class MemoryMerger:
    def __init__(self, store: MemoryStore, complete_fn: CompleteFn, recall_k: int = 5) -> None:
        self.store = store
        self.complete_fn = complete_fn
        self.recall_k = recall_k

    def _recall_existing(self, new_facts: List[ExtractedFact]) -> List[MemoryFact]:
        seen: Dict[str, MemoryFact] = {}
        for nf in new_facts:
            for fact, _score in self.store.search(nf.fact_content, top_k=self.recall_k):
                seen[fact.id] = fact
        return list(seen.values())

    def _build_user_prompt(self, existing: List[MemoryFact], new_facts: List[ExtractedFact]) -> str:
        existing_payload = [
            {"id": f.id, "fact_object": f.fact_object, "fact_content": f.fact_content, "visibility": f.visibility}
            for f in existing
        ]
        new_payload = [
            {"fact_object": f.fact_object, "fact_content": f.fact_content, "visibility": f.visibility}
            for f in new_facts
        ]
        return (
            "EXISTING memories:\n"
            + json.dumps(existing_payload, ensure_ascii=False, indent=2)
            + "\n\nNEW candidate facts:\n"
            + json.dumps(new_payload, ensure_ascii=False, indent=2)
        )

    def _add_all(self, new_facts: List[ExtractedFact], source: str) -> List[MergeOp]:
        ops: List[MergeOp] = []
        for nf in new_facts:
            self.store.add(
                MemoryFact(
                    fact_content=nf.fact_content,
                    fact_object=nf.fact_object,
                    visibility=nf.visibility,
                    source=source,
                )
            )
            ops.append(
                MergeOp(type="add", fact_content=nf.fact_content, fact_object=nf.fact_object,
                        visibility=nf.visibility, applied=True)
            )
        return ops

    def merge(self, new_facts: List[ExtractedFact], source: str = "chat") -> List[MergeOp]:
        if not new_facts:
            return []

        existing = self._recall_existing(new_facts)
        user_prompt = self._build_user_prompt(existing, new_facts)

        try:
            raw = self.complete_fn(_SYSTEM_PROMPT, user_prompt)
            parsed = json.loads(_strip_code_fences(raw))
            operations = parsed.get("operations") if isinstance(parsed, dict) else None
            if not isinstance(operations, list):
                raise ValueError("missing operations list")
        except Exception:
            # conservative fallback: never lose newly extracted info
            return self._add_all(new_facts, source)

        existing_ids = {f.id for f in existing}
        applied_ops: List[MergeOp] = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("type", "")).strip().lower()
            op_id = str(op.get("id", "")).strip()
            content = str(op.get("fact_content", "")).strip()
            obj = str(op.get("fact_object", "")).strip()
            vis = _normalize_visibility(op.get("visibility"))

            record = MergeOp(type=op_type, fact_content=content, fact_object=obj, visibility=vis, id=op_id)

            if op_type == "add" and content:
                self.store.add(MemoryFact(fact_content=content, fact_object=obj, visibility=vis, source=source))
                record.applied = True
            elif op_type == "update" and op_id in existing_ids:
                self.store.update(
                    op_id,
                    fact_content=content or None,
                    fact_object=obj or None,
                    visibility=vis,
                )
                record.applied = True
            elif op_type == "delete" and op_id in existing_ids:
                self.store.soft_delete(op_id)
                record.applied = True

            applied_ops.append(record)

        return applied_ops
