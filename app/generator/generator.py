import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.config import get_provider_config, get_settings


class OpenAICompatibleGenerator:
    def __init__(self, model: Optional[str] = None) -> None:
        settings = get_settings()
        cfg = get_provider_config(settings, settings.llm_provider)
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self.model = model or settings.llm_model

    @staticmethod
    def _citation_label(meta: Dict[str, Any]) -> str:
        source = str(meta.get("source", "unknown"))
        source_name = os.path.basename(source) if source not in {"", "unknown"} else "unknown"
        chunk_id = meta.get("chunk_id", "NA")
        return f"Chunk {chunk_id}, {source_name}"

    def _build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            label = self._citation_label(meta)
            text_raw = meta.get("text", "")
            text = text_raw if isinstance(text_raw, str) else ("" if text_raw is None else str(text_raw))
            text = text.strip()
            lines.append(f"[{label}]\n{text}")
        return "\n\n".join(lines).strip()

    def generate(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        context = self._build_context(retrieved_chunks)

        system_prompt = (
            "你是一个RAG问答助手。"
            "请优先基于给定上下文回答。"
            "如果上下文不足，可补充常识，但必须明确哪些结论来自上下文。"
            "引用格式必须是 [Chunk x, filename]，只允许引用提供给你的chunk，不要把chunk内出现的网址当作引用。"
        )
        user_prompt = (
            f"用户问题：\n{query}\n\n"
            f"检索上下文：\n{context}\n\n"
            "请给出简洁、准确的中文回答，并在关键结论后附上chunk引用。"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content if resp.choices else ""
        sources: List[Dict[str, Any]] = []
        for item in retrieved_chunks:
            meta_raw = item.get("metadata")
            meta = meta_raw if isinstance(meta_raw, dict) else {}
            sources.append(
                {
                    "citation": self._citation_label(meta),
                    "source": meta.get("source"),
                    "chunk_id": meta.get("chunk_id"),
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                }
            )
        return {"answer": answer, "sources": sources}
